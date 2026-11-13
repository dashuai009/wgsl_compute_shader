use bevy::camera::{primitives::Aabb, visibility::NoFrustumCulling};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::storage::ShaderBuffer;
use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix3, Vector3};

use crate::bin_asset_loader::PodArrayAsset;
use crate::material::SoftBodyMaterialExt;
use crate::mesh_gen::{
    build_bevy_surface_mesh, canonicalize_tets, compute_smooth_normals, extract_boundary_triangles,
};

pub const SUBSTEPS_PER_FRAME: u32 = 10;
pub const JACOBI_ITERS_PER_SUBSTEP: u32 = 8;
pub const FIXED_FRAME_DT: f32 = 1.0 / 60.0;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Preset {
    #[default]
    Bunny,
    Cube,
    Torus,
    Beam,
}

impl Preset {
    pub const ALL: [Preset; 4] = [Preset::Bunny, Preset::Cube, Preset::Torus, Preset::Beam];

    pub fn slug(self) -> &'static str {
        match self {
            Preset::Bunny => "bunny",
            Preset::Cube => "cube",
            Preset::Torus => "torus",
            Preset::Beam => "beam",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Preset::Bunny => "Bunny",
            Preset::Cube => "Cube",
            Preset::Torus => "Torus",
            Preset::Beam => "Beam",
        }
    }

    pub fn base_color(self) -> Color {
        match self {
            Preset::Bunny => Color::srgb(0.92, 0.52, 0.34),
            Preset::Cube => Color::srgb(0.32, 0.72, 0.92),
            Preset::Torus => Color::srgb(0.46, 0.84, 0.56),
            Preset::Beam => Color::srgb(0.90, 0.74, 0.30),
        }
    }

    pub fn enable_floor(self) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum StiffnessPreset {
    Soft,
    #[default]
    Medium,
    Stiff,
}

impl StiffnessPreset {
    pub const ALL: [StiffnessPreset; 3] = [
        StiffnessPreset::Soft,
        StiffnessPreset::Medium,
        StiffnessPreset::Stiff,
    ];

    pub fn label(self) -> &'static str {
        match self {
            StiffnessPreset::Soft => "Soft",
            StiffnessPreset::Medium => "Medium",
            StiffnessPreset::Stiff => "Stiff",
        }
    }

    /// Returns Lamé parameters `(mu, lambda)` derived from `(E, nu)`.
    pub fn lame_parameters(self) -> (f32, f32) {
        let (youngs_e, poisson_nu): (f32, f32) = match self {
            StiffnessPreset::Soft => (1.8e5, 0.35),
            StiffnessPreset::Medium => (6.5e5, 0.38),
            StiffnessPreset::Stiff => (2.6e6, 0.40),
        };
        let mu = youngs_e / (2.0 * (1.0 + poisson_nu));
        let lambda = youngs_e * poisson_nu / ((1.0 + poisson_nu) * (1.0 - 2.0 * poisson_nu));
        (mu, lambda)
    }
}

#[derive(Resource, Clone, ExtractResource, Debug)]
#[extract_app(bevy::render::RenderApp)]
pub struct SimConfig {
    pub preset: Preset,
    pub stiffness: StiffnessPreset,
    pub paused: bool,
    pub gravity_enabled: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            preset: Preset::Bunny,
            stiffness: StiffnessPreset::Medium,
            paused: false,
            gravity_enabled: true,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VertData {
    pub pos_mid_im: [f32; 4], // xyz = pos_mid, w = inv_mass
    pub pos_prev: [f32; 4],   // xyz = pos_prev
    pub vel: [f32; 4],        // xyz = vel
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TetStatic {
    pub indices: [u32; 4],
    pub b_row0: [f32; 4], // B00, B01, B02, rest_vol
    pub b_row1: [f32; 4], // B10, B11, B12, tet_omega
    pub b_row2: [f32; 4], // B20, B21, B22, 0.0
}

#[allow(dead_code)]
mod gpu_params {
    use bevy::render::render_resource::ShaderType;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(ShaderType, Default, Copy, Clone, Pod, Zeroable, Debug)]
    pub struct SimParams {
        pub dt: f32,
        pub omega: f32,
        pub mu: f32,
        pub lambda: f32,
        pub gravity_y: f32,
        pub damping: f32,
        pub floor_y: f32,
        pub friction: f32,
        pub nt: u32,
        pub nv: u32,
        pub n_surf_tris: u32,
        pub enable_floor: u32,
    }
}
pub use gpu_params::SimParams;

/// Pure-CPU prepared GPU arrays (shared by runtime asset spawning and headless GPU tests).
pub struct CpuPhysBuffers {
    pub pos_bits: Vec<u32>,
    pub normal_bits: Vec<u32>,
    pub verts: Vec<VertData>,
    pub tets: Vec<TetStatic>,
    pub tet_state: Vec<[f32; 4]>,
    pub surface_tris: Vec<u32>,
    pub canonical_tets: Vec<[u32; 4]>,
}

impl CpuPhysBuffers {
    pub fn build(
        positions: &[[f32; 3]],
        raw_tets: &[[u32; 4]],
        is_pinned: &[bool],
        density: f32,
    ) -> Self {
        assert_eq!(positions.len(), is_pinned.len());
        let nv = positions.len();
        let nt = raw_tets.len();

        let mut canonical_tets = raw_tets.to_vec();
        canonicalize_tets(positions, &mut canonical_tets);

        let surface_tris = extract_boundary_triangles(&canonical_tets);
        let init_normals = compute_smooth_normals(positions, &surface_tris);

        // Count incident tetrahedra per vertex for momentum-preserving Jacobi relaxation weight
        let mut valence = vec![0u32; nv];
        for &[i0, i1, i2, i3] in &canonical_tets {
            valence[i0 as usize] += 1;
            valence[i1 as usize] += 1;
            valence[i2 as usize] += 1;
            valence[i3 as usize] += 1;
        }

        let mut mass = vec![0.0f32; nv];
        let mut tets = Vec::with_capacity(nt);

        for &[i0, i1, i2, i3] in &canonical_tets {
            let p0 = Vector3::from(positions[i0 as usize]);
            let p1 = Vector3::from(positions[i1 as usize]);
            let p2 = Vector3::from(positions[i2 as usize]);
            let p3 = Vector3::from(positions[i3 as usize]);

            let d_m = Matrix3::from_columns(&[p1 - p0, p2 - p0, p3 - p0]);
            let det = d_m.determinant();
            assert!(
                det > 1e-10,
                "Tetrahedron has non-positive determinant {det}; check input mesh"
            );
            let rest_vol = det / 6.0;
            let b = d_m
                .try_inverse()
                .expect("invertible rest edge matrix D_m");

            let max_val = valence[i0 as usize]
                .max(valence[i1 as usize])
                .max(valence[i2 as usize])
                .max(valence[i3 as usize])
                .max(1) as f32;
            let tet_omega = (1.8 / max_val).clamp(0.08, 0.50);

            // Pack B = D_m^-1 in explicit ROW-MAJOR order matching arap.wgsl
            tets.push(TetStatic {
                indices: [i0, i1, i2, i3],
                b_row0: [b[(0, 0)], b[(0, 1)], b[(0, 2)], rest_vol],
                b_row1: [b[(1, 0)], b[(1, 1)], b[(1, 2)], tet_omega],
                b_row2: [b[(2, 0)], b[(2, 1)], b[(2, 2)], 0.0],
            });

            let quarter_mass = 0.25 * density * rest_vol;
            mass[i0 as usize] += quarter_mass;
            mass[i1 as usize] += quarter_mass;
            mass[i2 as usize] += quarter_mass;
            mass[i3 as usize] += quarter_mass;
        }

        let pos_bits: Vec<u32> = positions
            .iter()
            .flat_map(|p| [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
            .collect();

        let normal_bits: Vec<u32> = init_normals
            .iter()
            .flat_map(|n| [n[0].to_bits(), n[1].to_bits(), n[2].to_bits()])
            .collect();

        let verts: Vec<VertData> = (0..nv)
            .map(|v| {
                let p = positions[v];
                let im = if is_pinned[v] || mass[v] <= 1e-12 {
                    0.0
                } else {
                    1.0 / mass[v]
                };
                VertData {
                    pos_mid_im: [p[0], p[1], p[2], im],
                    pos_prev: [p[0], p[1], p[2], 0.0],
                    vel: [0.0, 0.0, 0.0, 0.0],
                }
            })
            .collect();

        let tet_state = vec![[0.0f32; 4]; nt];

        Self {
            pos_bits,
            normal_bits,
            verts,
            tets,
            tet_state,
            surface_tris,
            canonical_tets,
        }
    }
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct PhysData {
    pub generation: u64,
    pub pos: Handle<ShaderBuffer>,
    pub normals: Handle<ShaderBuffer>,
    pub verts: Handle<ShaderBuffer>,
    pub tets: Handle<ShaderBuffer>,
    pub tet_state: Handle<ShaderBuffer>,
    pub surface_tris: Handle<ShaderBuffer>,
    pub nv: u32,
    pub nt: u32,
    pub n_surf_tris: u32,
    pub enable_floor: bool,
}

#[derive(Component)]
pub struct SoftBodyEntity;

#[derive(Component)]
pub struct CantileverAnchorEntity;

#[derive(Resource)]
pub struct PendingModelLoad {
    pub preset: Preset,
    pub pos_handle: Handle<PodArrayAsset<[f32; 3]>>,
    pub tet_handle: Handle<PodArrayAsset<[u32; 4]>>,
}

#[derive(Message, Copy, Clone, Debug)]
pub struct RequestLoadPreset(pub Preset);

#[derive(Resource, Default)]
pub struct GenerationCounter(pub u64);

pub fn trigger_preset_load_system(
    mut commands: Commands,
    mut messages: MessageReader<RequestLoadPreset>,
    asset_server: Res<AssetServer>,
    mut sim_config: ResMut<SimConfig>,
) {
    let Some(msg) = messages.read().last().copied() else {
        return;
    };
    let preset = msg.0;
    sim_config.preset = preset;
    let slug = preset.slug();

    let pos_handle = asset_server.load(format!("models/{slug}/{slug}_positions.vec3f32bin"));
    let tet_handle = asset_server.load(format!("models/{slug}/{slug}_tet_indices.vec4u32bin"));

    commands.insert_resource(PendingModelLoad {
        preset,
        pos_handle,
        tet_handle,
    });
}

pub fn finalize_pending_model_load_system(
    mut commands: Commands,
    pending: Option<Res<PendingModelLoad>>,
    pos_assets: Res<Assets<PodArrayAsset<[f32; 3]>>>,
    tet_assets: Res<Assets<PodArrayAsset<[u32; 4]>>>,
    existing_soft: Query<Entity, With<SoftBodyEntity>>,
    existing_anchor: Query<Entity, With<CantileverAnchorEntity>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut ext_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, SoftBodyMaterialExt>>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut gen_counter: ResMut<GenerationCounter>,
) {
    let Some(pending) = pending else {
        return;
    };
    let (Some(pos_asset), Some(tet_asset)) = (
        pos_assets.get(&pending.pos_handle),
        tet_assets.get(&pending.tet_handle),
    ) else {
        return;
    };

    let positions = &pos_asset.data;
    let raw_tets = &tet_asset.data;

    let min_x = positions
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);
    let is_pinned: Vec<bool> = positions
        .iter()
        .map(|p| pending.preset == Preset::Beam && p[0] <= min_x + 1e-3)
        .collect();

    let cpu = CpuPhysBuffers::build(positions, raw_tets, &is_pinned, 1000.0);

    for e in existing_soft.iter() {
        commands.entity(e).despawn();
    }
    for e in existing_anchor.iter() {
        commands.entity(e).despawn();
    }

    gen_counter.0 += 1;
    let generation = gen_counter.0;

    let pos_buf = buffers.add(ShaderBuffer::from(cpu.pos_bits));
    let normals_buf = buffers.add(ShaderBuffer::from(cpu.normal_bits));
    let verts_buf = buffers.add(ShaderBuffer::from(cpu.verts));
    let tets_buf = buffers.add(ShaderBuffer::from(cpu.tets));
    let tet_state_buf = buffers.add(ShaderBuffer::from(cpu.tet_state));
    let surf_tris_buf = buffers.add(ShaderBuffer::from(cpu.surface_tris.clone()));

    let n_surf_tris = (cpu.surface_tris.len() / 3) as u32;
    let surface_mesh = build_bevy_surface_mesh(positions, &cpu.surface_tris);

    let material = ext_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: pending.preset.base_color(),
            perceptual_roughness: 0.42,
            metallic: 0.04,
            reflectance: 0.45,
            ..default()
        },
        extension: SoftBodyMaterialExt {
            vertex_pos: pos_buf.clone(),
            vertex_normals: normals_buf.clone(),
        },
    });

    commands.spawn((
        SoftBodyEntity,
        Mesh3d(meshes.add(surface_mesh)),
        MeshMaterial3d(material),
        Transform::IDENTITY,
        NoFrustumCulling,
        Aabb::from_min_max(Vec3::new(-10.0, -1.0, -10.0), Vec3::new(10.0, 10.0, 10.0)),
    ));

    if pending.preset == Preset::Beam {
        commands.spawn((
            CantileverAnchorEntity,
            Mesh3d(meshes.add(Cuboid::new(0.18, 2.1, 0.85))),
            MeshMaterial3d(std_materials.add(StandardMaterial {
                base_color: Color::srgb(0.24, 0.26, 0.30),
                perceptual_roughness: 0.75,
                metallic: 0.2,
                ..default()
            })),
            Transform::from_xyz(min_x - 0.09, 1.05, 0.0),
        ));
    }

    commands.insert_resource(PhysData {
        generation,
        pos: pos_buf,
        normals: normals_buf,
        verts: verts_buf,
        tets: tets_buf,
        tet_state: tet_state_buf,
        surface_tris: surf_tris_buf,
        nv: positions.len() as u32,
        nt: cpu.canonical_tets.len() as u32,
        n_surf_tris,
        enable_floor: pending.preset.enable_floor(),
    });

    commands.remove_resource::<PendingModelLoad>();
}
