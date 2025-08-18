use bevy::render::mesh::{Indices, VertexAttributeValues};
use bevy::render::render_graph::{RenderGraph, RenderLabel};
use bevy::render::render_resource::{
    binding_types, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
    BufferDescriptor, BufferInitDescriptor, BufferUsages, CachedComputePipelineId,
    CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
    ShaderStages,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::{render_graph, Render, RenderApp, RenderSet};
use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::{MeshVertexAttribute, MeshVertexBufferLayoutRef},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_resource::*,
        render_resource::{binding_types::texture_storage_2d, *},
        renderer::RenderQueue,
        texture::GpuImage,
    },
    sprite::{Material2d, Material2dKey},
    tasks::futures_lite,
};

use crate::bin_asset_loader::PodArrayAsset;
use bevy::diagnostic::FrameCount;
use bevy::ecs::error::panic;
use bevy::render::globals::{GlobalsBuffer, GlobalsUniform};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::{prelude::*, scene::SceneInstanceReady};
use bytemuck::cast_slice;
use nalgebra;
use slint::wgpu_24::wgpu;
use slint::{Model, SharedString};
use std::borrow::Cow;
// N_tri, N_pos, N_tet
// indices_u32: [[u32; 3]; N_tri]
// positions: [[f32; 3]; N_tri] in glb file, bageyalu -> 排序  --|
// tet_pos/pos: [[f32; 3]: N_tet]    -> 排序  ----->

/// This example uses a shader source file from the assets subdirectory
const VERTEX_SHADER_ASSET_PATH: &str = "shaders/sim_pipeline.wgsl";
const ARAP_SHADER_ASSET_PATH: &str = "shaders/arap.wgsl";

#[derive(Component)]
pub struct TetData {
    pub tet_indices: Handle<PodArrayAsset<[u32; 4]>>,
    pub tet_pos: Handle<PodArrayAsset<[f32; 3]>>,
}

#[derive(ShaderType, Default)]
pub struct SimParams {
    dt: f32,
    omega: f32,
    NT: u32,
    NV: u32,
}

/// The buffer containing the [`SimParams`]
#[derive(Resource, Default)]
pub struct SimParamsBuffer {
    pub buffer: UniformBuffer<SimParams>,
}

pub fn prepare_sim_params_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut sim_params_buffer: ResMut<SimParamsBuffer>,
    time: Res<Time>,
    frame_count: Res<FrameCount>,
) {
    let buffer = sim_params_buffer.buffer.get_mut();
    buffer.dt = time.elapsed_secs_wrapped();
    // buffer.delta_time = time.delta_secs();
    // buffer.frame_count = frame_count.0;

    sim_params_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);
}

#[derive(Resource, Clone, ExtractResource)]
pub struct PhysData {
    pub pos: Handle<ShaderStorageBuffer>,         // [f32; N_pos * 3]
    pub tet_indices: Handle<ShaderStorageBuffer>, // [[u32; 4]; N_tet]
    pub inv_mass: Handle<ShaderStorageBuffer>,    // [f32; N_pos]
    pub B: Handle<ShaderStorageBuffer>,           // [f32; N_tet * 9]
    pub pos_mid: Handle<ShaderStorageBuffer>,     // [f32; N_pos * 3]
    pub lagrange: Handle<ShaderStorageBuffer>,    // [f32; N_tet]
    pub residuals: Handle<ShaderStorageBuffer>,   // [f32; N_tet]
    pub is_dirven: Handle<ShaderStorageBuffer>,   // [f32; N_pos]
    pub points_size: u32,
    pub tet_size: u32,
}

impl PhysData {
    pub fn new(
        positions: &[[f32; 3]],
        tet_indices: &[[u32; 4]],
        mut buffers: &mut ResMut<Assets<ShaderStorageBuffer>>,
    ) -> Self {
        let points_size = positions.len() as u32;
        let tet_size = tet_indices.len() as u32;

        let pos_buffer_data = positions
            .into_iter()
            .map(|x| [x[0], 0.0,  x[2]])
            .flatten() // 自动展平 [f32; 3]
            .collect::<Vec<_>>();
        
        let pos_buffer = buffers.add(ShaderStorageBuffer::from(pos_buffer_data.clone()));

        let tet_indices_buffer_data = tet_indices
            .into_iter()
            .flatten() // 自动展平 [f32; 3]
            .collect::<Vec<_>>();
        let tet_indices_buffer = buffers.add(ShaderStorageBuffer::from(tet_indices_buffer_data));

        let inv_mass_buffer_data = vec![1.0f32; positions.len()];
        let inv_mass_buffer = buffers.add(ShaderStorageBuffer::from(inv_mass_buffer_data));

        let mut B_buffer_data = vec![];

        for (i, [ia, ib, ic, id]) in tet_indices.iter().enumerate() {
            let p0 = nalgebra::Vector3::from(positions[ia.clone() as usize]);
            let p1 = nalgebra::Vector3::from(positions[ib.clone() as usize]);
            let p2 = nalgebra::Vector3::from(positions[ic.clone() as usize]);
            let p3 = nalgebra::Vector3::from(positions[id.clone() as usize]);

            // 构造 D_m: 每一列是 pi - p0
            let d_m = nalgebra::Matrix3::from_columns(&[p1 - p0, p2 - p0, p3 - p0]);

            B_buffer_data.extend_from_slice(d_m.try_inverse().unwrap().as_slice());
        }
        let b_buffer_data = buffers.add(ShaderStorageBuffer::from(B_buffer_data));

        // let p = buffers.add(ShaderStorageBuffer::from(B_buffer_data));
        let pos_mid_buffer_data = pos_buffer_data;
        let pos_mid_buffer = buffers.add(ShaderStorageBuffer::from(pos_mid_buffer_data));

        let lagrange = vec![0.0f32; tet_indices.len()];
        let lagrange_buffer = buffers.add(ShaderStorageBuffer::from(lagrange));

        let residuals = vec![0.0f32; tet_indices.len()];
        let residuals_buffer = buffers.add(ShaderStorageBuffer::from(residuals));

        let is_driven = vec![0u32; positions.len()];
        let is_driven_buffer = buffers.add(ShaderStorageBuffer::from(is_driven));

        PhysData {
            pos: pos_buffer.clone(),
            tet_indices: tet_indices_buffer,
            inv_mass: inv_mass_buffer,
            B: b_buffer_data,
            pos_mid: pos_mid_buffer,
            lagrange: lagrange_buffer,
            residuals: residuals_buffer,
            is_dirven: is_driven_buffer,
            points_size,
            tet_size,
        }
    }
}
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct TriData {
    // We need to ensure that the bindings of the base material and the extension do not conflict,
    // so we start from binding slot 100, leaving slots 0-99 for the base material.
    #[storage(100)]
    tet_pos: Handle<ShaderStorageBuffer>, // [f32; points_size * 3],
    #[storage(101)]
    to_tet_indices: Handle<ShaderStorageBuffer>, // [f32; points_size * 3],
}

impl MaterialExtension for TriData {
    fn vertex_shader() -> ShaderRef {
        VERTEX_SHADER_ASSET_PATH.into()
    }
}

/// 在 SceneInstanceReady 触发时，遍历该 SceneRoot 的所有后代，
/// 对每个带 Mesh3d 的实体，从 `Assets<Mesh>` 读取 position 和 indices。
pub fn dump_mesh_cpu_data_system(
    trigger: Trigger<SceneInstanceReady>,
    mut commands: Commands,
    children: Query<&Children>,
    scene_roots: Query<(&SceneRoot, &TetData)>,
    tet_indices_assets: Res<Assets<PodArrayAsset<[u32; 4]>>>,
    positions_assets: Res<Assets<PodArrayAsset<[f32; 3]>>>,
    mut mesh_handles: Query<&mut Mesh3d>,
    mut meshes: ResMut<Assets<Mesh>>,
    mesh_materials: Query<&MeshMaterial3d<StandardMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut asset_materials: ResMut<Assets<StandardMaterial>>,
    mut ext_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, TriData>>>,
) {
    let Ok((_scene_root, tet)) = scene_roots.get(trigger.target()) else {
        return;
    };

    let tet_indices = match tet_indices_assets.get(&tet.tet_indices) {
        Some(a) => &a.data, // 假设你的 PodArrayAsset<T>(pub Vec<T>)
        None => {
            // 资产未就绪，先不做；也可以发个事件等资产加载完再处理
            return;
        }
    };
    let tet_pos = match positions_assets.get(&tet.tet_pos) {
        Some(a) => &a.data,
        None => {
            return;
        }
    };

    for e in children.iter_descendants(trigger.target()) {
        if let Ok(mesh3d) = &mut mesh_handles.get_mut(e) {
            if let Some(mesh) = meshes.get_mut(mesh3d.id()) {
                // ---------- 顶点坐标 ----------
                // glTF 常见的是 Float32x3；有些管线可能是 Float32x4（带 w）
                let positions: Vec<[f32; 3]> = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
                    Some(VertexAttributeValues::Float32x3(v)) => v.clone(),
                    Some(VertexAttributeValues::Float32x4(v)) => {
                        println!("errrrrrrr");
                        v.iter().map(|p| [p[0], p[1], p[2]]).collect()
                    }
                    other => {
                        warn!("Entity {:?}: unexpected POSITION format: {:?}", e, other);
                        Vec::new()
                    }
                };

                // ---------- 索引 ----------
                // 可能是 None（非索引绘制）、U16 或 U32
                let indices_u32: Vec<u32> = match mesh.indices() {
                    Some(Indices::U32(v)) => v.clone(),
                    Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
                    None => (0..positions.len() as u32).collect(), // 无索引时按顺序展开
                };

                info!(
                    "Entity {:?}:pos_size: {}, positions = {:?},indices_size: {}, indices = {:?}, topology = {:?}",
                    e,
                    positions.len(),
                    positions,
                    indices_u32.len(),
                    indices_u32,
                    mesh.primitive_topology()
                );

                // println!("positions: {:?}, tet_pos : {:?}", positions, tet_pos);
                let mut new_idx = vec![]; //
                for i in &indices_u32 {
                    for k in 0..tet_pos.len() {
                        if tet_pos[k] == positions[*i as usize] {
                            new_idx.push(k as u32);
                            break;
                        }
                    }
                }
                println!("new_idx: {:?}", new_idx);

                if let Some(mut iii) = mesh.indices_mut() {
                    *iii = Indices::U32(new_idx.clone());
                };

                // Get the material of the descendant or use the default one
                let material: StandardMaterial = mesh_materials
                    .get(e)
                    .ok()
                    .and_then(|id| asset_materials.get(id.id()))
                    .cloned()
                    .unwrap_or_else(|| {
                        // If the material is not found, create a new one with the default color
                        let new_material = StandardMaterial::from(Color::WHITE);
                        // asset_materials.add(new_material)
                        new_material
                    });

                let phys_data = PhysData::new(&tet_pos, &tet_indices, &mut buffers);

                let to_tet_indices = buffers.add(ShaderStorageBuffer::from(new_idx));
                // Create a copy of the material and override base color
                // If you intend on creating multiple models with the same tint, it
                // is best to cache the handle somewhere, as having multiple materials
                // that are identical is expensive
                // new_material.base_color = color_override.0;
                let extended_material = ExtendedMaterial {
                    base: material,
                    extension: TriData {
                        tet_pos: phys_data.pos.clone(),
                        to_tet_indices,
                    },
                };
                // Override `MeshMaterial3d` with new material
                commands
                    .entity(e)
                    .remove::<MeshMaterial3d<StandardMaterial>>()
                    .insert(MeshMaterial3d(ext_materials.add(extended_material)))
                    .log_components();

                commands.insert_resource(phys_data);
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum SimState {
    LoadingMesh,
    InitPhysData,
    UpdatePos,
}

#[derive(Resource)]
pub struct SimPipelineData {
    layout: BindGroupLayout,
    sim_pipeline: CachedComputePipelineId,
}

impl SimPipelineData {
    fn setup_pipeline(
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        asset_server: &AssetServer,
    ) -> SimPipelineData {
        let layout = render_device.create_bind_group_layout(
            "phys_data_buffer_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    binding_types::storage_buffer::<f32>(false), // pos_buffer
                    binding_types::storage_buffer_read_only::<[u32; 4]>(false), // tet_indices_buffer
                    binding_types::storage_buffer_read_only::<f32>(false),      // inv_mass_buffer
                    binding_types::storage_buffer_read_only::<f32>(false),      // B_flat_buffer
                    binding_types::storage_buffer_read_only::<f32>(false),      // pos_mid_buffer
                    binding_types::storage_buffer::<f32>(false),                // lagrange_buffer
                    binding_types::storage_buffer::<f32>(false),                // residuals_buffer
                    binding_types::storage_buffer_read_only::<u32>(false),      // is_driven_buffer
                    binding_types::uniform_buffer::<SimParams>(false),          // params
                ),
            ),
        );
        let shader = asset_server.load(ARAP_SHADER_ASSET_PATH);
        let sim_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![layout.clone()],
            shader,
            entry_point: Cow::from("cs_arap"),
            label: Some(Cow::from("sim_compute_pipeline")),
            push_constant_ranges: vec![],
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        });

        return SimPipelineData {
            layout,
            sim_pipeline,
        };
    }
}

impl FromWorld for SimPipelineData {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

        return SimPipelineData::setup_pipeline(device, pipeline_cache, asset_server);
    }
}

pub struct SimPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SimLabel;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        // Extract the MeshInputData resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        // 让 MeshInputData 从 main world 自动同步到 render world
        app.add_plugins(ExtractResourcePlugin::<PhysData>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(
                Render,
                prepare_sim_data_bind_groups
                    .in_set(RenderSet::PrepareBindGroups)
                    .run_if(resource_added::<PhysData>), // 只在第一次出现时执行,
            )
            .add_systems(
                Render,
                prepare_sim_params_buffer.in_set(RenderSet::PrepareResources),
            );
        // render_app.add_systems(Render, readback_and_log_system.in_set(RenderSet::Cleanup)); // for test

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(SimLabel, SimNode::default());
        render_graph.add_node_edge(SimLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<SimPipelineData>();
        render_app.init_resource::<SimParamsBuffer>();
    }
}

#[derive(Resource)]
struct PhysDataBindGroups {
    phys_data_bind_group: BindGroup,
}

fn prepare_sim_data_bind_groups(
    mut commands: Commands,
    pipeline: Res<SimPipelineData>,
    phys_data: Res<PhysData>,
    gpu_ssbos: Res<RenderAssets<GpuShaderStorageBuffer>>,
    render_device: Res<RenderDevice>,
    mut sim_params_buffer: ResMut<SimParamsBuffer>,
    render_queue: Res<RenderQueue>,
) {
    let buffer = sim_params_buffer.buffer.get_mut();
    buffer.omega = 1.0f32;
    buffer.NT = phys_data.tet_size;
    buffer.NV = phys_data.points_size;

    sim_params_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);
    info!("prepare phys data bind groups");
    // 句柄 -> GPU 侧资源
    let Some(pos_gpu) = gpu_ssbos.get(&phys_data.pos) else {
        warn!("phys data pos bind group does not exist");
        return;
    };
    let Some(tet_gpu) = gpu_ssbos.get(&phys_data.tet_indices) else {
        warn!("phys data tet index bind group does not exist");
        return;
    };
    let Some(inv_mass_gpu) = gpu_ssbos.get(&phys_data.inv_mass) else {
        warn!("phys data inv_mass bind group does not exist");
        return;
    };
    let Some(B_gpu) = gpu_ssbos.get(&phys_data.B) else {
        warn!("phys data B bind group does not exist");
        return;
    };
    let Some(pos_mid_gpu) = gpu_ssbos.get(&phys_data.pos_mid) else {
        warn!("phys data pos_mid_bind group does not exist");
        return;
    };
    let Some(lagrange_gpu) = gpu_ssbos.get(&phys_data.lagrange) else {
        warn!("phys data lagrange_bind group does not exist");
        return;
    };
    let Some(is_driven_gpu) = gpu_ssbos.get(&phys_data.is_dirven) else {
        warn!("phys data is_dirven_bind group does not exist");
        return;
    };
    let Some(residuals_gpu) = gpu_ssbos.get(&phys_data.residuals) else {
        warn!("phys data residuals_gpu does not exist");
        return;
    };
    // let Some(sim_params_gpu) = gpu_ssbos.get(&phys_data.is_dirven) else {
    //     return;
    // };
    info!("prepare phys data bind groups end");

    let pos_buffer_id = pos_gpu.buffer.as_entire_binding();
    let tet_indices_buffer_id = tet_gpu.buffer.as_entire_binding();
    let inv_mass_buffer_id = inv_mass_gpu.buffer.as_entire_binding();
    let B_buffer_id = B_gpu.buffer.as_entire_binding();
    let pos_mid_buffer_id = pos_mid_gpu.buffer.as_entire_binding();
    let lagrange_buffer_id = lagrange_gpu.buffer.as_entire_binding();
    let is_driven_buffer_id = is_driven_gpu.buffer.as_entire_binding();
    let residuals_buffer_id = residuals_gpu.buffer.as_entire_binding();
    let sim_params_buffer_id = sim_params_buffer
        .buffer
        .buffer()
        .unwrap()
        .as_entire_binding();

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((
            pos_buffer_id,
            tet_indices_buffer_id,
            inv_mass_buffer_id,
            B_buffer_id,
            pos_mid_buffer_id,
            lagrange_buffer_id,
            residuals_buffer_id,
            is_driven_buffer_id,
            sim_params_buffer_id,
        )),
    );
    commands.insert_resource(PhysDataBindGroups {
        phys_data_bind_group: bind_group_0,
    });
}

struct SimNode {
    sim_state: SimState,
}

impl Default for SimNode {
    fn default() -> Self {
        SimNode {
            sim_state: SimState::LoadingMesh,
        }
    }
}

impl render_graph::Node for SimNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<SimPipelineData>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.sim_state {
            SimState::LoadingMesh => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.sim_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        println!("okkk");
                        self.sim_state = SimState::InitPhysData;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{ARAP_SHADER_ASSET_PATH:?}:\n{err}")
                    }
                    _ => {}
                }
            }
            SimState::InitPhysData => {
                self.sim_state = SimState::UpdatePos;
            }
            SimState::UpdatePos => {}
        }
    }

    fn run(
        &self,
        graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> std::result::Result<(), render_graph::NodeRunError> {
        if !world.contains_resource::<PhysDataBindGroups>() {
            return Ok(());
        }
        let phys_data = world.resource::<PhysData>();
        let bind_groups = world.resource::<PhysDataBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SimPipelineData>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.sim_state {
            SimState::LoadingMesh => {}
            SimState::InitPhysData => {
                // let init_pipeline = pipeline_cache
                //     .get_compute_pipeline(pipeline.init_pipeline)
                //     .unwrap();
                // pass.set_bind_group(0, &bind_groups[0], &[]);
                // pass.set_pipeline(init_pipeline);
                // pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            SimState::UpdatePos => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.sim_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups.phys_data_bind_group, &[]);
                pass.set_pipeline(update_pipeline);
                let wg = 64u32;
                pass.dispatch_workgroups((phys_data.tet_size + wg - 1) / wg, 1, 1);
            }
        }

        Ok(())
    }
}


#[test]
mod tests {
    use super::*;
    use bevy::prelude::*;
    
    fn test_arap() {

        let (device, queue) = utils::get_wgpu_device_and_queue().await;
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.run();
        let buffers = app.world_mut().resource_mut::<Assets<ShaderStorageBuffer>>();
    }
    
}