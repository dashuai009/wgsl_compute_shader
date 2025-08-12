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

use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::{prelude::*, scene::SceneInstanceReady};
use bytemuck::cast_slice;
use slint::wgpu_24::wgpu;
use slint::{Model, SharedString};
use std::borrow::Cow;

/// This example uses a shader source file from the assets subdirectory
const VERTEX_SHADER_ASSET_PATH: &str = "shaders/sim_pipeline.wgsl";
const ARAP_SHADER_ASSET_PATH: &str = "shaders/arap.wgsl";

const DISPLAY_FACTOR: u32 = 4;

const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

#[derive(Resource, Clone, ExtractResource)]
pub struct PhysData {
    pub pos: Handle<ShaderStorageBuffer>,
    pub tet_indices: Handle<ShaderStorageBuffer>,
    pub points_size: u32,
    pub tet_size: u32,
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct PositionData {
    // We need to ensure that the bindings of the base material and the extension do not conflict,
    // so we start from binding slot 100, leaving slots 0-99 for the base material.
    #[storage(100)]
    pos: Handle<ShaderStorageBuffer>, // [vec3<f32>; points_size]
}

impl MaterialExtension for PositionData {
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
    scene_roots: Query<&SceneRoot>,
    mesh_handles: Query<&Mesh3d>,
    meshes: Res<Assets<Mesh>>,
    mesh_materials: Query<&MeshMaterial3d<StandardMaterial>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    mut asset_materials: ResMut<Assets<StandardMaterial>>,
    mut ext_materials: ResMut<Assets<ExtendedMaterial<StandardMaterial, PositionData>>>,
) {
    let Ok(_scene_root) = scene_roots.get(trigger.target()) else {
        return;
    };

    for e in children.iter_descendants(trigger.target()) {
        if let Ok(mesh3d) = mesh_handles.get(e) {
            if let Some(mesh) = meshes.get(mesh3d.id()) {
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

                // Get the material of the descendant
                if let Some(material) = mesh_materials
                    .get(e)
                    .ok()
                    .and_then(|id| asset_materials.get_mut(id.id()))
                {
                    let points_size = positions.len() as u32;
                    let tet_size = indices_u32.len() as u32;
                    let pos_buffer = buffers.add(ShaderStorageBuffer::from(
                        positions
                            .into_iter()
                            .flatten() // 自动展平 [f32; 3]
                            .collect::<Vec<_>>(),
                    ));

                    let tet_indices_buffer = buffers.add(ShaderStorageBuffer::from(indices_u32));

                    let phys_data = PhysData {
                        pos: pos_buffer.clone(),
                        tet_indices: tet_indices_buffer,
                        points_size,
                        tet_size,
                    };
                    // Create a copy of the material and override base color
                    // If you intend on creating multiple models with the same tint, it
                    // is best to cache the handle somewhere, as having multiple materials
                    // that are identical is expensive
                    let new_material = material.clone();
                    // new_material.base_color = color_override.0;
                    let extended_material = ExtendedMaterial {
                        base: new_material,
                        extension: PositionData { pos: pos_buffer },
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
                    binding_types::storage_buffer_read_only::<u32>(false), // tet_indices_buffer
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
        render_app.add_systems(
            Render,
            prepare_sim_data_bind_groups
                .in_set(RenderSet::PrepareBindGroups)
                .run_if(resource_added::<PhysData>), // 只在第一次出现时执行,
        );
        // render_app.add_systems(Render, readback_and_log_system.in_set(RenderSet::Cleanup)); // for test

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(SimLabel, SimNode::default());
        render_graph.add_node_edge(SimLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<SimPipelineData>();
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
) {
    info!("prepare phys data bind groups");
    // 句柄 -> GPU 侧资源
    let Some(pos_gpu) = gpu_ssbos.get(&phys_data.pos) else {
        return;
    };
    let Some(tet_gpu) = gpu_ssbos.get(&phys_data.tet_indices) else {
        return;
    };
    info!("prepare phys data bind groups end");

    let pos_buffer_id = pos_gpu.buffer.as_entire_binding();
    let tet_indices_buffer_id = tet_gpu.buffer.as_entire_binding();

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((pos_buffer_id, tet_indices_buffer_id)),
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
                        self.sim_state = SimState::UpdatePos;
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
                // self.sim_state = SimState::LoadingMesh;
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
        // let phys_data = world.resource::<PhysData>();
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
                // pass.dispatch_workgroups(phys_data.points_size as u32 / WORKGROUP_SIZE, phys_data.points_size as u32 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}
