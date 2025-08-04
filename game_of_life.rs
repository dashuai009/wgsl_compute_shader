//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    color::palettes::basic::PURPLE,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{binding_types::texture_storage_2d, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSet,
    }, tasks::futures_lite,
};
use slint::wgpu_24::wgpu;
use std::borrow::Cow;

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/game_of_life.wgsl";

const DISPLAY_FACTOR: u32 = 4;
const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
    println!("SIZE: {:?}", SIZE);
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (
                            (SIZE.0 * DISPLAY_FACTOR) as f32,
                            (SIZE.1 * DISPLAY_FACTOR) as f32,
                        )
                            .into(),
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            GameOfLifeComputePlugin,
            SimPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, switch_textures)
        .run();
}
#[derive(Resource)]
struct ComputeData {
    input: Buffer,
    output: Buffer,
    size: usize,
}

// prepare data
fn setup_buffers(mut commands: Commands, render_device: Res<RenderDevice>) {
    let input_data = vec![1u32, 2, 3, 4, 42];
    let size_bytes = (input_data.len() * std::mem::size_of::<u32>()) as u64;

    let input_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("InputBuffer"),
        contents: bytemuck::cast_slice(&input_data),
        usage: BufferUsages::STORAGE,
    });

    let output_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("OutputBuffer"),
        size: size_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    commands.insert_resource(ComputeData {
        input: input_buffer,
        output: output_buffer,
        size: input_data.len(),
    });
}

#[derive(Resource)]
struct ComputePipelineData {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl ComputePipelineData {
    // create pipeline
    // fn setup_pipeline(
    //     render_device: Res<RenderDevice>,
    //     pipeline_cache: Res<PipelineCache>,
    //     asset_server: Res<AssetServer>,
    //     buffers: Res<ComputeBuffers>,
    // ) -> ComputePipelineData{
    fn setup_pipeline(
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        asset_server: &AssetServer,
    ) -> ComputePipelineData {
        let layout = render_device.create_bind_group_layout(
            "buffer_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    binding_types::storage_buffer_read_only::<u32>(false),
                    binding_types::storage_buffer::<u32>(false),
                ),
            ),
        );

        // let bind_group = render_device.create_bind_group(
        //     Some("buffer_bind_group"),
        //     &layout,
        //     &[
        //         BindGroupEntry {
        //             binding: 0,
        //             resource: buffers.input.as_entire_binding(),
        //         },
        //         BindGroupEntry {
        //             binding: 1,
        //             resource: buffers.output.as_entire_binding(),
        //         },
        //     ],
        // );

        let shader = asset_server.load("shaders/buffer_add.wgsl");
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            layout: vec![layout.clone()],
            shader,
            entry_point: Cow::from("add_one"),
            label: Some(Cow::from("compute_pipeline")),
            push_constant_ranges: vec![],
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        });

        return ComputePipelineData { layout, pipeline };
    }
}

impl FromWorld for ComputePipelineData {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

        return ComputePipelineData::setup_pipeline(
            device,
            pipeline_cache,
            asset_server
        );
    }
}

// // commit command to compute
// fn run_compute_shader(
//     pipeline_cache: Res<PipelineCache>,
//     pipeline_data: Res<ComputePipelineData>,
//     buffers: Res<ComputeData>,
//     render_context: &mut RenderContext,
// ) {
//     if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_data.pipeline) {
//         let mut pass = render_context
//             .command_encoder()
//             .begin_compute_pass(&ComputePassDescriptor::default());
//         pass.set_pipeline(pipeline);
//         pass.set_bind_group(0, &pipeline_data.bind_group, &[]);
//         pass.dispatch_workgroups(buffers.size as u32, 1, 1);
//     }
// }

struct SimPlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SimLabel;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.

        // app.add_plugins(ExtractResourcePlugin::<ComputeBuffers>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems( ExtractSchedule, setup_buffers);
        render_app.add_systems(Render, prepare_sim_data_bind_groups.in_set(RenderSet::PrepareBindGroups));
        render_app.add_systems(Render, readback_and_log_system.in_set(RenderSet::Cleanup)); // for test

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(SimLabel, SimNode::default());
        render_graph.add_node_edge(SimLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<ComputePipelineData>();
    }
}

#[derive(Resource)]
struct SimDataBindGroups {
    sim_data: BindGroup,
}

fn prepare_sim_data_bind_groups(
    mut commands: Commands,
    pipeline: Res<ComputePipelineData>,
    sim_buffers: Res<ComputeData>,
    render_device: Res<RenderDevice>,
) {
    let input_id = sim_buffers.input.as_entire_binding();
    let output_id = sim_buffers.output.as_entire_binding();

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::sequential((input_id, output_id)),
    );
    commands.insert_resource(SimDataBindGroups {
        sim_data: bind_group_0,
    });
}

enum SimState {
    Loading,
    Init,
    Update,
}

struct SimNode {
    state: SimState,
}

impl Default for SimNode {
    fn default() -> Self {
        SimNode {
            state: SimState::Loading,
        }
    }
}

impl render_graph::Node for SimNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<ComputePipelineData>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
             SimState::Loading  => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = SimState::Update;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            SimState::Init => {
            }
            SimState::Update => {}
        }
    }

    fn run(
        &self,
        graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> std::result::Result<(), render_graph::NodeRunError> {
        let bind_groups = world.resource::<SimDataBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<ComputePipelineData>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            SimState::Loading => {}
            SimState::Init => {
                // let init_pipeline = pipeline_cache
                //     .get_compute_pipeline(pipeline.init_pipeline)
                //     .unwrap();
                // pass.set_bind_group(0, &bind_groups[0], &[]);
                // pass.set_pipeline(init_pipeline);
                // pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            SimState::Update => {
                println!("update");
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups.sim_data, &[]);
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}



/// 每帧把 GPU Buffer 拷回 CPU 并打印
fn readback_and_log_system(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu: Res<ComputeData>,
) {
    // 1. 创建 staging buffer
    let staging = render_device.create_buffer(&BufferDescriptor {
        label: Some("staging_readback"),
        size: (gpu.size * std::mem::size_of::<u32>()) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 2. 拷贝 GPU→CPU 缓冲
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(&gpu.output, 0, &staging, 0, (gpu.size * std::mem::size_of::<u32>()) as u64);
    render_queue.submit(Some(encoder.finish()));

    // 3. 异步映射 & 轮询完成
    //    这里用 block_on + poll(WAIT) 简化演示；正式项目可用 bevy_task::IoTaskPool
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    staging
        .slice(..)
        .map_async(MapMode::Read, move |v| sender.send(v).unwrap());

    // 阻塞等完成（非 WASM 环境可用 WAIT；WASM 需放到回调里）
    render_device.wgpu_device().poll(wgpu::Maintain::Wait);
    futures_lite::future::block_on(async {
        receiver.receive().await;
    });

    // 4. 读取数据
    let bytes = staging.slice(..).get_mapped_range();
    let floats: &[u32] = bytemuck::cast_slice(&bytes);
    println!("GPU buffer contents: {:?}", floats);

    // 5. 解除映射，方便下次使用
    drop(bytes);
    staging.unmap();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::R32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image0 = images.add(image.clone());
    let image1 = images.add(image);

    commands.spawn((
        Sprite {
            image: image0.clone(),
            custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
            ..default()
        },
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    commands.insert_resource(GameOfLifeImages {
        texture_a: image0,
        texture_b: image1,
    });
}

// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<GameOfLifeImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone_weak();
    } else {
        sprite.image = images.texture_a.clone_weak();
    }
}

struct GameOfLifeComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct GameOfLifeLabel;

impl Plugin for GameOfLifeComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(ExtractResourcePlugin::<GameOfLifeImages>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GameOfLifeLabel, GameOfLifeNode::default());
        render_graph.add_node_edge(GameOfLifeLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<GameOfLifePipeline>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct GameOfLifeImages {
    texture_a: Handle<Image>,
    texture_b: Handle<Image>,
}

#[derive(Resource)]
struct GameOfLifeImageBindGroups([BindGroup; 2]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<GameOfLifePipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    game_of_life_images: Res<GameOfLifeImages>,
    render_device: Res<RenderDevice>,
) {
    let view_a = gpu_images.get(&game_of_life_images.texture_a).unwrap();
    let view_b = gpu_images.get(&game_of_life_images.texture_b).unwrap();
    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((&view_a.texture_view, &view_b.texture_view)),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((&view_b.texture_view, &view_a.texture_view)),
    );
    commands.insert_resource(GameOfLifeImageBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct GameOfLifePipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for GameOfLifePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = render_device.create_bind_group_layout(
            "GameOfLifeImages",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let shader = world.load_asset(SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("init"),
            zero_initialize_workgroup_memory: false,
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
            zero_initialize_workgroup_memory: false,
        });

        GameOfLifePipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

enum GameOfLifeState {
    Loading,
    Init,
    Update(usize),
}

struct GameOfLifeNode {
    state: GameOfLifeState,
}

impl Default for GameOfLifeNode {
    fn default() -> Self {
        Self {
            state: GameOfLifeState::Loading,
        }
    }
}

impl render_graph::Node for GameOfLifeNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<GameOfLifePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            GameOfLifeState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = GameOfLifeState::Init;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            GameOfLifeState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = GameOfLifeState::Update(1);
                }
            }
            GameOfLifeState::Update(0) => {
                self.state = GameOfLifeState::Update(1);
            }
            GameOfLifeState::Update(1) => {
                self.state = GameOfLifeState::Update(0);
            }
            GameOfLifeState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<GameOfLifeImageBindGroups>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<GameOfLifePipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            GameOfLifeState::Loading => {}
            GameOfLifeState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
            GameOfLifeState::Update(index) => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.0 / WORKGROUP_SIZE, SIZE.1 / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}
