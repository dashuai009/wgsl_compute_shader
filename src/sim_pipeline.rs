use std::borrow::Cow;

use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor,
    PipelineCache, ShaderStages, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::GpuShaderBuffer;
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::ShaderCacheError;

use crate::phys_data::{
    FIXED_FRAME_DT, JACOBI_ITERS_PER_SUBSTEP, PhysData, SUBSTEPS_PER_FRAME, SimConfig, SimParams,
};

pub const ARAP_SHADER_ASSET_PATH: &str = "shaders/arap.wgsl";
const WORKGROUP_SIZE: u32 = 64;

#[derive(Resource, Default)]
pub struct SimParamsBuffer {
    pub buffer: UniformBuffer<SimParams>,
}

#[derive(Resource, Default)]
pub struct SimPipelineReady(pub bool);

#[derive(Resource)]
pub struct SimPipelineData {
    pub layout: BindGroupLayoutDescriptor,
    pub predict_pipeline: CachedComputePipelineId,
    pub reset_lagrange_pipeline: CachedComputePipelineId,
    pub copy_mid_pos_pipeline: CachedComputePipelineId,
    pub arap_pipeline: CachedComputePipelineId,
    pub update_vel_pipeline: CachedComputePipelineId,
    pub clear_normals_pipeline: CachedComputePipelineId,
    pub accum_normals_pipeline: CachedComputePipelineId,
    pub norm_normals_pipeline: CachedComputePipelineId,
}

impl SimPipelineData {
    pub fn all_ids(&self) -> [CachedComputePipelineId; 8] {
        [
            self.predict_pipeline,
            self.reset_lagrange_pipeline,
            self.copy_mid_pos_pipeline,
            self.arap_pipeline,
            self.update_vel_pipeline,
            self.clear_normals_pipeline,
            self.accum_normals_pipeline,
            self.norm_normals_pipeline,
        ]
    }
}

fn init_sim_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "xpbd_phys_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_sized(false, None),           // 0: pos
                storage_buffer_sized(false, None),           // 1: normals
                storage_buffer_sized(false, None),           // 2: verts
                storage_buffer_read_only_sized(false, None), // 3: tets
                storage_buffer_sized(false, None),           // 4: tet_state
                storage_buffer_read_only_sized(false, None), // 5: surface_tris
                uniform_buffer::<SimParams>(false),          // 6: params
            ),
        ),
    );

    let shader = asset_server.load(ARAP_SHADER_ASSET_PATH);
    let queue_kernel = |entry: &'static str| -> CachedComputePipelineId {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(Cow::Borrowed(entry)),
            layout: vec![layout.clone()],
            shader: shader.clone(),
            entry_point: Some(Cow::Borrowed(entry)),
            ..default()
        })
    };

    let predict_pipeline = queue_kernel("predict_positions");
    let reset_lagrange_pipeline = queue_kernel("reset_lagrange");
    let copy_mid_pos_pipeline = queue_kernel("copy_mid_pos");
    let arap_pipeline = queue_kernel("cs_arap");
    let update_vel_pipeline = queue_kernel("update_velocities");
    let clear_normals_pipeline = queue_kernel("clear_normals");
    let accum_normals_pipeline = queue_kernel("accumulate_normals");
    let norm_normals_pipeline = queue_kernel("normalize_normals");

    commands.insert_resource(SimPipelineData {
        layout,
        predict_pipeline,
        reset_lagrange_pipeline,
        copy_mid_pos_pipeline,
        arap_pipeline,
        update_vel_pipeline,
        clear_normals_pipeline,
        accum_normals_pipeline,
        norm_normals_pipeline,
    });
}

#[derive(Resource)]
pub struct PhysDataBindGroups {
    pub generation: u64,
    pub bind_group: BindGroup,
}

fn check_pipeline_ready(
    pipeline: Res<SimPipelineData>,
    cache: Res<PipelineCache>,
    mut ready: ResMut<SimPipelineReady>,
) {
    if ready.0 {
        return;
    }
    let mut all_ok = true;
    for id in pipeline.all_ids() {
        match cache.get_compute_pipeline_state(id) {
            CachedPipelineState::Ok(_) => {}
            CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_))
            | CachedPipelineState::Err(ShaderCacheError::ShaderImportNotYetAvailable) => {
                all_ok = false;
            }
            CachedPipelineState::Err(err) => {
                panic!("Failed compiling {ARAP_SHADER_ASSET_PATH}:\n{err}");
            }
            _ => {
                all_ok = false;
            }
        }
    }
    ready.0 = all_ok;
}

fn prepare_sim_params_buffer(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut sim_params_buffer: ResMut<SimParamsBuffer>,
    phys_data: Option<Res<PhysData>>,
    sim_config: Option<Res<SimConfig>>,
) {
    let default_cfg = SimConfig::default();
    let cfg = sim_config.as_deref().unwrap_or(&default_cfg);
    let (mu, lambda) = cfg.stiffness.lame_parameters();

    let p = sim_params_buffer.buffer.get_mut();
    p.dt = FIXED_FRAME_DT / (SUBSTEPS_PER_FRAME as f32);
    p.omega = 1.0;
    p.mu = mu;
    p.lambda = lambda;
    p.gravity_y = if cfg.gravity_enabled { -9.81 } else { 0.0 };
    p.damping = 0.25;
    p.floor_y = 0.0;
    p.friction = 0.60;

    if let Some(pd) = phys_data {
        p.nt = pd.nt;
        p.nv = pd.nv;
        p.n_surf_tris = pd.n_surf_tris;
        p.enable_floor = u32::from(pd.enable_floor);
    }

    sim_params_buffer
        .buffer
        .write_buffer(&render_device, &render_queue);
}

fn prepare_sim_data_bind_groups(
    mut commands: Commands,
    pipeline: Res<SimPipelineData>,
    pipeline_cache: Res<PipelineCache>,
    phys_data: Res<PhysData>,
    existing: Option<Res<PhysDataBindGroups>>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
    sim_params_buffer: Res<SimParamsBuffer>,
    render_device: Res<RenderDevice>,
) {
    if existing.as_ref().map(|b| b.generation) == Some(phys_data.generation) {
        return;
    }

    let (
        Some(pos_gpu),
        Some(normals_gpu),
        Some(verts_gpu),
        Some(tets_gpu),
        Some(tet_state_gpu),
        Some(surf_tris_gpu),
        Some(params_gpu),
    ) = (
        gpu_buffers.get(&phys_data.pos),
        gpu_buffers.get(&phys_data.normals),
        gpu_buffers.get(&phys_data.verts),
        gpu_buffers.get(&phys_data.tets),
        gpu_buffers.get(&phys_data.tet_state),
        gpu_buffers.get(&phys_data.surface_tris),
        sim_params_buffer.buffer.buffer(),
    )
    else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        "xpbd_phys_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            pos_gpu.buffer.as_entire_binding(),
            normals_gpu.buffer.as_entire_binding(),
            verts_gpu.buffer.as_entire_binding(),
            tets_gpu.buffer.as_entire_binding(),
            tet_state_gpu.buffer.as_entire_binding(),
            surf_tris_gpu.buffer.as_entire_binding(),
            params_gpu.as_entire_binding(),
        )),
    );

    commands.insert_resource(PhysDataBindGroups {
        generation: phys_data.generation,
        bind_group,
    });
}

fn run_xpbd_simulation(
    mut render_context: RenderContext,
    ready: Res<SimPipelineReady>,
    phys_data: Option<Res<PhysData>>,
    bind_groups: Option<Res<PhysDataBindGroups>>,
    sim_config: Option<Res<SimConfig>>,
    cache: Res<PipelineCache>,
    p: Res<SimPipelineData>,
) {
    if !ready.0 {
        return;
    }
    let (Some(phys_data), Some(bind_groups)) = (phys_data, bind_groups) else {
        return;
    };
    if bind_groups.generation != phys_data.generation {
        return;
    }
    if sim_config.map(|c| c.paused).unwrap_or(false) {
        return;
    }

    let (
        Some(predict_pl),
        Some(reset_pl),
        Some(copy_mid_pl),
        Some(arap_pl),
        Some(update_vel_pl),
        Some(clear_n_pl),
        Some(accum_n_pl),
        Some(norm_n_pl),
    ) = (
        cache.get_compute_pipeline(p.predict_pipeline),
        cache.get_compute_pipeline(p.reset_lagrange_pipeline),
        cache.get_compute_pipeline(p.copy_mid_pos_pipeline),
        cache.get_compute_pipeline(p.arap_pipeline),
        cache.get_compute_pipeline(p.update_vel_pipeline),
        cache.get_compute_pipeline(p.clear_normals_pipeline),
        cache.get_compute_pipeline(p.accum_normals_pipeline),
        cache.get_compute_pipeline(p.norm_normals_pipeline),
    )
    else {
        return;
    };

    let wg_verts = phys_data.nv.div_ceil(WORKGROUP_SIZE);
    let wg_tets = phys_data.nt.div_ceil(WORKGROUP_SIZE);
    let wg_tris = phys_data.n_surf_tris.div_ceil(WORKGROUP_SIZE);

    let mut pass = render_context
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("xpbd_substep_pass"),
            timestamp_writes: None,
        });
    pass.set_bind_group(0, &bind_groups.bind_group, &[]);

    for _ in 0..SUBSTEPS_PER_FRAME {
        pass.set_pipeline(predict_pl);
        pass.dispatch_workgroups(wg_verts, 1, 1);

        pass.set_pipeline(reset_pl);
        pass.dispatch_workgroups(wg_tets, 1, 1);

        for _ in 0..JACOBI_ITERS_PER_SUBSTEP {
            pass.set_pipeline(copy_mid_pl);
            pass.dispatch_workgroups(wg_verts, 1, 1);

            pass.set_pipeline(arap_pl);
            pass.dispatch_workgroups(wg_tets, 1, 1);
        }

        pass.set_pipeline(update_vel_pl);
        pass.dispatch_workgroups(wg_verts, 1, 1);
    }

    if wg_tris > 0 {
        pass.set_pipeline(clear_n_pl);
        pass.dispatch_workgroups(wg_verts, 1, 1);

        pass.set_pipeline(accum_n_pl);
        pass.dispatch_workgroups(wg_tris, 1, 1);

        pass.set_pipeline(norm_n_pl);
        pass.dispatch_workgroups(wg_verts, 1, 1);
    }
}

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractResourcePlugin::<PhysData>::default(),
            ExtractResourcePlugin::<SimConfig>::default(),
        ));

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SimParamsBuffer>()
            .init_resource::<SimPipelineReady>()
            .add_systems(RenderStartup, init_sim_pipeline)
            .add_systems(
                Render,
                (
                    check_pipeline_ready.in_set(RenderSystems::Prepare),
                    prepare_sim_params_buffer.in_set(RenderSystems::PrepareResources),
                    prepare_sim_data_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups)
                        .run_if(resource_exists::<PhysData>),
                ),
            )
            .add_systems(RenderGraph, run_xpbd_simulation.before(camera_driver));
    }
}
