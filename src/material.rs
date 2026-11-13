use bevy::pbr::MaterialExtension;
use bevy::prelude::*;
use bevy::render::render_resource::AsBindGroup;
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;

pub const VERTEX_SHADER_ASSET_PATH: &str = "shaders/sim_pipeline.wesl";

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct SoftBodyMaterialExt {
    #[storage(100, read_only, visibility(vertex))]
    pub vertex_pos: Handle<ShaderBuffer>,
    #[storage(101, read_only, visibility(vertex))]
    pub vertex_normals: Handle<ShaderBuffer>,
}

impl MaterialExtension for SoftBodyMaterialExt {
    fn vertex_shader() -> ShaderRef {
        VERTEX_SHADER_ASSET_PATH.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        VERTEX_SHADER_ASSET_PATH.into()
    }
}
