use bevy::prelude::*;
use bevy::render::mesh::{MeshVertexAttribute, VertexAttributeValues};
use bevy::render::render_resource::VertexFormat;
use bevy::gltf::GltfPlugin;


pub const ATTR_TET_POS: MeshVertexAttribute =
    MeshVertexAttribute::new("pos", 1_234_567_890, VertexFormat::Float32); // tet pos buffer
pub const ATTR_TET_INDICES: MeshVertexAttribute =
MeshVertexAttribute::new("tet_indices", 1_234_567_891, VertexFormat::Uint32); // tet indices