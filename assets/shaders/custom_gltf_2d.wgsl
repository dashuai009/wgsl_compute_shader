#import bevy_sprite::{
    mesh2d_view_bindings::globals,
    mesh2d_functions::{get_world_from_local, mesh2d_position_local_to_clip},
}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif
@group(2) @binding(100)
var<uniform> my_extended_material: MyExtendedMaterial;


struct VertexInput {
    @builtin(vertex_index) index: u32
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>
};


@vertex
fn vertex(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vertex.position.xyz1 + 0.01;
    return out;
}