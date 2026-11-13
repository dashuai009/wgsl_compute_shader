@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::{Vertex, VertexOutput, decompress_vertex};
@else
import bevy_pbr::render::forward_io::{Vertex, VertexOutput, decompress_vertex};

import bevy_pbr::render::{
    mesh_bindings::mesh,
    mesh_functions,
    view_transformations::position_world_to_clip,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<storage, read> vertex_pos: array<u32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<storage, read> vertex_normals: array<u32>;

fn load_pos(vid: u32) -> vec3<f32> {
    let base = 3u * vid;
    return vec3<f32>(
        bitcast<f32>(vertex_pos[base + 0u]),
        bitcast<f32>(vertex_pos[base + 1u]),
        bitcast<f32>(vertex_pos[base + 2u]),
    );
}

fn load_normal(vid: u32) -> vec3<f32> {
    let base = 3u * vid;
    return vec3<f32>(
        bitcast<f32>(vertex_normals[base + 0u]),
        bitcast<f32>(vertex_normals[base + 1u]),
        bitcast<f32>(vertex_normals[base + 2u]),
    );
}

@vertex
fn vertex(
    vertex_in: Vertex,
    @if(!MORPH_TARGETS) @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let vertex = decompress_vertex(vertex_in, vertex_in.instance_index);
    let first_vertex = mesh[vertex_in.instance_index].first_vertex_index;
@if(MORPH_TARGETS)
    let raw_vid = vertex.index;
@else
    let raw_vid = vertex_index;
    let local_vid = raw_vid - first_vertex;

    let world_from_local = mesh_functions::get_world_from_local(vertex_in.instance_index);
    let dyn_pos = load_pos(local_vid);
    let dyn_normal = load_normal(local_vid);

    out.world_position = mesh_functions::mesh_position_local_to_world(
        world_from_local,
        vec4<f32>(dyn_pos, 1.0),
    );
    out.position = position_world_to_clip(out.world_position.xyz);

@if(PREPASS_PIPELINE) {
@if(UNCLIPPED_DEPTH_ORTHO_EMULATION) {
    out.unclipped_depth = out.position.z;
    out.position.z = min(out.position.z, 1.0);
}
@if(NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        dyn_normal,
        vertex_in.instance_index,
    );
@if(VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex_in.instance_index,
    );
}
@if(MOTION_VECTOR_PREPASS) {
    let prev_model = mesh_functions::get_previous_world_from_local(vertex_in.instance_index);
    out.previous_world_position = mesh_functions::mesh_position_local_to_world(
        prev_model,
        vec4<f32>(dyn_pos, 1.0),
    );
}
} @else {
    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        dyn_normal,
        vertex_in.instance_index,
    );
@if(VERTEX_TANGENTS)
    out.world_tangent = mesh_functions::mesh_tangent_local_to_world(
        world_from_local,
        vertex.tangent,
        vertex_in.instance_index,
    );
}

@if(VERTEX_UVS_A)
    out.uv = vertex.uv;
@if(VERTEX_UVS_B)
    out.uv_b = vertex.uv_b;
@if(VERTEX_COLORS)
    out.color = vertex.color;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex_in.instance_index;
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = mesh_functions::get_visibility_range_dither_level(
        vertex_in.instance_index,
        world_from_local[3],
    );

    return out;
}
