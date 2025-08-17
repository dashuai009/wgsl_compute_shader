#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip};
#import bevy_pbr::mesh_bindings::mesh;
#ifdef PREPASS_PIPELINE
// https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/prepass/prepass_io.wgsl
#import bevy_pbr::{
    prepass_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
// https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/render/forward_io.wgsl
#import bevy_pbr::{
    forward_io::{Vertex, VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(2) @binding(100) var<storage, read_write> tet_pos: array<f32>;
@group(2) @binding(101) var<storage, read_write> to_tet_indices: array<u32>;


// ===== 顶点着色器：只改位置，其余照常填充 =====
@vertex
fn vertex(v: Vertex, @builtin(vertex_index) vid: u32) -> VertexOutput {
    var out: VertexOutput;

    // 物体的 world_from_local（实例变换）
    let world_from_local = get_world_from_local(v.instance_index);


    let first_vertex = mesh[v.instance_index].first_vertex_index;
    let vvid = vid - first_vertex;

    // 本地位置（扩成 vec4），加一个小偏移（只动 xyz，w 仍为 1）
    // let vertex_id = tet_indices[vid];     // 先查索引
    // let base = 3u * vertex_id;            // 再定位位置
    let base = 3u * vvid;                      // xyz 连续存放
    let p = vec3<f32>(tet_pos[base], tet_pos[base+1u], tet_pos[base+2u]);
    let local_pos4 =
        vec4<f32>(p + vec3<f32>(0, 0, 0), 1.0);

    // 裁剪空间位置
    out.position = mesh_position_local_to_clip(world_from_local, local_pos4);

    // 世界空间位置（你的输出是 vec4<f32>）
    out.world_position = world_from_local * local_pos4;

#ifdef MOTION_VECTOR_PREPASS
    // 运动矢量需要上一帧的 world_from_local
    let prev_world_from_local = get_previous_world_from_local(v.instance_index);
    out.previous_world_position = prev_world_from_local * local_pos4;
#endif


#ifdef VERTEX_NORMALS
    // 简化：直接用 world_from_local 的线性部分处理法线（非均匀缩放场景下请改为 normal matrix）
    let n4 = (world_from_local * vec4<f32>(v.normal, 0.0));
    out.world_normal = normalize(n4.xyz);
#endif


#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
  #ifdef VERTEX_NORMALS
    // 世界空间法线：把 (n,0) 用同一矩阵变换（多数情况下可行；严格应使用 normal matrix）
    let wn4 = world_from_local * vec4<f32>(v.normal, 0.0);
    out.world_normal = normalize(wn4.xyz);
  #endif

  #ifdef VERTEX_TANGENTS
    // 切线（w 为手性），副切线 = n × t * 手性
    let wt4 = world_from_local * vec4<f32>(v.tangent.xyz, 0.0);
    let wt = normalize(wt4.xyz);
    // 若没有 NORMAL 宏，也要避免未定义变量，这里简单从 v.normal 推导或设默认
  #ifdef VERTEX_NORMALS
    let wb = normalize(cross(out.world_normal, wt)) * v.tangent.w;
  #else
    // 无法线时，给个兜底（不常见场景）
    let wb = vec3<f32>(0.0, 0.0, 0.0);
  #endif
    out.world_tangent = vec4<f32>(wt, v.tangent.w);
  #endif // VERTEX_TANGENTS
#endif // NORMAL_PREPASS_OR_DEFERRED_PREPASS

#ifdef VERTEX_UVS_A
    out.uv = v.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = v.uv_b;
#endif


#ifdef VERTEX_TANGENTS
    let t4 = world_from_local * vec4<f32>(v.tangent.xyz, 0.0);
    out.world_tangent = vec4<f32>(normalize(t4.xyz), v.tangent.w);
#endif


#ifdef VERTEX_COLORS
    out.color = v.color;
#endif

#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = v.instance_index;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = 0; // 如有需要可按实例自定义
#endif



#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    // 简单做法：用裁剪空间的 z 值；如果你的管线有特殊约定，可替换
    out.unclipped_depth = out.position.z;
#endif

#ifdef VISIBILITY_RANGE_DITHER
    // 给一个稳定的抖动值（示例：基于实例索引的 hash）
    let h = (v.instance_index * 1664525u + 1013904223u) & 1023u;
    out.visibility_range_dither = i32(h);
#endif

    return out;
}