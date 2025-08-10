@group(0) @binding(0) var<storage, read_write> pos: array<f32>;
@group(0) @binding(1) var<storage, read> tet_indices: array<u32>;

@compute @workgroup_size(1)
fn add_one(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    pos[index] = pos[index] + 1.0;
}