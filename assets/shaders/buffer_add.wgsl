// assets/shaders/buffer_add.wgsl

struct Data {
    values: array<u32>,
};

@group(0) @binding(0) var<storage, read> input: Data;
@group(0) @binding(1) var<storage, read_write> output: Data;

@compute @workgroup_size(1)
fn add_one(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    output.values[index] = input.values[index] + 1u;
}