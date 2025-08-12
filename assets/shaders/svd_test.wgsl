
// Bindings for the input and output storage buffers.
@group(0) @binding(0) var<storage, read> input_matrices: array<Mat3x3f>;
@group(0) @binding(1) var<storage, read_write> output_results: array<SVDResult>;


// The entry point for the compute shader.
// Each invocation handles one matrix from the input array.
@compute @workgroup_size(64, 1, 1)
fn svd_test(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // let idx = global_id.x;
    // let num_matrices = arrayLength(&input_matrices);

    // if (idx >= num_matrices) {
    //     return;
    // }

    // // Unpack input matrix A
    // let A = input_matrices[idx];
    // let a11 = A.r1.x; let a12 = A.r1.y; let a13 = A.r1.z;
    // let a21 = A.r2.x; let a22 = A.r2.y; let a23 = A.r2.z;
    // let a31 = A.r3.x; let a32 = A.r3.y; let a33 = A.r3.z;

    // // Perform the SVD calculation
    // let result = svd3x3(
    //     a11, a12, a13, 
    //     a21, a22, a23, 
    //     a31, a32, a33
    // );
    
    // // Store the result
    // output_results[idx] = result;
}