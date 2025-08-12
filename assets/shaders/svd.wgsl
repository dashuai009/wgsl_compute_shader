


// Struct for a 3x3 matrix.
// Using 3 vec3s is idiomatic in WGSL.
struct SVD_Mat3x3 {
    r1: vec3<f32>,
    r2: vec3<f32>,
    r3: vec3<f32>,
}

// Struct for the SVD result.
// S is a diagonal matrix, so we only store the diagonal elements.
struct SVDResult {
    U: SVD_Mat3x3,
    S: vec3<f32>,
    V: SVD_Mat3x3,
}


// // Pre-computed constants from the original code.
// // These are used in the Jacobi and Givens rotation calculations.
// let gsine_pi_over_eight : f32 = bitcast<f32>(1053028117u); // 0.38268346
// let gcosine_pi_over_eight : f32 = bitcast<f32>(1064076127u); // 0.9238795
// let gsmall_number : f32 = 1.0e-12;
// let gtiny_number : f32 = 1.0e-20;
// let gfour_gamma_squared : f32 = 5.8284273;

// Helper function for a conditional swap. Replaces the XOR-swap trick.
fn cond_swap(a: f32, b: f32, cond: bool) -> vec2<f32> {
    return vec2<f32>(select(a, b, cond), select(b, a, cond));
}

// The core SVD function, translated from the CUDA source.
fn svd3x3(
    in_a11: f32, in_a12: f32, in_a13: f32, 
    in_a21: f32, in_a22: f32, in_a23: f32, 
    in_a31: f32, in_a32: f32, in_a33: f32
) -> SVDResult {

    // // Assign inputs to mutable variables
    // var a11 = in_a11; var a12 = in_a12; var a13 = in_a13;
    // var a21 = in_a21; var a22 = in_a22; var a23 = in_a23;
    // var a31 = in_a31; var a32 = in_a32; var a33 = in_a33;

    // // Intermediate variables
    // var tmp1: f32; var tmp2: f32; var tmp3: f32; var tmp4: f32; var tmp5: f32;
    // var c: f32; var s: f32; var ch: f32; var sh: f32;

    // // Matrix S = A^T * A
    // var s11: f32; var s21: f32; var s31: f32; 
    // var s22: f32; var s32: f32; var s33: f32;

    // // Quaternion for V
    // var qvs: f32; var qvvx: f32; var qvvy: f32; var qvvz: f32;
    
    // //###########################################################
    // // 1. Compute normal equations matrix S = A^T * A
    // //###########################################################
    // s11 = a11*a11 + a21*a21 + a31*a31;
    // s21 = a12*a11 + a22*a21 + a32*a31;
    // s31 = a13*a11 + a23*a21 + a33*a31;
    // s22 = a12*a12 + a22*a22 + a32*a32;
    // s32 = a13*a12 + a23*a22 + a33*a32;
    // s33 = a13*a13 + a23*a23 + a33*a33;

    // qvs = 1.0; qvvx = 0.0; qvvy = 0.0; qvvz = 0.0;

    // //###########################################################
    // // 2. Solve symmetric eigenproblem for S using 4 Jacobi sweeps
    // //###########################################################
    // for (var i = 0; i < 4; i = i + 1) {
    //     // Jacobi rotation for the (1,2) off-diagonal element
    //     sh = s21 * 0.5;
    //     tmp5 = s11 - s22;
    //     tmp2 = sh * sh;
    //     var cond = tmp2 >= gtiny_number;
    //     sh = select(0.0, sh, cond);
    //     ch = select(1.0, tmp5, cond);
    //     tmp1 = sh * sh;
    //     tmp2 = ch * ch;
    //     tmp3 = tmp1 + tmp2;
    //     tmp4 = inverseSqrt(tmp3);
    //     sh = tmp4 * sh;
    //     ch = tmp4 * ch;
    //     cond = (ch * ch) <= (gfour_gamma_squared * (sh * sh));
    //     sh = select(sh, gsine_pi_over_eight, cond);
    //     ch = select(ch, gcosine_pi_over_eight, cond);
    //     tmp1 = sh * sh;
    //     tmp2 = ch * ch;
    //     c = tmp2 - tmp1;
    //     s = 2.0 * ch * sh;

    //     // Givens conjugation S' = G^T * S * G
    //     tmp3 = tmp1 + tmp2;
    //     s33 = s33 * tmp3; s31 = s31 * tmp3; s32 = s32 * tmp3; s33 = s33 * tmp3; // Note: s33 updated twice in original
    //     tmp1 = s * s31; tmp2 = s * s32;
    //     s31 = c * s31 + tmp2;
    //     s32 = c * s32 - tmp1;
    //     tmp2 = s * s;
    //     tmp1 = s22 * tmp2; tmp3 = s11 * tmp2; tmp4 = c * c;
    //     s11 = s11 * tmp4 + tmp1;
    //     s22 = s22 * tmp4 + tmp3;
    //     tmp4 = tmp4 - tmp2;
    //     tmp2 = 2.0 * s21;
    //     s21 = s21 * tmp4;
    //     tmp4 = c * s;
    //     tmp2 = tmp2 * tmp4;
    //     tmp5 = tmp5 * tmp4;
    //     s11 = s11 + tmp2;
    //     s21 = s21 - tmp5;
    //     s22 = s22 - tmp2;
        
    //     // Update quaternion for V
    //     tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
    //     qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
    //     qvvz = qvvz + sh;
    //     qvs = qvs - tmp3;
    //     qvvx = qvvx + tmp2;
    //     qvvy = qvvy - tmp1;

    //     // Jacobi rotation for the (2,3) off-diagonal element
    //     sh = s32 * 0.5;
    //     tmp5 = s22 - s33;
    //     tmp2 = sh * sh;
    //     cond = tmp2 >= gtiny_number;
    //     sh = select(0.0, sh, cond);
    //     ch = select(1.0, tmp5, cond);
    //     tmp1 = sh * sh;
    //     tmp2 = ch * ch;
    //     tmp3 = tmp1 + tmp2;
    //     tmp4 = inverseSqrt(tmp3);
    //     sh = tmp4 * sh;
    //     ch = tmp4 * ch;
    //     cond = (ch * ch) <= (gfour_gamma_squared * (sh * sh));
    //     sh = select(sh, gsine_pi_over_eight, cond);
    //     ch = select(ch, gcosine_pi_over_eight, cond);
    //     tmp1 = sh * sh;
    //     tmp2 = ch * ch;
    //     c = tmp2 - tmp1;
    //     s = 2.0 * ch * sh;

    //     // Givens conjugation
    //     tmp3 = tmp1 + tmp2;
    //     s11 = s11 * tmp3; s21 = s21 * tmp3; s31 = s31 * tmp3; s11 = s11 * tmp3; // Note: s11 updated twice
    //     tmp1 = s * s21; tmp2 = s * s31;
    //     s21 = c * s21 + tmp2;
    //     s31 = c * s31 - tmp1;
    //     tmp2 = s * s;
    //     tmp1 = s33 * tmp2; tmp3 = s22 * tmp2; tmp4 = c * c;
    //     s22 = s22 * tmp4 + tmp1;
    //     s33 = s33 * tmp4 + tmp3;
    //     tmp4 = tmp4 - tmp2;
    //     tmp2 = 2.0 * s32;
    //     s32 = s32 * tmp4;
    //     tmp4 = c * s;
    //     tmp2 = tmp2 * tmp4;
    //     tmp5 = tmp5 * tmp4;
    //     s22 = s22 + tmp2;
    //     s32 = s32 - tmp5;
    //     s33 = s33 - tmp2;

    //     // Update quaternion
    //     tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
    //     qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
    //     qvvx = qvvx + sh;
    //     qvs = qvs - tmp1;
    //     qvvy = qvvy + tmp3;
    //     qvvz = qvvz - tmp2;

    //     // Jacobi rotation for the (1,3) off-diagonal element
    //     sh = s31 * 0.5;
    //     tmp5 = s33 - s11;
    //     tmp2 = sh * sh;
    //     cond = tmp2 >= gtiny_number;
    //     sh = select(0.0, sh, cond);
    //     ch = select(1.0, tmp5, cond);
    //     tmp1 = sh * sh;
    //     tmp2 = ch * ch;
    //     tmp3 = tmp1 + tmp2;
    //     tmp4 = inverseSqrt(tmp3);
    //     sh = tmp4 * sh;
    //     ch = tmp4 * ch;
    //     cond = (ch * ch) <= (gfour_gamma_squared * (sh * sh));
    //     sh = select(sh, gsine_pi_over_eight, cond);
    //     ch = select(ch, gcosine_pi_over_eight, cond);
    //     tmp1 = sh * sh;
    //     tmp2 = ch * ch;
    //     c = tmp2 - tmp1;
    //     s = 2.0 * ch * sh;

    //     // Givens conjugation
    //     tmp3 = tmp1 + tmp2;
    //     s22 = s22 * tmp3; s32 = s32 * tmp3; s21 = s21 * tmp3; s22 = s22 * tmp3; // Note: s22 updated twice
    //     tmp1 = s * s32; tmp2 = s * s21;
    //     s32 = c * s32 + tmp2;
    //     s21 = c * s21 - tmp1;
    //     tmp2 = s * s;
    //     tmp1 = s11 * tmp2; tmp3 = s33 * tmp2; tmp4 = c * c;
    //     s33 = s33 * tmp4 + tmp1;
    //     s11 = s11 * tmp4 + tmp3;
    //     tmp4 = tmp4 - tmp2;
    //     tmp2 = 2.0 * s31;
    //     s31 = s31 * tmp4;
    //     tmp4 = c * s;
    //     tmp2 = tmp2 * tmp4;
    //     tmp5 = tmp5 * tmp4;
    //     s33 = s33 + tmp2;
    //     s31 = s31 - tmp5;
    //     s11 = s11 - tmp2;

    //     // Update quaternion
    //     tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
    //     qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
    //     qvvy = qvvy + sh;
    //     qvs = qvs - tmp2;
    //     qvvz = qvvz + tmp1;
    //     qvvx = qvvx - tmp3;
    // }

    // //###########################################################
    // // 3. Normalize and convert quaternion to matrix V
    // //###########################################################
    // tmp2 = qvs*qvs + qvvx*qvvx + qvvy*qvvy + qvvz*qvvz;
    // // Refine reciprocal sqrt with one Newton-Raphson iteration
    // tmp1 = inverseSqrt(tmp2);
    // tmp4 = tmp1 * 0.5;
    // tmp3 = tmp1 * tmp4;
    // tmp3 = tmp1 * tmp3;
    // tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;

    // qvs = qvs * tmp1; qvvx = qvvx * tmp1; qvvy = qvvy * tmp1; qvvz = qvvz * tmp1;

    // // Convert quaternion to matrix V
    // var v11: f32; var v12: f32; var v13: f32;
    // var v21: f32; var v22: f32; var v23: f32;
    // var v31: f32; var v32: f32; var v33: f32;
    
    // tmp1 = qvvx * qvvx; tmp2 = qvvy * qvvy; tmp3 = qvvz * qvvz;
    // v11 = qvs * qvs;
    // v22 = v11 - tmp1 + tmp2 - tmp3;
    // v33 = v11 - tmp1 - tmp2 + tmp3;
    // v11 = v11 + tmp1 - tmp2 - tmp3;
    // tmp1 = 2.0 * qvvx; tmp2 = 2.0 * qvvy; tmp3 = 2.0 * qvvz;
    // v32 = qvs * tmp1; v13 = qvs * tmp2; v21 = qvs * tmp3;
    // tmp1 = qvvy * tmp1; tmp2 = qvvz * tmp2; tmp3 = qvvx * tmp3;
    // v12 = tmp1 - v21; v23 = tmp2 - v32; v31 = tmp3 - v13;
    // v21 = tmp1 + v21; v32 = tmp2 + v32; v13 = tmp3 + v13;

    // //###########################################################
    // // 4. Compute B = A * V
    // //###########################################################
    // // Row 1
    // tmp1 = a11; tmp2 = a12; tmp3 = a13;
    // a11 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    // a12 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    // a13 = tmp1*v13 + tmp2*v23 + tmp3*v33;
    // // Row 2
    // tmp1 = a21; tmp2 = a22; tmp3 = a23;
    // a21 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    // a22 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    // a23 = tmp1*v13 + tmp2*v23 + tmp3*v33;
    // // Row 3
    // tmp1 = a31; tmp2 = a32; tmp3 = a33;
    // a31 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    // a32 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    // a33 = tmp1*v13 + tmp2*v23 + tmp3*v33;
    
    // //###########################################################
    // // 5. Sort columns of B and V to ensure descending singular values
    // //###########################################################
    // var c1_norm_sq = a11*a11 + a21*a21 + a31*a31;
    // var c2_norm_sq = a12*a12 + a22*a22 + a32*a32;
    // var c3_norm_sq = a13*a13 + a23*a23 + a33*a33;
    
    // // Swap columns 1 and 2
    // var swap_cond = c1_norm_sq < c2_norm_sq;
    // var swap_res = cond_swap(a11, a12, swap_cond); a11 = swap_res.x; a12 = swap_res.y;
    // swap_res = cond_swap(a21, a22, swap_cond); a21 = swap_res.x; a22 = swap_res.y;
    // swap_res = cond_swap(a31, a32, swap_cond); a31 = swap_res.x; a32 = swap_res.y;
    // swap_res = cond_swap(v11, v12, swap_cond); v11 = swap_res.x; v12 = swap_res.y;
    // swap_res = cond_swap(v21, v22, swap_cond); v21 = swap_res.x; v22 = swap_res.y;
    // swap_res = cond_swap(v31, v32, swap_cond); v31 = swap_res.x; v32 = swap_res.y;
    // swap_res = cond_swap(c1_norm_sq, c2_norm_sq, swap_cond); c1_norm_sq = swap_res.x; c2_norm_sq = swap_res.y;
    // var multiplier = select(1.0, -1.0, swap_cond);
    // a12 *= multiplier; a22 *= multiplier; a32 *= multiplier;
    // v12 *= multiplier; v22 *= multiplier; v32 *= multiplier;

    // // Swap columns 1 and 3
    // swap_cond = c1_norm_sq < c3_norm_sq;
    // swap_res = cond_swap(a11, a13, swap_cond); a11 = swap_res.x; a13 = swap_res.y;
    // swap_res = cond_swap(a21, a23, swap_cond); a21 = swap_res.x; a23 = swap_res.y;
    // swap_res = cond_swap(a31, a33, swap_cond); a31 = swap_res.x; a33 = swap_res.y;
    // swap_res = cond_swap(v11, v13, swap_cond); v11 = swap_res.x; v13 = swap_res.y;
    // swap_res = cond_swap(v21, v23, swap_cond); v21 = swap_res.x; v23 = swap_res.y;
    // swap_res = cond_swap(v31, v33, swap_cond); v31 = swap_res.x; v33 = swap_res.y;
    // swap_res = cond_swap(c1_norm_sq, c3_norm_sq, swap_cond); c1_norm_sq = swap_res.x; c3_norm_sq = swap_res.y;
    // multiplier = select(1.0, -1.0, swap_cond);
    // a11 *= multiplier; a21 *= multiplier; a31 *= multiplier;
    // v11 *= multiplier; v21 *= multiplier; v31 *= multiplier;

    // // Swap columns 2 and 3
    // swap_cond = c2_norm_sq < c3_norm_sq;
    // swap_res = cond_swap(a12, a13, swap_cond); a12 = swap_res.x; a13 = swap_res.y;
    // swap_res = cond_swap(a22, a23, swap_cond); a22 = swap_res.x; a23 = swap_res.y;
    // swap_res = cond_swap(a32, a33, swap_cond); a32 = swap_res.x; a33 = swap_res.y;
    // swap_res = cond_swap(v12, v13, swap_cond); v12 = swap_res.x; v13 = swap_res.y;
    // swap_res = cond_swap(v22, v23, swap_cond); v22 = swap_res.x; v23 = swap_res.y;
    // swap_res = cond_swap(v32, v33, swap_cond); v32 = swap_res.x; v33 = swap_res.y;
    // swap_res = cond_swap(c2_norm_sq, c3_norm_sq, swap_cond); c2_norm_sq = swap_res.x; c3_norm_sq = swap_res.y;
    // multiplier = select(1.0, -1.0, swap_cond);
    // a13 *= multiplier; a23 *= multiplier; a33 *= multiplier;
    // v13 *= multiplier; v23 *= multiplier; v33 *= multiplier;

    // //###########################################################
    // // 6. QR factorization of B (= U * S) using Givens rotations
    // //###########################################################
    // var u11 = 1.0; var u12 = 0.0; var u13 = 0.0;
    // var u21 = 0.0; var u22 = 1.0; var u23 = 0.0;
    // var u31 = 0.0; var __u32 = 0.0; var u33 = 1.0;
    
    // // Givens rotation 1 (zeroes a21)
    // cond = (a21*a21) >= gsmall_number;
    // sh = select(0.0, a21, cond);
    // tmp5 = select(1.0, 0.0, a11 >= 0.0);
    // ch = max(max(-a11, a11), gsmall_number);
    // tmp2 = sh*sh + ch*ch;
    // tmp1 = inverseSqrt(tmp2);
    // // Refine 1/sqrt
    // tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;
    // tmp1 *= tmp2;
    // ch += tmp1;
    // tmp1 = select(sh, ch, tmp5 == 0.0);
    // tmp2 = select(ch, sh, tmp5 == 0.0);
    // ch = tmp1; sh = tmp2;
    // tmp2 = sh*sh + ch*ch;
    // tmp1 = inverseSqrt(tmp2);
    // // Refine 1/sqrt
    // tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;
    // ch *= tmp1; sh *= tmp1;
    // c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    // // Rotate A and U
    // tmp1 = s*a11; tmp2 = s*a21; a11 = c*a11 + tmp2; a21 = c*a21 - tmp1;
    // tmp1 = s*a12; tmp2 = s*a22; a12 = c*a12 + tmp2; a22 = c*a22 - tmp1;
    // tmp1 = s*a13; tmp2 = s*a23; a13 = c*a13 + tmp2; a23 = c*a23 - tmp1;
    // tmp1 = s*u11; tmp2 = s*u12; u11 = c*u11 + tmp2; u12 = c*u12 - tmp1;
    // tmp1 = s*u21; tmp2 = s*u22; u21 = c*u21 + tmp2; u22 = c*u22 - tmp1;
    // tmp1 = s*u31; tmp2 = s*__u32; u31 = c*u31 + tmp2; _u32 = c*_u32 - tmp1;

    // // Givens rotation 2 (zeroes a31)
    // cond = (a31*a31) >= gsmall_number;
    // sh = select(0.0, a31, cond);
    // tmp5 = select(1.0, 0.0, a11 >= 0.0);
    // ch = max(max(-a11, a11), gsmall_number);
    // tmp2 = sh*sh + ch*ch;
    // tmp1 = inverseSqrt(tmp2);
    // // Refine 1/sqrt
    // tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;
    // tmp1 *= tmp2;
    // ch += tmp1;
    // tmp1 = select(sh, ch, tmp5 == 0.0);
    // tmp2 = select(ch, sh, tmp5 == 0.0);
    // ch = tmp1; sh = tmp2;
    // tmp2 = sh*sh + ch*ch;
    // tmp1 = inverseSqrt(tmp2);
    // // Refine 1/sqrt
    // tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;
    // ch *= tmp1; sh *= tmp1;
    // c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    // // Rotate A and U
    // tmp1 = s*a11; tmp2 = s*a31; a11 = c*a11 + tmp2; a31 = c*a31 - tmp1;
    // tmp1 = s*a12; tmp2 = s*a32; a12 = c*a12 + tmp2; a32 = c*a32 - tmp1;
    // tmp1 = s*a13; tmp2 = s*a33; a13 = c*a13 + tmp2; a33 = c*a33 - tmp1;
    // tmp1 = s*u11; tmp2 = s*u13; u11 = c*u11 + tmp2; u13 = c*u13 - tmp1;
    // tmp1 = s*u21; tmp2 = s*u23; u21 = c*u21 + tmp2; u23 = c*u23 - tmp1;
    // tmp1 = s*u31; tmp2 = s*u33; u31 = c*u31 + tmp2; u33 = c*u33 - tmp1;

    // // Givens rotation 3 (zeroes a32)
    // cond = (a32*a32) >= gsmall_number;
    // sh = select(0.0, a32, cond);
    // tmp5 = select(1.0, 0.0, a22 >= 0.0);
    // ch = max(max(-a22, a22), gsmall_number);
    // tmp2 = sh*sh + ch*ch;
    // tmp1 = inverseSqrt(tmp2);
    // // Refine 1/sqrt
    // tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;
    // tmp1 *= tmp2;
    // ch += tmp1;
    // tmp1 = select(sh, ch, tmp5 == 0.0);
    // tmp2 = select(ch, sh, tmp5 == 0.0);
    // ch = tmp1; sh = tmp2;
    // tmp2 = sh*sh + ch*ch;
    // tmp1 = inverseSqrt(tmp2);
    // // Refine 1/sqrt
    // tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    // tmp1 = tmp1 + tmp4 - tmp3;
    // ch *= tmp1; sh *= tmp1;
    // c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    // // Rotate A and U
    // tmp1 = s*a21; tmp2 = s*a31; a21 = c*a21 + tmp2; a31 = c*a31 - tmp1;
    // tmp1 = s*a22; tmp2 = s*a32; a22 = c*a22 + tmp2; a32 = c*a32 - tmp1;
    // tmp1 = s*a23; tmp2 = s*a33; a23 = c*a23 + tmp2; a33 = c*a33 - tmp1;
    // tmp1 = s*u12; tmp2 = s*u13; u12 = c*u12 + tmp2; u13 = c*u13 - tmp1;
    // tmp1 = s*u22; tmp2 = s*u23; u22 = c*u22 + tmp2; u23 = c*u23 - tmp1;
    // tmp1 = s*_u32; tmp2 = s*u33; _u32 = c*_u32 + tmp2; u33 = c*u33 - tmp1;
    
    // // Package and return the results
    // var result: SVDResult;
    // result.U.r1 = vec3(u11, u12, u13);
    // result.U.r2 = vec3(u21, u22, u23);
    // result.U.r3 = vec3(u31, _u32, u33);
    // result.V.r1 = vec3(v11, v12, v13);
    // result.V.r2 = vec3(v21, v22, v23);
    // result.V.r3 = vec3(v31, v32, v33);
    // result.S = vec3(a11, a22, a33);
    
    // return result;
}
