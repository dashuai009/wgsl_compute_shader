

// ---------- Common types & helpers ----------
alias Mat3x3Row = array<f32, 9>;
alias Grad12    = array<f32, 12>;  // 4*vec3
// =================================== svd =========================

// Struct for the SVD result.
// S is a diagonal matrix, so we only store the diagonal elements.
struct SVDResult {
    U: Mat3x3Row,
    S: vec3<f32>,
    V: Mat3x3Row,
}


// Pre-computed constants from the original code.
// These are used in the Jacobi and Givens rotation calculations.
let gsine_pi_over_eight : f32 = bitcast<f32>(1053028117u); // 0.38268346
let gcosine_pi_over_eight : f32 = bitcast<f32>(1064076127u); // 0.9238795
let gsmall_number : f32 = 1.0e-12;
let gtiny_number : f32 = 1.0e-20;
let gfour_gamma_squared : f32 = 5.8284273;

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

    // Assign inputs to mutable variables
    var a11 = in_a11; var a12 = in_a12; var a13 = in_a13;
    var a21 = in_a21; var a22 = in_a22; var a23 = in_a23;
    var a31 = in_a31; var a32 = in_a32; var a33 = in_a33;

    // Intermediate variables
    var tmp1: f32; var tmp2: f32; var tmp3: f32; var tmp4: f32; var tmp5: f32;
    var c: f32; var s: f32; var ch: f32; var sh: f32;

    // Matrix S = A^T * A
    var s11: f32; var s21: f32; var s31: f32; 
    var s22: f32; var s32: f32; var s33: f32;

    // Quaternion for V
    var qvs: f32; var qvvx: f32; var qvvy: f32; var qvvz: f32;
    
    //###########################################################
    // 1. Compute normal equations matrix S = A^T * A
    //###########################################################
    s11 = a11*a11 + a21*a21 + a31*a31;
    s21 = a12*a11 + a22*a21 + a32*a31;
    s31 = a13*a11 + a23*a21 + a33*a31;
    s22 = a12*a12 + a22*a22 + a32*a32;
    s32 = a13*a12 + a23*a22 + a33*a32;
    s33 = a13*a13 + a23*a23 + a33*a33;

    qvs = 1.0; qvvx = 0.0; qvvy = 0.0; qvvz = 0.0;

    //###########################################################
    // 2. Solve symmetric eigenproblem for S using 4 Jacobi sweeps
    //###########################################################
    for (var i = 0; i < 4; i = i + 1) {
        // Jacobi rotation for the (1,2) off-diagonal element
        sh = s21 * 0.5;
        tmp5 = s11 - s22;
        tmp2 = sh * sh;
        var cond = tmp2 >= gtiny_number;
        sh = select(0.0, sh, cond);
        ch = select(1.0, tmp5, cond);
        tmp1 = sh * sh;
        tmp2 = ch * ch;
        tmp3 = tmp1 + tmp2;
        tmp4 = inverseSqrt(tmp3);
        sh = tmp4 * sh;
        ch = tmp4 * ch;
        cond = (ch * ch) <= (gfour_gamma_squared * (sh * sh));
        sh = select(sh, gsine_pi_over_eight, cond);
        ch = select(ch, gcosine_pi_over_eight, cond);
        tmp1 = sh * sh;
        tmp2 = ch * ch;
        c = tmp2 - tmp1;
        s = 2.0 * ch * sh;

        // Givens conjugation S' = G^T * S * G
        tmp3 = tmp1 + tmp2;
        s33 = s33 * tmp3; s31 = s31 * tmp3; s32 = s32 * tmp3; s33 = s33 * tmp3; // Note: s33 updated twice in original
        tmp1 = s * s31; tmp2 = s * s32;
        s31 = c * s31 + tmp2;
        s32 = c * s32 - tmp1;
        tmp2 = s * s;
        tmp1 = s22 * tmp2; tmp3 = s11 * tmp2; tmp4 = c * c;
        s11 = s11 * tmp4 + tmp1;
        s22 = s22 * tmp4 + tmp3;
        tmp4 = tmp4 - tmp2;
        tmp2 = 2.0 * s21;
        s21 = s21 * tmp4;
        tmp4 = c * s;
        tmp2 = tmp2 * tmp4;
        tmp5 = tmp5 * tmp4;
        s11 = s11 + tmp2;
        s21 = s21 - tmp5;
        s22 = s22 - tmp2;
        
        // Update quaternion for V
        tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
        qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
        qvvz = qvvz + sh;
        qvs = qvs - tmp3;
        qvvx = qvvx + tmp2;
        qvvy = qvvy - tmp1;

        // Jacobi rotation for the (2,3) off-diagonal element
        sh = s32 * 0.5;
        tmp5 = s22 - s33;
        tmp2 = sh * sh;
        cond = tmp2 >= gtiny_number;
        sh = select(0.0, sh, cond);
        ch = select(1.0, tmp5, cond);
        tmp1 = sh * sh;
        tmp2 = ch * ch;
        tmp3 = tmp1 + tmp2;
        tmp4 = inverseSqrt(tmp3);
        sh = tmp4 * sh;
        ch = tmp4 * ch;
        cond = (ch * ch) <= (gfour_gamma_squared * (sh * sh));
        sh = select(sh, gsine_pi_over_eight, cond);
        ch = select(ch, gcosine_pi_over_eight, cond);
        tmp1 = sh * sh;
        tmp2 = ch * ch;
        c = tmp2 - tmp1;
        s = 2.0 * ch * sh;

        // Givens conjugation
        tmp3 = tmp1 + tmp2;
        s11 = s11 * tmp3; s21 = s21 * tmp3; s31 = s31 * tmp3; s11 = s11 * tmp3; // Note: s11 updated twice
        tmp1 = s * s21; tmp2 = s * s31;
        s21 = c * s21 + tmp2;
        s31 = c * s31 - tmp1;
        tmp2 = s * s;
        tmp1 = s33 * tmp2; tmp3 = s22 * tmp2; tmp4 = c * c;
        s22 = s22 * tmp4 + tmp1;
        s33 = s33 * tmp4 + tmp3;
        tmp4 = tmp4 - tmp2;
        tmp2 = 2.0 * s32;
        s32 = s32 * tmp4;
        tmp4 = c * s;
        tmp2 = tmp2 * tmp4;
        tmp5 = tmp5 * tmp4;
        s22 = s22 + tmp2;
        s32 = s32 - tmp5;
        s33 = s33 - tmp2;

        // Update quaternion
        tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
        qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
        qvvx = qvvx + sh;
        qvs = qvs - tmp1;
        qvvy = qvvy + tmp3;
        qvvz = qvvz - tmp2;

        // Jacobi rotation for the (1,3) off-diagonal element
        sh = s31 * 0.5;
        tmp5 = s33 - s11;
        tmp2 = sh * sh;
        cond = tmp2 >= gtiny_number;
        sh = select(0.0, sh, cond);
        ch = select(1.0, tmp5, cond);
        tmp1 = sh * sh;
        tmp2 = ch * ch;
        tmp3 = tmp1 + tmp2;
        tmp4 = inverseSqrt(tmp3);
        sh = tmp4 * sh;
        ch = tmp4 * ch;
        cond = (ch * ch) <= (gfour_gamma_squared * (sh * sh));
        sh = select(sh, gsine_pi_over_eight, cond);
        ch = select(ch, gcosine_pi_over_eight, cond);
        tmp1 = sh * sh;
        tmp2 = ch * ch;
        c = tmp2 - tmp1;
        s = 2.0 * ch * sh;

        // Givens conjugation
        tmp3 = tmp1 + tmp2;
        s22 = s22 * tmp3; s32 = s32 * tmp3; s21 = s21 * tmp3; s22 = s22 * tmp3; // Note: s22 updated twice
        tmp1 = s * s32; tmp2 = s * s21;
        s32 = c * s32 + tmp2;
        s21 = c * s21 - tmp1;
        tmp2 = s * s;
        tmp1 = s11 * tmp2; tmp3 = s33 * tmp2; tmp4 = c * c;
        s33 = s33 * tmp4 + tmp1;
        s11 = s11 * tmp4 + tmp3;
        tmp4 = tmp4 - tmp2;
        tmp2 = 2.0 * s31;
        s31 = s31 * tmp4;
        tmp4 = c * s;
        tmp2 = tmp2 * tmp4;
        tmp5 = tmp5 * tmp4;
        s33 = s33 + tmp2;
        s31 = s31 - tmp5;
        s11 = s11 - tmp2;

        // Update quaternion
        tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
        qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
        qvvy = qvvy + sh;
        qvs = qvs - tmp2;
        qvvz = qvvz + tmp1;
        qvvx = qvvx - tmp3;
    }

    //###########################################################
    // 3. Normalize and convert quaternion to matrix V
    //###########################################################
    tmp2 = qvs*qvs + qvvx*qvvx + qvvy*qvvy + qvvz*qvvz;
    // Refine reciprocal sqrt with one Newton-Raphson iteration
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5;
    tmp3 = tmp1 * tmp4;
    tmp3 = tmp1 * tmp3;
    tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;

    qvs = qvs * tmp1; qvvx = qvvx * tmp1; qvvy = qvvy * tmp1; qvvz = qvvz * tmp1;

    // Convert quaternion to matrix V
    var v11: f32; var v12: f32; var v13: f32;
    var v21: f32; var v22: f32; var v23: f32;
    var v31: f32; var v32: f32; var v33: f32;
    
    tmp1 = qvvx * qvvx; tmp2 = qvvy * qvvy; tmp3 = qvvz * qvvz;
    v11 = qvs * qvs;
    v22 = v11 - tmp1 + tmp2 - tmp3;
    v33 = v11 - tmp1 - tmp2 + tmp3;
    v11 = v11 + tmp1 - tmp2 - tmp3;
    tmp1 = 2.0 * qvvx; tmp2 = 2.0 * qvvy; tmp3 = 2.0 * qvvz;
    v32 = qvs * tmp1; v13 = qvs * tmp2; v21 = qvs * tmp3;
    tmp1 = qvvy * tmp1; tmp2 = qvvz * tmp2; tmp3 = qvvx * tmp3;
    v12 = tmp1 - v21; v23 = tmp2 - v32; v31 = tmp3 - v13;
    v21 = tmp1 + v21; v32 = tmp2 + v32; v13 = tmp3 + v13;

    //###########################################################
    // 4. Compute B = A * V
    //###########################################################
    // Row 1
    tmp1 = a11; tmp2 = a12; tmp3 = a13;
    a11 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    a12 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    a13 = tmp1*v13 + tmp2*v23 + tmp3*v33;
    // Row 2
    tmp1 = a21; tmp2 = a22; tmp3 = a23;
    a21 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    a22 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    a23 = tmp1*v13 + tmp2*v23 + tmp3*v33;
    // Row 3
    tmp1 = a31; tmp2 = a32; tmp3 = a33;
    a31 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    a32 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    a33 = tmp1*v13 + tmp2*v23 + tmp3*v33;
    
    //###########################################################
    // 5. Sort columns of B and V to ensure descending singular values
    //###########################################################
    var c1_norm_sq = a11*a11 + a21*a21 + a31*a31;
    var c2_norm_sq = a12*a12 + a22*a22 + a32*a32;
    var c3_norm_sq = a13*a13 + a23*a23 + a33*a33;
    
    // Swap columns 1 and 2
    var swap_cond = c1_norm_sq < c2_norm_sq;
    var swap_res = cond_swap(a11, a12, swap_cond); a11 = swap_res.x; a12 = swap_res.y;
    swap_res = cond_swap(a21, a22, swap_cond); a21 = swap_res.x; a22 = swap_res.y;
    swap_res = cond_swap(a31, a32, swap_cond); a31 = swap_res.x; a32 = swap_res.y;
    swap_res = cond_swap(v11, v12, swap_cond); v11 = swap_res.x; v12 = swap_res.y;
    swap_res = cond_swap(v21, v22, swap_cond); v21 = swap_res.x; v22 = swap_res.y;
    swap_res = cond_swap(v31, v32, swap_cond); v31 = swap_res.x; v32 = swap_res.y;
    swap_res = cond_swap(c1_norm_sq, c2_norm_sq, swap_cond); c1_norm_sq = swap_res.x; c2_norm_sq = swap_res.y;
    var multiplier = select(1.0, -1.0, swap_cond);
    a12 *= multiplier; a22 *= multiplier; a32 *= multiplier;
    v12 *= multiplier; v22 *= multiplier; v32 *= multiplier;

    // Swap columns 1 and 3
    swap_cond = c1_norm_sq < c3_norm_sq;
    swap_res = cond_swap(a11, a13, swap_cond); a11 = swap_res.x; a13 = swap_res.y;
    swap_res = cond_swap(a21, a23, swap_cond); a21 = swap_res.x; a23 = swap_res.y;
    swap_res = cond_swap(a31, a33, swap_cond); a31 = swap_res.x; a33 = swap_res.y;
    swap_res = cond_swap(v11, v13, swap_cond); v11 = swap_res.x; v13 = swap_res.y;
    swap_res = cond_swap(v21, v23, swap_cond); v21 = swap_res.x; v23 = swap_res.y;
    swap_res = cond_swap(v31, v33, swap_cond); v31 = swap_res.x; v33 = swap_res.y;
    swap_res = cond_swap(c1_norm_sq, c3_norm_sq, swap_cond); c1_norm_sq = swap_res.x; c3_norm_sq = swap_res.y;
    multiplier = select(1.0, -1.0, swap_cond);
    a11 *= multiplier; a21 *= multiplier; a31 *= multiplier;
    v11 *= multiplier; v21 *= multiplier; v31 *= multiplier;

    // Swap columns 2 and 3
    swap_cond = c2_norm_sq < c3_norm_sq;
    swap_res = cond_swap(a12, a13, swap_cond); a12 = swap_res.x; a13 = swap_res.y;
    swap_res = cond_swap(a22, a23, swap_cond); a22 = swap_res.x; a23 = swap_res.y;
    swap_res = cond_swap(a32, a33, swap_cond); a32 = swap_res.x; a33 = swap_res.y;
    swap_res = cond_swap(v12, v13, swap_cond); v12 = swap_res.x; v13 = swap_res.y;
    swap_res = cond_swap(v22, v23, swap_cond); v22 = swap_res.x; v23 = swap_res.y;
    swap_res = cond_swap(v32, v33, swap_cond); v32 = swap_res.x; v33 = swap_res.y;
    swap_res = cond_swap(c2_norm_sq, c3_norm_sq, swap_cond); c2_norm_sq = swap_res.x; c3_norm_sq = swap_res.y;
    multiplier = select(1.0, -1.0, swap_cond);
    a13 *= multiplier; a23 *= multiplier; a33 *= multiplier;
    v13 *= multiplier; v23 *= multiplier; v33 *= multiplier;

    //###########################################################
    // 6. QR factorization of B (= U * S) using Givens rotations
    //###########################################################
    var u11 = 1.0; var u12 = 0.0; var u13 = 0.0;
    var u21 = 0.0; var u22 = 1.0; var u23 = 0.0;
    var u31 = 0.0; var __u32 = 0.0; var u33 = 1.0;
    
    // Givens rotation 1 (zeroes a21)
    cond = (a21*a21) >= gsmall_number;
    sh = select(0.0, a21, cond);
    tmp5 = select(1.0, 0.0, a11 >= 0.0);
    ch = max(max(-a11, a11), gsmall_number);
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    // Refine 1/sqrt
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    tmp1 *= tmp2;
    ch += tmp1;
    tmp1 = select(sh, ch, tmp5 == 0.0);
    tmp2 = select(ch, sh, tmp5 == 0.0);
    ch = tmp1; sh = tmp2;
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    // Refine 1/sqrt
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    ch *= tmp1; sh *= tmp1;
    c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    // Rotate A and U
    tmp1 = s*a11; tmp2 = s*a21; a11 = c*a11 + tmp2; a21 = c*a21 - tmp1;
    tmp1 = s*a12; tmp2 = s*a22; a12 = c*a12 + tmp2; a22 = c*a22 - tmp1;
    tmp1 = s*a13; tmp2 = s*a23; a13 = c*a13 + tmp2; a23 = c*a23 - tmp1;
    tmp1 = s*u11; tmp2 = s*u12; u11 = c*u11 + tmp2; u12 = c*u12 - tmp1;
    tmp1 = s*u21; tmp2 = s*u22; u21 = c*u21 + tmp2; u22 = c*u22 - tmp1;
    tmp1 = s*u31; tmp2 = s*__u32; u31 = c*u31 + tmp2; _u32 = c*_u32 - tmp1;

    // Givens rotation 2 (zeroes a31)
    cond = (a31*a31) >= gsmall_number;
    sh = select(0.0, a31, cond);
    tmp5 = select(1.0, 0.0, a11 >= 0.0);
    ch = max(max(-a11, a11), gsmall_number);
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    // Refine 1/sqrt
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    tmp1 *= tmp2;
    ch += tmp1;
    tmp1 = select(sh, ch, tmp5 == 0.0);
    tmp2 = select(ch, sh, tmp5 == 0.0);
    ch = tmp1; sh = tmp2;
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    // Refine 1/sqrt
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    ch *= tmp1; sh *= tmp1;
    c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    // Rotate A and U
    tmp1 = s*a11; tmp2 = s*a31; a11 = c*a11 + tmp2; a31 = c*a31 - tmp1;
    tmp1 = s*a12; tmp2 = s*a32; a12 = c*a12 + tmp2; a32 = c*a32 - tmp1;
    tmp1 = s*a13; tmp2 = s*a33; a13 = c*a13 + tmp2; a33 = c*a33 - tmp1;
    tmp1 = s*u11; tmp2 = s*u13; u11 = c*u11 + tmp2; u13 = c*u13 - tmp1;
    tmp1 = s*u21; tmp2 = s*u23; u21 = c*u21 + tmp2; u23 = c*u23 - tmp1;
    tmp1 = s*u31; tmp2 = s*u33; u31 = c*u31 + tmp2; u33 = c*u33 - tmp1;

    // Givens rotation 3 (zeroes a32)
    cond = (a32*a32) >= gsmall_number;
    sh = select(0.0, a32, cond);
    tmp5 = select(1.0, 0.0, a22 >= 0.0);
    ch = max(max(-a22, a22), gsmall_number);
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    // Refine 1/sqrt
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    tmp1 *= tmp2;
    ch += tmp1;
    tmp1 = select(sh, ch, tmp5 == 0.0);
    tmp2 = select(ch, sh, tmp5 == 0.0);
    ch = tmp1; sh = tmp2;
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    // Refine 1/sqrt
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    ch *= tmp1; sh *= tmp1;
    c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    // Rotate A and U
    tmp1 = s*a21; tmp2 = s*a31; a21 = c*a21 + tmp2; a31 = c*a31 - tmp1;
    tmp1 = s*a22; tmp2 = s*a32; a22 = c*a22 + tmp2; a32 = c*a32 - tmp1;
    tmp1 = s*a23; tmp2 = s*a33; a23 = c*a23 + tmp2; a33 = c*a33 - tmp1;
    tmp1 = s*u12; tmp2 = s*u13; u12 = c*u12 + tmp2; u13 = c*u13 - tmp1;
    tmp1 = s*u22; tmp2 = s*u23; u22 = c*u22 + tmp2; u23 = c*u23 - tmp1;
    tmp1 = s*_u32; tmp2 = s*u33; _u32 = c*_u32 + tmp2; u33 = c*u33 - tmp1;
    
    // Package and return the results
    var result: SVDResult;
    result.U = Mat3x3Row(u11, u12, u13, u21, u22, u23, u31, _u32, u33);
    result.V = Mat3x3Row(v11, v12, v13, v21, v22, v23, v31, v32, v33);
    result.S = vec3(a11, a22, a33);
    
    return result;
}


fn my_sqr(x: f32) -> f32 { return x * x; }

fn vec3_norm2(v: vec3<f32>) -> f32 { return dot(v, v); }

// a * b for row-major 3x3
fn mul_row3(a: Mat3x3Row, b: Mat3x3Row) -> Mat3x3Row {
    var r: Mat3x3Row;
    for (var i: u32 = 0u; i < 3u; i++) {
        for (var j: u32 = 0u; j < 3u; j++) {
            var s: f32 = 0.0;
            for (var k: u32 = 0u; k < 3u; k++) {
                s += a[i*3u + k] * b[k*3u + j];
            }
            r[i*3u + j] = s;
        }
    }
    return r;
}

fn compute_gradient_row_major(
    U: Mat3x3Row, S: vec3<f32>, V: Mat3x3Row, B: Mat3x3Row
) -> Grad12{

}


// ---------- Pure function: one-tet solve (no bindings, no atomics) ----------

struct ARAPInputs {
    x0: vec3<f32>,
    x1: vec3<f32>,
    x2: vec3<f32>,
    x3: vec3<f32>,
    B : Mat3x3Row,         // row-major
    inv_mass: vec4<f32>,   // (m0, m1, m2, m3)
    is_driven: vec4<u32>,  // 0/1 for (p0..p3)
    alpha: f32,
    lagrange: f32,
    dt: f32,
    omega: f32,
}

struct ARAPOutputs {
    dp0: vec3<f32>,
    dp1: vec3<f32>,
    dp2: vec3<f32>,
    dp3: vec3<f32>,
    lagrange_new: f32,
    residual: f32,
    skip: bool, // true -> 小于阈值，调用方可不写回
}

fn arap_solve_one_tet(inp: ARAPInputs) -> ARAPOutputs {
    // // D_s (row-major)
    // var D_s: Mat3x3Row;
    // D_s[0] = inp.x1.x - inp.x0.x;  D_s[1] = inp.x2.x - inp.x0.x;  D_s[2] = inp.x3.x - inp.x0.x;
    // D_s[3] = inp.x1.y - inp.x0.y;  D_s[4] = inp.x2.y - inp.x0.y;  D_s[5] = inp.x3.y - inp.x0.y;
    // D_s[6] = inp.x1.z - inp.x0.z;  D_s[7] = inp.x2.z - inp.x0.z;  D_s[8] = inp.x3.z - inp.x0.z;

    // // F = D_s * B
    // let F = mul_row3(D_s, inp.B);

    // // SVD
    // let svd = svd3x3(F);
    // let S   = svd.S;

    // // constraint = ||S - (1,1,1)||
    // let c = sqrt(sqr(S.x - 1.0) + sqr(S.y - 1.0) + sqr(S.z - 1.0));

    // if (c < EPS) {
    //     return ARAPOutputs(
    //         vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0),
    //         inp.lagrange, 0.0, true
    //     );
    // }

    // // gradient
    // let gradC = compute_gradient_row_major(svd.U, S, svd.V, inp.B);
    // let g0 = vec3<f32>(gradC[0],  gradC[1],  gradC[2]);
    // let g1 = vec3<f32>(gradC[3],  gradC[4],  gradC[5]);
    // let g2 = vec3<f32>(gradC[6],  gradC[7],  gradC[8]);
    // let g3 = vec3<f32>(gradC[9],  gradC[10], gradC[11]);

    // // denominator
    // let denom =
    //       inp.inv_mass.x * vec3_norm2(g0)
    //     + inp.inv_mass.y * vec3_norm2(g1)
    //     + inp.inv_mass.z * vec3_norm2(g2)
    //     + inp.inv_mass.w * vec3_norm2(g3);

    // let alpha_tilde = inp.alpha / (inp.dt * inp.dt);
    // let dlambda = -(c + alpha_tilde * inp.lagrange) / (denom + alpha_tilde);
    // let lambda_new = inp.lagrange + dlambda;

    // // 推荐让调用者根据 is_driven 决定是否写回；这里仍给出完整贡献
    // let dp0 = inp.omega * inp.inv_mass.x * dlambda * g0;
    // let dp1 = inp.omega * inp.inv_mass.y * dlambda * g1;
    // let dp2 = inp.omega * inp.inv_mass.z * dlambda * g2;
    // let dp3 = inp.omega * inp.inv_mass.w * dlambda * g3;

    // let res = abs(c + alpha_tilde * lambda_new);

    // return ARAPOutputs(dp0, dp1, dp2, dp3, lambda_new, res, false);
}

// ---------- Atomic float add via CAS on atomic<u32> ----------
fn atomic_add_f32(p: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old_bits: u32 = atomicLoad(p);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_f = old_f + val;
        let new_bits = bitcast<u32>(new_f);
        let r = atomicCompareExchangeWeak(p, old_bits, new_bits);
        if (r.exchanged) { break; }
        old_bits = r.old_value;
    }
}

// // ---------- Bound compute variant ----------

// // 注意：根据你的 CPU 侧布局，必要时加上 @stride(12) 等。这里假定：
// // - pos_mid 为紧凑 vec3<f32>（12 字节递增）
// // - pos 是 3*NV 长度的 array<atomic<u32>>，每个 f32 以位模式存储
// // - tet_indices: array<vec4<u32>>
// // - B_flat: array<f32>，每 9 个为一矩阵（row-major）
// // - is_driven: array<u32> (0/1)
// // - inv_mass: array<f32>，长度 NV
// // - alpha, lagrange, residuals: array<f32>，长度 NT

// struct Params {
//     dt: f32,
//     omega: f32,
//     _pad0: vec2<f32>,   // 16B 对齐
//     NT: u32,
//     NV: u32,
//     _pad1: vec2<u32>,
// };

@group(0) @binding(0) var<storage, read>       is_driven   : array<u32>;
@group(0) @binding(1) var<storage, read>       tet_indices : array<vec4<u32>>;
@group(0) @binding(2) var<storage, read>       inv_mass    : array<f32>;
@group(0) @binding(3) var<storage, read>       B_flat      : array<f32>;
@group(0) @binding(4) var<storage, read>       pos_mid     : array<vec3<f32>>; // add @stride(12) if packed
@group(0) @binding(5) var<storage, read_write> lagrange    : array<f32>;
@group(0) @binding(6) var<storage, read_write> residuals   : array<f32>;
// pos as bit-pattern float buffer with atomics
@group(0) @binding(7) var<storage, read_write> pos_bits    : array<atomic<u32>>;
// @group(0) @binding(8) var<uniform>             params      : Params;

fn load_mat3_row_from_flat(base: u32) -> Mat3x3Row {
    var M: Mat3x3Row;
    // base points to the first of 9 floats
    for (var i: u32 = 0u; i < 9u; i++) {
        M[i] = B_flat[base + i];
    }
    return M;
}

@compute @workgroup_size(128)
fn cs_arap(@builtin(global_invocation_id) gid: vec3<u32>) {
    // let t = gid.x;
    // if (t >= params.NT) { return; }

    // // indices
    // let ids = tet_indices[t];
    // let p0 = ids.x;
    // let p1 = ids.y;
    // let p2 = ids.z;
    // let p3 = ids.w;

    // // gather positions from pos_mid
    // let x0 = pos_mid[p0];
    // let x1 = pos_mid[p1];
    // let x2 = pos_mid[p2];
    // let x3 = pos_mid[p3];

    // // gather per-vertex inv_mass
    // let im = vec4<f32>(inv_mass[p0], inv_mass[p1], inv_mass[p2], inv_mass[p3]);

    // // driven flags (0/1)
    // let drv = vec4<u32>(is_driven[p0], is_driven[p1], is_driven[p2], is_driven[p3]);

    // // B(t) row-major
    // let B_t = load_mat3_row_from_flat(t * 9u);

    // // scalar fields
    // let alpha_t = /* alpha per tet */ residuals[t]; // <--- 占位：请把 alpha[t] 绑定成单独的缓冲或合并在别处
    // // ^^^ 这里先占位引用，提醒你把 alpha[t] 真正接到一个缓冲。
    // // 如果你已经有 alpha 缓冲，替换为： let alpha_t = alpha[t];

    // let inp = ARAPInputs(
    //     x0, x1, x2, x3,
    //     B_t,
    //     im,
    //     drv,
    //     alpha_t,
    //     lagrange[t],
    //     params.dt,
    //     params.omega
    // );

    // let outp = arap_solve_one_tet(inp);

    // if (!outp.skip) {
    //     // 写回 lagrange/residual（每个 t 唯一，无需原子）
    //     lagrange[t]  = outp.lagrange_new;
    //     residuals[t] = outp.residual;

    //     // 原子累加 pos：pos 是按 (x,y,z) 顺序的 f32 位模式缓冲
    //     // 每个顶点的基址 = p*3
    //     let base0 = p0 * 3u;
    //     let base1 = p1 * 3u;
    //     let base2 = p2 * 3u;
    //     let base3 = p3 * 3u;

    //     if (drv.x == 0u) {
    //         atomic_add_f32(&pos_bits[base0 + 0u], outp.dp0.x);
    //         atomic_add_f32(&pos_bits[base0 + 1u], outp.dp0.y);
    //         atomic_add_f32(&pos_bits[base0 + 2u], outp.dp0.z);
    //     }
    //     if (drv.y == 0u) {
    //         atomic_add_f32(&pos_bits[base1 + 0u], outp.dp1.x);
    //         atomic_add_f32(&pos_bits[base1 + 1u], outp.dp1.y);
    //         atomic_add_f32(&pos_bits[base1 + 2u], outp.dp1.z);
    //     }
    //     if (drv.z == 0u) {
    //         atomic_add_f32(&pos_bits[base2 + 0u], outp.dp2.x);
    //         atomic_add_f32(&pos_bits[base2 + 1u], outp.dp2.y);
    //         atomic_add_f32(&pos_bits[base2 + 2u], outp.dp2.z);
    //     }
    //     if (drv.w == 0u) {
    //         atomic_add_f32(&pos_bits[base3 + 0u], outp.dp3.x);
    //         atomic_add_f32(&pos_bits[base3 + 1u], outp.dp3.y);
    //         atomic_add_f32(&pos_bits[base3 + 2u], outp.dp3.z);
    //     }
    // } else {
    //     // 小于阈值：只清 residual，保持 lagrange 不变（与 CUDA 一致）
    //     residuals[t] = 0.0;
    // }
}




// Bindings for the input and output storage buffers.
@group(3) @binding(0) var<storage, read> input_matrices: array<Mat3x3Row>;
@group(3) @binding(1) var<storage, read_write> output_results: array<SVDResult>;


// The entry point for the compute shader.
// Each invocation handles one matrix from the input array.
@compute @workgroup_size(64, 1, 1)
fn svd_test(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let num_matrices = arrayLength(&input_matrices);

    if (idx >= num_matrices) {
        return;
    }

    // Unpack input matrix A
    let A = input_matrices[idx];
    let a11 = A.r1.x; let a12 = A.r1.y; let a13 = A.r1.z;
    let a21 = A.r2.x; let a22 = A.r2.y; let a23 = A.r2.z;
    let a31 = A.r3.x; let a32 = A.r3.y; let a33 = A.r3.z;

    // Perform the SVD calculation
    let result = svd3x3(
        a11, a12, a13, 
        a21, a22, a23, 
        a31, a32, a33
    );
    
    // Store the result
    output_results[idx] = result;
}