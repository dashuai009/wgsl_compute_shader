// ============================================================================
// WGSL Compute Shader: Signed-SVD ARAP + Hydrostatic Volume XPBD Solver
// ============================================================================

const EPS: f32 = 1e-6;
alias Mat3x3Row = array<f32, 9>;

struct Grad4 {
    g0: vec3<f32>,
    g1: vec3<f32>,
    g2: vec3<f32>,
    g3: vec3<f32>,
}

// =================================== 3x3 Signed SVD =========================
// Ported from McAdams et al. (2011) `svd3_cuda.h`.
// Guarantees det(U) = +1, det(V) = +1, and sign(S.z) = sign(det(A)),
// enabling inversion-safe finite-element / XPBD strain recovery.

struct SVDResult {
    U: Mat3x3Row,
    S: vec3<f32>,
    V: Mat3x3Row,
}

fn cond_swap(a: f32, b: f32, cond: bool) -> vec2<f32> {
    return vec2<f32>(select(a, b, cond), select(b, a, cond));
}

fn sqr(x: f32) -> f32 {
    return x * x;
}

fn svd3x3(input_A: Mat3x3Row) -> SVDResult {
    var gsine_pi_over_eight : f32 = bitcast<f32>(1053028117u);   // 0.38268346
    var gcosine_pi_over_eight : f32 = bitcast<f32>(1064076127u); // 0.9238795
    var gsmall_number : f32 = 1.0e-12;
    var gtiny_number : f32 = 1.0e-20;
    var gfour_gamma_squared : f32 = 5.8284273;

    var a11 = input_A[0]; var a12 = input_A[1]; var a13 = input_A[2];
    var a21 = input_A[3]; var a22 = input_A[4]; var a23 = input_A[5];
    var a31 = input_A[6]; var a32 = input_A[7]; var a33 = input_A[8];

    var tmp1: f32; var tmp2: f32; var tmp3: f32; var tmp4: f32; var tmp5: f32;
    var c: f32; var s: f32; var ch: f32; var sh: f32;

    var s11: f32; var s21: f32; var s31: f32;
    var s22: f32; var s32: f32; var s33: f32;

    var qvs: f32; var qvvx: f32; var qvvy: f32; var qvvz: f32;

    // 1. Normal equations S = A^T * A
    s11 = a11*a11 + a21*a21 + a31*a31;
    s21 = a12*a11 + a22*a21 + a32*a31;
    s31 = a13*a11 + a23*a21 + a33*a31;
    s22 = a12*a12 + a22*a22 + a32*a32;
    s32 = a13*a12 + a23*a22 + a33*a32;
    s33 = a13*a13 + a23*a23 + a33*a33;

    qvs = 1.0; qvvx = 0.0; qvvy = 0.0; qvvz = 0.0;

    // 2. 4 cyclic Jacobi sweeps
    for (var i = 0; i < 4; i = i + 1) {
        // (1, 2)
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

        tmp3 = tmp1 + tmp2;
        s33 = s33 * tmp3; s31 = s31 * tmp3; s32 = s32 * tmp3; s33 = s33 * tmp3;
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

        tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
        qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
        qvvz = qvvz + sh;
        qvs = qvs - tmp3;
        qvvx = qvvx + tmp2;
        qvvy = qvvy - tmp1;

        // (2, 3)
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

        tmp3 = tmp1 + tmp2;
        s11 = s11 * tmp3; s21 = s21 * tmp3; s31 = s31 * tmp3; s11 = s11 * tmp3;
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

        tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
        qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
        qvvx = qvvx + sh;
        qvs = qvs - tmp1;
        qvvy = qvvy + tmp3;
        qvvz = qvvz - tmp2;

        // (1, 3)
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

        tmp3 = tmp1 + tmp2;
        s22 = s22 * tmp3; s32 = s32 * tmp3; s21 = s21 * tmp3; s22 = s22 * tmp3;
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

        tmp1 = sh * qvvx; tmp2 = sh * qvvy; tmp3 = sh * qvvz; sh = sh * qvs;
        qvs = ch * qvs; qvvx = ch * qvvx; qvvy = ch * qvvy; qvvz = ch * qvvz;
        qvvy = qvvy + sh;
        qvs = qvs - tmp2;
        qvvz = qvvz + tmp1;
        qvvx = qvvx - tmp3;
    }

    // 3. Normalize quaternion -> V
    tmp2 = qvs*qvs + qvvx*qvvx + qvvy*qvvy + qvvz*qvvz;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5;
    tmp3 = tmp1 * tmp4;
    tmp3 = tmp1 * tmp3;
    tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;

    qvs = qvs * tmp1; qvvx = qvvx * tmp1; qvvy = qvvy * tmp1; qvvz = qvvz * tmp1;

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

    // 4. B = A * V
    tmp1 = a11; tmp2 = a12; tmp3 = a13;
    a11 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    a12 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    a13 = tmp1*v13 + tmp2*v23 + tmp3*v33;

    tmp1 = a21; tmp2 = a22; tmp3 = a23;
    a21 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    a22 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    a23 = tmp1*v13 + tmp2*v23 + tmp3*v33;

    tmp1 = a31; tmp2 = a32; tmp3 = a33;
    a31 = tmp1*v11 + tmp2*v21 + tmp3*v31;
    a32 = tmp1*v12 + tmp2*v22 + tmp3*v32;
    a33 = tmp1*v13 + tmp2*v23 + tmp3*v33;

    // 5. Sort columns of B and V preserving det(V) = +1
    var c1_norm_sq = a11*a11 + a21*a21 + a31*a31;
    var c2_norm_sq = a12*a12 + a22*a22 + a32*a32;
    var c3_norm_sq = a13*a13 + a23*a23 + a33*a33;

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

    // 6. Givens QR factorization B = U * S preserving det(U) = +1
    var u11 = 1.0; var u12 = 0.0; var u13 = 0.0;
    var u21 = 0.0; var u22 = 1.0; var u23 = 0.0;
    var u31 = 0.0; var _u32 = 0.0; var u33 = 1.0;

    var cond = (a21*a21) >= gsmall_number;
    sh = select(0.0, a21, cond);
    tmp5 = select(1.0, 0.0, a11 >= 0.0);
    ch = max(max(-a11, a11), gsmall_number);
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    tmp1 *= tmp2;
    ch += tmp1;
    tmp1 = select(sh, ch, tmp5 == 0.0);
    tmp2 = select(ch, sh, tmp5 == 0.0);
    ch = tmp1; sh = tmp2;
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    ch *= tmp1; sh *= tmp1;
    c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    tmp1 = s*a11; tmp2 = s*a21; a11 = c*a11 + tmp2; a21 = c*a21 - tmp1;
    tmp1 = s*a12; tmp2 = s*a22; a12 = c*a12 + tmp2; a22 = c*a22 - tmp1;
    tmp1 = s*a13; tmp2 = s*a23; a13 = c*a13 + tmp2; a23 = c*a23 - tmp1;
    tmp1 = s*u11; tmp2 = s*u12; u11 = c*u11 + tmp2; u12 = c*u12 - tmp1;
    tmp1 = s*u21; tmp2 = s*u22; u21 = c*u21 + tmp2; u22 = c*u22 - tmp1;
    tmp1 = s*u31; tmp2 = s*_u32; u31 = c*u31 + tmp2; _u32 = c*_u32 - tmp1;

    cond = (a31*a31) >= gsmall_number;
    sh = select(0.0, a31, cond);
    tmp5 = select(1.0, 0.0, a11 >= 0.0);
    ch = max(max(-a11, a11), gsmall_number);
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    tmp1 *= tmp2;
    ch += tmp1;
    tmp1 = select(sh, ch, tmp5 == 0.0);
    tmp2 = select(ch, sh, tmp5 == 0.0);
    ch = tmp1; sh = tmp2;
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    ch *= tmp1; sh *= tmp1;
    c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    tmp1 = s*a11; tmp2 = s*a31; a11 = c*a11 + tmp2; a31 = c*a31 - tmp1;
    tmp1 = s*a12; tmp2 = s*a32; a12 = c*a12 + tmp2; a32 = c*a32 - tmp1;
    tmp1 = s*a13; tmp2 = s*a33; a13 = c*a13 + tmp2; a33 = c*a33 - tmp1;
    tmp1 = s*u11; tmp2 = s*u13; u11 = c*u11 + tmp2; u13 = c*u13 - tmp1;
    tmp1 = s*u21; tmp2 = s*u23; u21 = c*u21 + tmp2; u23 = c*u23 - tmp1;
    tmp1 = s*u31; tmp2 = s*u33; u31 = c*u31 + tmp2; u33 = c*u33 - tmp1;

    cond = (a32*a32) >= gsmall_number;
    sh = select(0.0, a32, cond);
    tmp5 = select(1.0, 0.0, a22 >= 0.0);
    ch = max(max(-a22, a22), gsmall_number);
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    tmp1 *= tmp2;
    ch += tmp1;
    tmp1 = select(sh, ch, tmp5 == 0.0);
    tmp2 = select(ch, sh, tmp5 == 0.0);
    ch = tmp1; sh = tmp2;
    tmp2 = sh*sh + ch*ch;
    tmp1 = inverseSqrt(tmp2);
    tmp4 = tmp1 * 0.5; tmp3 = tmp1 * tmp4; tmp3 = tmp1 * tmp3; tmp3 = tmp2 * tmp3;
    tmp1 = tmp1 + tmp4 - tmp3;
    ch *= tmp1; sh *= tmp1;
    c = ch*ch - sh*sh; s = 2.0 * sh * ch;

    tmp1 = s*a21; tmp2 = s*a31; a21 = c*a21 + tmp2; a31 = c*a31 - tmp1;
    tmp1 = s*a22; tmp2 = s*a32; a22 = c*a22 + tmp2; a32 = c*a32 - tmp1;
    tmp1 = s*a23; tmp2 = s*a33; a23 = c*a23 + tmp2; a33 = c*a33 - tmp1;
    tmp1 = s*u12; tmp2 = s*u13; u12 = c*u12 + tmp2; u13 = c*u13 - tmp1;
    tmp1 = s*u22; tmp2 = s*u23; u22 = c*u22 + tmp2; u23 = c*u23 - tmp1;
    tmp1 = s*_u32; tmp2 = s*u33; _u32 = c*_u32 + tmp2; u33 = c*u33 - tmp1;

    var result: SVDResult;
    result.U[0] = u11; result.U[1] = u12; result.U[2] = u13;
    result.U[3] = u21; result.U[4] = u22; result.U[5] = u23;
    result.U[6] = u31; result.U[7] = _u32; result.U[8] = u33;

    result.V[0] = v11; result.V[1] = v12; result.V[2] = v13;
    result.V[3] = v21; result.V[4] = v22; result.V[5] = v23;
    result.V[6] = v31; result.V[7] = v32; result.V[8] = v33;

    result.S.x = a11;
    result.S.y = a22;
    result.S.z = a33;
    return result;
}

// =================================== ARAP Helpers ===========================

fn vec3_norm2(v: vec3<f32>) -> f32 {
    return dot(v, v);
}

fn mul_row3(a: Mat3x3Row, b: Mat3x3Row) -> Mat3x3Row {
    var r: Mat3x3Row;
    for (var i: u32 = 0u; i < 3u; i++) {
        for (var j: u32 = 0u; j < 3u; j++) {
            var s: f32 = 0.0;
            for (var k: u32 = 0u; k < 3u; k++) {
                s += a[i * 3u + k] * b[k * 3u + j];
            }
            r[i * 3u + j] = s;
        }
    }
    return r;
}

fn compute_gradient_row_major(
    U: Mat3x3Row,
    S: vec3<f32>,
    V: Mat3x3Row,
    B: Mat3x3Row
) -> Grad4 {
    let ones = vec3<f32>(1.0, 1.0, 1.0);
    let diff = S - ones;
    let constraint: f32 = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

    if (constraint < EPS) {
        return Grad4(
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0),
            vec3<f32>(0.0)
        );
    }

    let dcdS: vec3<f32> = diff / constraint;

    let B00 = B[0u]; let B01 = B[1u]; let B02 = B[2u];
    let B10 = B[3u]; let B11 = B[4u]; let B12 = B[5u];
    let B20 = B[6u]; let B21 = B[7u]; let B22 = B[8u];

    let dsdF00 = vec3<f32>(U[0u] * V[0u], U[1u] * V[1u], U[2u] * V[2u]);
    let dsdF10 = vec3<f32>(U[3u] * V[0u], U[4u] * V[1u], U[5u] * V[2u]);
    let dsdF20 = vec3<f32>(U[6u] * V[0u], U[7u] * V[1u], U[8u] * V[2u]);

    let dsdF01 = vec3<f32>(U[0u] * V[3u], U[1u] * V[4u], U[2u] * V[5u]);
    let dsdF11 = vec3<f32>(U[3u] * V[3u], U[4u] * V[4u], U[5u] * V[5u]);
    let dsdF21 = vec3<f32>(U[6u] * V[3u], U[7u] * V[4u], U[8u] * V[5u]);

    let dsdF02 = vec3<f32>(U[0u] * V[6u], U[1u] * V[7u], U[2u] * V[8u]);
    let dsdF12 = vec3<f32>(U[3u] * V[6u], U[4u] * V[7u], U[5u] * V[8u]);
    let dsdF22 = vec3<f32>(U[6u] * V[6u], U[7u] * V[7u], U[8u] * V[8u]);

    var dcdF: array<f32, 9>;
    dcdF[0u] = dot(dsdF00, dcdS);
    dcdF[1u] = dot(dsdF10, dcdS);
    dcdF[2u] = dot(dsdF20, dcdS);
    dcdF[3u] = dot(dsdF01, dcdS);
    dcdF[4u] = dot(dsdF11, dcdS);
    dcdF[5u] = dot(dsdF21, dcdS);
    dcdF[6u] = dot(dsdF02, dcdS);
    dcdF[7u] = dot(dsdF12, dcdS);
    dcdF[8u] = dot(dsdF22, dcdS);

    let g1 = vec3<f32>(
        B00 * dcdF[0u] + B01 * dcdF[3u] + B02 * dcdF[6u],
        B00 * dcdF[1u] + B01 * dcdF[4u] + B02 * dcdF[7u],
        B00 * dcdF[2u] + B01 * dcdF[5u] + B02 * dcdF[8u]
    );
    let g2 = vec3<f32>(
        B10 * dcdF[0u] + B11 * dcdF[3u] + B12 * dcdF[6u],
        B10 * dcdF[1u] + B11 * dcdF[4u] + B12 * dcdF[7u],
        B10 * dcdF[2u] + B11 * dcdF[5u] + B12 * dcdF[8u]
    );
    let g3 = vec3<f32>(
        B20 * dcdF[0u] + B21 * dcdF[3u] + B22 * dcdF[6u],
        B20 * dcdF[1u] + B21 * dcdF[4u] + B22 * dcdF[7u],
        B20 * dcdF[2u] + B21 * dcdF[5u] + B22 * dcdF[8u]
    );
    let g0 = -(g1 + g2 + g3);

    return Grad4(g0, g1, g2, g3);
}

// =================================== Bindings ===============================
// Packed into 6 storage buffers + 1 uniform buffer (fits WebGPU's 8-SSBO limit).

struct VertData {
    pos_mid_im: vec4<f32>, // xyz = Jacobi snapshot pos_mid, w = inv_mass (0 if pinned)
    pos_prev:   vec4<f32>, // xyz = substep start position
    vel:        vec4<f32>, // xyz = linear velocity
}

struct TetStatic {
    indices: vec4<u32>, // (i0, i1, i2, i3)
    b_row0:  vec4<f32>, // (B00, B01, B02, rest_vol)
    b_row1:  vec4<f32>, // (B10, B11, B12, tet_omega)
    b_row2:  vec4<f32>, // (B20, B21, B22, 0.0)
}

struct SimParams {
    dt:           f32,
    omega:        f32,
    mu:           f32,
    lambda:       f32,
    gravity_y:    f32,
    damping:      f32,
    floor_y:      f32,
    friction:     f32,
    nt:           u32,
    nv:           u32,
    n_surf_tris:  u32,
    enable_floor: u32,
}

@group(0) @binding(0) var<storage, read_write> pos          : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> normals      : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> verts        : array<VertData>;
@group(0) @binding(3) var<storage, read>       tets         : array<TetStatic>;
@group(0) @binding(4) var<storage, read_write> tet_state    : array<vec4<f32>>; // x=lambda_arap, y=lambda_vol, z=residual
@group(0) @binding(5) var<storage, read>       surface_tris : array<u32>;
@group(0) @binding(6) var<uniform>             params       : SimParams;

fn load_pos(v: u32) -> vec3<f32> {
    let base = 3u * v;
    return vec3<f32>(
        bitcast<f32>(atomicLoad(&pos[base + 0u])),
        bitcast<f32>(atomicLoad(&pos[base + 1u])),
        bitcast<f32>(atomicLoad(&pos[base + 2u]))
    );
}

fn store_pos(v: u32, p: vec3<f32>) {
    let base = 3u * v;
    atomicStore(&pos[base + 0u], bitcast<u32>(p.x));
    atomicStore(&pos[base + 1u], bitcast<u32>(p.y));
    atomicStore(&pos[base + 2u], bitcast<u32>(p.z));
}

fn atomic_add_pos_f32(idx: u32, val: f32) {
    if (abs(val) < 1e-20) {
        return;
    }
    var old_bits: u32 = atomicLoad(&pos[idx]);
    loop {
        let new_f = bitcast<f32>(old_bits) + val;
        let new_bits = bitcast<u32>(new_f);
        let r = atomicCompareExchangeWeak(&pos[idx], old_bits, new_bits);
        if (r.exchanged) {
            break;
        }
        old_bits = r.old_value;
    }
}

fn atomic_add_normal_f32(idx: u32, val: f32) {
    if (abs(val) < 1e-20) {
        return;
    }
    var old_bits: u32 = atomicLoad(&normals[idx]);
    loop {
        let new_f = bitcast<f32>(old_bits) + val;
        let new_bits = bitcast<u32>(new_f);
        let r = atomicCompareExchangeWeak(&normals[idx], old_bits, new_bits);
        if (r.exchanged) {
            break;
        }
        old_bits = r.old_value;
    }
}

// =================================== XPBD Kernels ===========================

// 1. Substep start: Semi-implicit Euler prediction with gravity & air damping
@compute @workgroup_size(64, 1, 1)
fn predict_positions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.nv) {
        return;
    }

    let x = load_pos(v);
    let im = verts[v].pos_mid_im.w;
    verts[v].pos_prev = vec4<f32>(x, 0.0);

    if (im <= 0.0) {
        verts[v].vel = vec4<f32>(0.0);
        verts[v].pos_mid_im = vec4<f32>(x, 0.0);
        return;
    }

    let damp = max(0.0, 1.0 - params.damping * params.dt);
    let v_pred = (verts[v].vel.xyz + vec3<f32>(0.0, params.gravity_y, 0.0) * params.dt) * damp;
    var x_pred = x + v_pred * params.dt;

    if (params.enable_floor != 0u && x_pred.y < params.floor_y) {
        x_pred.y = params.floor_y;
    }

    verts[v].vel = vec4<f32>(v_pred, 0.0);
    verts[v].pos_mid_im = vec4<f32>(x_pred, im);
    store_pos(v, x_pred);
}

// 2. Substep start: Reset XPBD Lagrange multipliers ONCE per substep
@compute @workgroup_size(64, 1, 1)
fn reset_lagrange(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.nt) {
        return;
    }
    tet_state[t] = vec4<f32>(0.0);
}

// 3. Before each Jacobi iteration: Clamp floor & snapshot `pos` -> `pos_mid`
@compute @workgroup_size(64, 1, 1)
fn copy_mid_pos(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.nv) {
        return;
    }
    var p = load_pos(v);
    let im = verts[v].pos_mid_im.w;
    if (params.enable_floor != 0u && im > 0.0 && p.y < params.floor_y) {
        p.y = params.floor_y;
        store_pos(v, p);
    }
    verts[v].pos_mid_im = vec4<f32>(p, im);
}

// 4. Race-free Jacobi XPBD Tetrahedron Constraint Projection (Signed-SVD ARAP + Volume)
@compute @workgroup_size(64, 1, 1)
fn cs_arap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= params.nt) {
        return;
    }

    let tet = tets[t];
    let p0 = tet.indices.x;
    let p1 = tet.indices.y;
    let p2 = tet.indices.z;
    let p3 = tet.indices.w;

    let v0 = verts[p0].pos_mid_im;
    let v1 = verts[p1].pos_mid_im;
    let v2 = verts[p2].pos_mid_im;
    let v3 = verts[p3].pos_mid_im;

    let x0 = v0.xyz;
    let x1 = v1.xyz;
    let x2 = v2.xyz;
    let x3 = v3.xyz;
    let im = vec4<f32>(v0.w, v1.w, v2.w, v3.w);

    let rest_vol = tet.b_row0.w;
    let omega_eff = clamp(tet.b_row1.w * params.omega, 0.0, 1.0);
    let inv_dt2 = 1.0 / max(params.dt * params.dt, 1e-16);

    var B: Mat3x3Row;
    B[0] = tet.b_row0.x; B[1] = tet.b_row0.y; B[2] = tet.b_row0.z;
    B[3] = tet.b_row1.x; B[4] = tet.b_row1.y; B[5] = tet.b_row1.z;
    B[6] = tet.b_row2.x; B[7] = tet.b_row2.y; B[8] = tet.b_row2.z;

    let state = tet_state[t];
    var lambda_arap = state.x;
    var lambda_vol  = state.y;

    // ---- (A) Signed-SVD ARAP Deviatoric/Isotropic Strain Constraint ----
    var D_s: Mat3x3Row;
    D_s[0] = x1.x - x0.x; D_s[1] = x2.x - x0.x; D_s[2] = x3.x - x0.x;
    D_s[3] = x1.y - x0.y; D_s[4] = x2.y - x0.y; D_s[5] = x3.y - x0.y;
    D_s[6] = x1.z - x0.z; D_s[7] = x2.z - x0.z; D_s[8] = x3.z - x0.z;

    let F = mul_row3(D_s, B);
    let svd = svd3x3(F);
    let S = svd.S;

    let c_arap = sqrt(sqr(S.x - 1.0) + sqr(S.y - 1.0) + sqr(S.z - 1.0));
    var dp0 = vec3<f32>(0.0);
    var dp1 = vec3<f32>(0.0);
    var dp2 = vec3<f32>(0.0);
    var dp3 = vec3<f32>(0.0);
    var residual = 0.0;

    if (c_arap >= EPS && params.mu > 0.0) {
        let grad = compute_gradient_row_major(svd.U, S, svd.V, B);
        let g0 = grad.g0;
        let g1 = grad.g1;
        let g2 = grad.g2;
        let g3 = grad.g3;

        let denom = im.x * vec3_norm2(g0)
                  + im.y * vec3_norm2(g1)
                  + im.z * vec3_norm2(g2)
                  + im.w * vec3_norm2(g3);
        let alpha_tilde = inv_dt2 / (params.mu * rest_vol);
        let sys_denom = denom + alpha_tilde;
        if (sys_denom > 1e-20) {
            // Apply relaxation `omega_eff` consistently to BOTH delta_lambda and dp
            // so the converged XPBD compliance equals the exact physical `alpha_tilde`.
            let dlambda = -omega_eff * (c_arap + alpha_tilde * lambda_arap) / sys_denom;
            lambda_arap += dlambda;
            dp0 += im.x * dlambda * g0;
            dp1 += im.y * dlambda * g1;
            dp2 += im.z * dlambda * g2;
            dp3 += im.w * dlambda * g3;
            residual = abs(c_arap + alpha_tilde * lambda_arap);
        }
    }

    // ---- (B) Hydrostatic Volume Preservation Constraint C_vol = det(F) - 1 ----
    if (params.lambda > 0.0) {
        let y0 = x0 + dp0;
        let y1 = x1 + dp1;
        let y2 = x2 + dp2;
        let y3 = x3 + dp3;

        let e1 = y1 - y0;
        let e2 = y2 - y0;
        let e3 = y3 - y0;
        let inv_6v0 = 1.0 / (6.0 * rest_vol);

        let g1_v = cross(e2, e3) * inv_6v0;
        let g2_v = cross(e3, e1) * inv_6v0;
        let g3_v = cross(e1, e2) * inv_6v0;
        let g0_v = -(g1_v + g2_v + g3_v);

        let c_vol = dot(e1, cross(e2, e3)) * inv_6v0 - 1.0;
        let denom_v = im.x * vec3_norm2(g0_v)
                    + im.y * vec3_norm2(g1_v)
                    + im.z * vec3_norm2(g2_v)
                    + im.w * vec3_norm2(g3_v);
        let alpha_tilde_v = inv_dt2 / (params.lambda * rest_vol);
        let sys_denom_v = denom_v + alpha_tilde_v;
        if (sys_denom_v > 1e-20 && abs(c_vol + alpha_tilde_v * lambda_vol) >= 1e-7) {
            let dlambda_v = -omega_eff * (c_vol + alpha_tilde_v * lambda_vol) / sys_denom_v;
            lambda_vol += dlambda_v;
            dp0 += im.x * dlambda_v * g0_v;
            dp1 += im.y * dlambda_v * g1_v;
            dp2 += im.z * dlambda_v * g2_v;
            dp3 += im.w * dlambda_v * g3_v;
            residual = max(residual, abs(c_vol + alpha_tilde_v * lambda_vol));
        }
    }

    tet_state[t] = vec4<f32>(lambda_arap, lambda_vol, residual, 0.0);

    if (im.x > 0.0) {
        let b0 = 3u * p0;
        atomic_add_pos_f32(b0 + 0u, dp0.x);
        atomic_add_pos_f32(b0 + 1u, dp0.y);
        atomic_add_pos_f32(b0 + 2u, dp0.z);
    }
    if (im.y > 0.0) {
        let b1 = 3u * p1;
        atomic_add_pos_f32(b1 + 0u, dp1.x);
        atomic_add_pos_f32(b1 + 1u, dp1.y);
        atomic_add_pos_f32(b1 + 2u, dp1.z);
    }
    if (im.z > 0.0) {
        let b2 = 3u * p2;
        atomic_add_pos_f32(b2 + 0u, dp2.x);
        atomic_add_pos_f32(b2 + 1u, dp2.y);
        atomic_add_pos_f32(b2 + 2u, dp2.z);
    }
    if (im.w > 0.0) {
        let b3 = 3u * p3;
        atomic_add_pos_f32(b3 + 0u, dp3.x);
        atomic_add_pos_f32(b3 + 1u, dp3.y);
        atomic_add_pos_f32(b3 + 2u, dp3.z);
    }
}

// 5. Substep end: XPBD velocity update & ground contact friction
@compute @workgroup_size(64, 1, 1)
fn update_velocities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.nv) {
        return;
    }
    let im = verts[v].pos_mid_im.w;
    if (im <= 0.0) {
        verts[v].vel = vec4<f32>(0.0);
        return;
    }

    var x = load_pos(v);
    var in_contact = false;
    if (params.enable_floor != 0u && x.y <= params.floor_y + 1e-5) {
        if (x.y < params.floor_y) {
            x.y = params.floor_y;
            store_pos(v, x);
        }
        in_contact = true;
    }

    let x_prev = verts[v].pos_prev.xyz;
    var vel = (x - x_prev) / max(params.dt, 1e-8);

    if (in_contact) {
        let slip_damp = max(0.0, 1.0 - params.friction * 60.0 * params.dt);
        vel.x *= slip_damp;
        vel.z *= slip_damp;
        vel.y = max(vel.y, 0.0);
    }

    verts[v].vel = vec4<f32>(vel, 0.0);
}

// =================================== Dynamic Surface Normals ================

@compute @workgroup_size(64, 1, 1)
fn clear_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.nv) {
        return;
    }
    let base = 3u * v;
    atomicStore(&normals[base + 0u], 0u);
    atomicStore(&normals[base + 1u], 0u);
    atomicStore(&normals[base + 2u], 0u);
}

@compute @workgroup_size(64, 1, 1)
fn accumulate_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri = gid.x;
    if (tri >= params.n_surf_tris) {
        return;
    }
    let ia = surface_tris[3u * tri + 0u];
    let ib = surface_tris[3u * tri + 1u];
    let ic = surface_tris[3u * tri + 2u];

    let pa = load_pos(ia);
    let pb = load_pos(ib);
    let pc = load_pos(ic);
    let fn_area = cross(pb - pa, pc - pa);

    for (var k: u32 = 0u; k < 3u; k++) {
        let vid = surface_tris[3u * tri + k];
        let base = 3u * vid;
        atomic_add_normal_f32(base + 0u, fn_area.x);
        atomic_add_normal_f32(base + 1u, fn_area.y);
        atomic_add_normal_f32(base + 2u, fn_area.z);
    }
}

@compute @workgroup_size(64, 1, 1)
fn normalize_normals(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.nv) {
        return;
    }
    let base = 3u * v;
    let n = vec3<f32>(
        bitcast<f32>(atomicLoad(&normals[base + 0u])),
        bitcast<f32>(atomicLoad(&normals[base + 1u])),
        bitcast<f32>(atomicLoad(&normals[base + 2u]))
    );
    let len_sq = dot(n, n);
    let unit = select(vec3<f32>(0.0, 1.0, 0.0), n * inverseSqrt(max(len_sq, 1e-24)), len_sq > 1e-20);
    atomicStore(&normals[base + 0u], bitcast<u32>(unit.x));
    atomicStore(&normals[base + 1u], bitcast<u32>(unit.y));
    atomicStore(&normals[base + 2u], bitcast<u32>(unit.z));
}

// =================================== Unit Test Entry: 3x3 SVD ===============

@group(3) @binding(0) var<storage, read>       input_matrices : array<Mat3x3Row>;
@group(3) @binding(1) var<storage, read_write> output_results : array<SVDResult>;

@compute @workgroup_size(64, 1, 1)
fn svd_test(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_matrices)) {
        return;
    }
    output_results[idx] = svd3x3(input_matrices[idx]);
}
