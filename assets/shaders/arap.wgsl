struct Params {
    dt: f32,
    omega: f32,
};
@group(0) @binding(0) var<uniform> params : Params;

@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_mid : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tet_indices : array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> inv_mass : array<f32>;
@group(0) @binding(5) var<storage, read_write> lagrangian : array<f32>;
@group(0) @binding(6) var<storage, read> B_mats : array<mat3x3<f32>>;
@group(0) @binding(7) var<storage, read> alpha : array<f32>;
@group(0) @binding(8) var<storage, read_write> constraint : array<f32>;
@group(0) @binding(9) var<storage, read_write> residual : array<f32>;

// ---- Helper math ---------------------------------------------------------
fn svd3x3(F : mat3x3<f32>) -> struct {
    U: mat3x3<f32>,
    S: vec3<f32>,
    V: mat3x3<f32>,
} {
    // *** NOTE ***
    // WGSL has no native SVD; here we provide a placeholder based on
    // analytic polar decomposition + eigen decomposition approximation.
    // For production, replace with a robust Newton‑based SVD implementation
    // (see McAdams et al. 2011, "SVD with CUDA"), or precompute on CPU.
    var U = mat3x3<f32>(
        vec3<f32>(1.0,0.0,0.0),
        vec3<f32>(0.0,1.0,0.0),
        vec3<f32>(0.0,0.0,1.0));
    var V = U;
    var S = vec3<f32>(1.0,1.0,1.0);
    return .{U, S, V};
}

fn compute_gradient(
        U: mat3x3<f32>,
        S: vec3<f32>,
        V: mat3x3<f32>,
        B: mat3x3<f32>) -> array<vec3<f32>, 4> {
    // Placeholder gradient – must be replaced with the analytic formula that
    // matches the Taichi compute_gradient() implementation.
    var g : array<vec3<f32>, 4>;
    g[0] = vec3<f32>(0.0);
    g[1] = vec3<f32>(0.0);
    g[2] = vec3<f32>(0.0);
    g[3] = vec3<f32>(0.0);
    return g;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    // One thread per tetrahedron
    let t = gid.x;
    if (t >= arrayLength(&tet_indices)) { return; }

    // Copy pos -> pos_mid for each vertex touched by this tet.
    let inds = tet_indices[t];
    pos_mid[inds.x] = pos[inds.x];
    pos_mid[inds.y] = pos[inds.y];
    pos_mid[inds.z] = pos[inds.z];
    pos_mid[inds.w] = pos[inds.w];

    // Load positions as vec3
    let x0 = pos_mid[inds.x].xyz;
    let x1 = pos_mid[inds.y].xyz;
    let x2 = pos_mid[inds.z].xyz;
    let x3 = pos_mid[inds.w].xyz;

    // Construct Ds and deformation gradient F
    let Ds = mat3x3<f32>(x1 - x0, x2 - x0, x3 - x0);
    let F  = Ds * B_mats[t];

    // SVD: F = U * diag(S) * V^T
    let svd = svd3x3(F);
    let U = svd.U;
    let S = svd.S;
    let V = svd.V;

    var c = sqrt((S.x - 1.0)*(S.x - 1.0) + (S.y - 1.0)*(S.y - 1.0) + (S.z - 1.0)*(S.z - 1.0));
    constraint[t] = c;
    if (c < 1e-6) {
        return;
    }

    let g = compute_gradient(U,S,V,B_mats[t]);
    let g0 = g[0];
    let g1 = g[1];
    let g2 = g[2];
    let g3 = g[3];

    let den = inv_mass[inds.x] * dot(g0, g0) +
              inv_mass[inds.y] * dot(g1, g1) +
              inv_mass[inds.z] * dot(g2, g2) +
              inv_mass[inds.w] * dot(g3, g3);

    let alpha_tilde = alpha[t] / (params.dt * params.dt);
    let dlambda = -(c + alpha_tilde * lagrangian[t]) / (den + alpha_tilde);

    lagrangian[t] = lagrangian[t] + dlambda;

    pos[inds.x].xyz = pos[inds.x].xyz + params.omega * inv_mass[inds.x] * dlambda * g0;
    pos[inds.y].xyz = pos[inds.y].xyz + params.omega * inv_mass[inds.y] * dlambda * g1;
    pos[inds.z].xyz = pos[inds.z].xyz + params.omega * inv_mass[inds.z] * dlambda * g2;
    pos[inds.w].xyz = pos[inds.w].xyz + params.omega * inv_mass[inds.w] * dlambda * g3;

    residual[t] = c + alpha_tilde * lagrangian[t];
}