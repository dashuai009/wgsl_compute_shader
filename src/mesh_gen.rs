use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use nalgebra::{Matrix3, Rotation3, Vector3};

/// Compute signed volume of tetrahedron `[p0, p1, p2, p3]`:
/// `V = det([p1 - p0, p2 - p0, p3 - p0]) / 6`.
pub fn signed_tet_volume(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f32 {
    let v0 = Vector3::from(p0);
    let d1 = Vector3::from(p1) - v0;
    let d2 = Vector3::from(p2) - v0;
    let d3 = Vector3::from(p3) - v0;
    Matrix3::from_columns(&[d1, d2, d3]).determinant() / 6.0
}

/// Ensure every tetrahedron has strictly positive signed volume (`det(D_m) > 0`),
/// swapping the last two indices `[i0, i1, i3, i2]` whenever inverted.
pub fn canonicalize_tets(positions: &[[f32; 3]], tets: &mut [[u32; 4]]) {
    for tet in tets.iter_mut() {
        let [i0, i1, i2, i3] = *tet;
        let vol = signed_tet_volume(
            positions[i0 as usize],
            positions[i1 as usize],
            positions[i2 as usize],
            positions[i3 as usize],
        );
        if vol < 0.0 {
            tet.swap(2, 3);
        }
    }
}

/// Extract outward-oriented CCW boundary triangles (`3 * N_surf_tris` indices)
/// from positively oriented tetrahedra (`det(D_m) > 0`).
pub fn extract_boundary_triangles(tets: &[[u32; 4]]) -> Vec<u32> {
    let mut face_map: HashMap<[u32; 3], ([u32; 3], u32)> = HashMap::with_capacity(tets.len() * 2);

    for &[i0, i1, i2, i3] in tets {
        let faces = [
            [i1, i2, i3],
            [i0, i3, i2],
            [i0, i1, i3],
            [i0, i2, i1],
        ];
        for oriented in faces {
            let mut key = oriented;
            key.sort_unstable();
            face_map
                .entry(key)
                .and_modify(|e| e.1 += 1)
                .or_insert((oriented, 1));
        }
    }

    let mut keys: Vec<[u32; 3]> = face_map
        .iter()
        .filter_map(|(k, (_, count))| (*count == 1).then_some(*k))
        .collect();
    keys.sort_unstable();

    let mut tris = Vec::with_capacity(keys.len() * 3);
    for k in keys {
        let (oriented, _) = face_map[&k];
        tris.extend_from_slice(&oriented);
    }
    tris
}

/// Compute area-weighted smooth vertex normals over `positions` (`NV` vertices)
/// from `surface_tris` (`3 * N_surf_tris` indices).
pub fn compute_smooth_normals(positions: &[[f32; 3]], surface_tris: &[u32]) -> Vec<[f32; 3]> {
    let mut accum = vec![Vector3::<f32>::zeros(); positions.len()];
    for tri in surface_tris.chunks_exact(3) {
        let ia = tri[0] as usize;
        let ib = tri[1] as usize;
        let ic = tri[2] as usize;
        let pa = Vector3::from(positions[ia]);
        let pb = Vector3::from(positions[ib]);
        let pc = Vector3::from(positions[ic]);
        let n = (pb - pa).cross(&(pc - pa));
        accum[ia] += n;
        accum[ib] += n;
        accum[ic] += n;
    }

    accum
        .into_iter()
        .map(|n| {
            let len = n.norm();
            if len > 1e-12 {
                let u = n / len;
                [u.x, u.y, u.z]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect()
}

/// Construct a Bevy `Mesh` directly from `(positions, surface_tris)`.
/// Vertex buffer indices `0..NV` match `positions` 1-to-1 so `@vertex` can index
/// deformed positions & normals directly by `@builtin(vertex_index)`.
pub fn build_bevy_surface_mesh(positions: &[[f32; 3]], surface_tris: &[u32]) -> Mesh {
    let normals = compute_smooth_normals(positions, surface_tris);
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions.to_vec());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(surface_tris.to_vec()));
    mesh
}

/// Conformal 6-tetrahedra Kuhn / Freudenthal decomposition of hex cell
/// `(v000, v100, v010, v110, v001, v101, v011, v111)` along main diagonal `v000 -> v111`.
fn push_freudenthal_hex(tets: &mut Vec<[u32; 4]>, v: [u32; 8]) {
    let [v000, v100, v010, v110, v001, v101, v011, v111] = v;
    tets.push([v000, v100, v110, v111]);
    tets.push([v000, v100, v111, v101]);
    tets.push([v000, v010, v111, v110]);
    tets.push([v000, v010, v011, v111]);
    tets.push([v000, v001, v101, v111]);
    tets.push([v000, v001, v111, v011]);
}

/// Generate a conformal tetrahedral box grid of resolution `(nx, ny, nz)` cells
/// spanning `[-hx, hx] x [-hy, hy] x [-hz, hz]`, rotated by Euler `(rx, ry, rz)` and translated by `offset`.
pub fn generate_box_grid_tet(
    nx: usize,
    ny: usize,
    nz: usize,
    half_extents: [f32; 3],
    euler_xyz: [f32; 3],
    offset: [f32; 3],
) -> (Vec<[f32; 3]>, Vec<[u32; 4]>) {
    let rot = Rotation3::from_euler_angles(euler_xyz[0], euler_xyz[1], euler_xyz[2]);
    let trans = Vector3::from(offset);
    let [hx, hy, hz] = half_extents;

    let vid = |ix: usize, iy: usize, iz: usize| -> u32 {
        ((iz * (ny + 1) + iy) * (nx + 1) + ix) as u32
    };

    let mut positions = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for iz in 0..=nz {
        let tz = iz as f32 / nz as f32;
        let z = -hz + 2.0 * hz * tz;
        for iy in 0..=ny {
            let ty = iy as f32 / ny as f32;
            let y = -hy + 2.0 * hy * ty;
            for ix in 0..=nx {
                let tx = ix as f32 / nx as f32;
                let x = -hx + 2.0 * hx * tx;
                let p = rot * Vector3::new(x, y, z) + trans;
                positions.push([p.x, p.y, p.z]);
            }
        }
    }

    let mut tets = Vec::with_capacity(nx * ny * nz * 6);
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                push_freudenthal_hex(
                    &mut tets,
                    [
                        vid(ix, iy, iz),
                        vid(ix + 1, iy, iz),
                        vid(ix, iy + 1, iz),
                        vid(ix + 1, iy + 1, iz),
                        vid(ix, iy, iz + 1),
                        vid(ix + 1, iy, iz + 1),
                        vid(ix, iy + 1, iz + 1),
                        vid(ix + 1, iy + 1, iz + 1),
                    ],
                );
            }
        }
    }

    canonicalize_tets(&positions, &mut tets);
    (positions, tets)
}

///Shirley-Chiu concentric mapping from `[-1, 1]^2` square to unit disk `r <= 1`.
fn square_to_unit_disk(a: f32, b: f32) -> (f32, f32) {
    if a.abs() < 1e-7 && b.abs() < 1e-7 {
        return (0.0, 0.0);
    }
    let (r, phi) = if a * a > b * b {
        (a, std::f32::consts::FRAC_PI_4 * (b / a))
    } else {
        (b, std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (a / b))
    };
    (r * phi.cos(), r * phi.sin())
}

/// Generate a conformal tetrahedral torus (`n_major` ring segments, `n_minor x n_minor` cross-section).
pub fn generate_torus_tet(
    n_major: usize,
    n_minor: usize,
    major_radius: f32,
    minor_radius: f32,
    euler_xyz: [f32; 3],
    offset: [f32; 3],
) -> (Vec<[f32; 3]>, Vec<[u32; 4]>) {
    let rot = Rotation3::from_euler_angles(euler_xyz[0], euler_xyz[1], euler_xyz[2]);
    let trans = Vector3::from(offset);

    let ring_stride = (n_minor + 1) * (n_minor + 1);
    let vid = |imaj: usize, iu: usize, iv: usize| -> u32 {
        let r_idx = imaj % n_major;
        (r_idx * ring_stride + iv * (n_minor + 1) + iu) as u32
    };

    let mut positions = Vec::with_capacity(n_major * ring_stride);
    for imaj in 0..n_major {
        let theta = 2.0 * std::f32::consts::PI * (imaj as f32) / (n_major as f32);
        let (ct, st) = (theta.cos(), theta.sin());
        for iv in 0..=n_minor {
            let b = -1.0 + 2.0 * (iv as f32) / (n_minor as f32);
            for iu in 0..=n_minor {
                let a = -1.0 + 2.0 * (iu as f32) / (n_minor as f32);
                let (du, dv) = square_to_unit_disk(a, b);
                let radial = major_radius + minor_radius * du;
                let local = Vector3::new(radial * ct, minor_radius * dv, radial * st);
                let p = rot * local + trans;
                positions.push([p.x, p.y, p.z]);
            }
        }
    }

    let mut tets = Vec::with_capacity(n_major * n_minor * n_minor * 6);
    for imaj in 0..n_major {
        for iv in 0..n_minor {
            for iu in 0..n_minor {
                push_freudenthal_hex(
                    &mut tets,
                    [
                        vid(imaj, iu, iv),
                        vid(imaj, iu + 1, iv),
                        vid(imaj, iu, iv + 1),
                        vid(imaj, iu + 1, iv + 1),
                        vid(imaj + 1, iu, iv),
                        vid(imaj + 1, iu + 1, iv),
                        vid(imaj + 1, iu, iv + 1),
                        vid(imaj + 1, iu + 1, iv + 1),
                    ],
                );
            }
        }
    }

    canonicalize_tets(&positions, &mut tets);
    (positions, tets)
}
