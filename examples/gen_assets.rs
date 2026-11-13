use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use nalgebra::{Rotation3, Vector3};
use wgsl_compute_shader::mesh_gen::{
    canonicalize_tets, extract_boundary_triangles, generate_box_grid_tet, generate_torus_tet,
    signed_tet_volume,
};

/// Minimal binary GLB reader specifically for extracting indexed triangle soup `(verts, tris)`.
fn load_watertight_glb_surface(path: &Path) -> (Vec<Vector3<f32>>, Vec<[usize; 3]>) {
    let bytes = fs::read(path).expect("failed to read GLB");
    assert!(&bytes[0..4] == b"glTF", "invalid GLB magic");
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let json_str = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
    let bin_header = 20 + json_len;
    let bin_len = u32::from_le_bytes(bytes[bin_header..bin_header + 4].try_into().unwrap()) as usize;
    let bin = &bytes[bin_header + 8..bin_header + 8 + bin_len];

    // Parse bufferViews and accessors from known Houdini bunny_h.glb layout or lightweight JSON scan:
    // Use serde-free minimal parser for bufferViews/accessors in bunny_h.glb.
    let extract_u32_list = |section: &str, key: &str| -> Vec<usize> {
        let mut out = Vec::new();
        let needle = format!("\"{}\":", key);
        let mut rest = section;
        while let Some(pos) = rest.find(&needle) {
            rest = &rest[pos + needle.len()..];
            let trimmed = rest.trim_start();
            let end = trimmed
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(trimmed.len());
            if end > 0 {
                out.push(trimmed[..end].parse::<usize>().unwrap());
            }
        }
        out
    };

    fn find_json_array<'a>(json: &'a str, key: &str) -> &'a str {
        let start_key = json.find(key).unwrap() + key.len();
        let after = &json[start_key..];
        let arr_open = after.find('[').unwrap();
        let slice = &after[arr_open..];
        let mut depth = 0usize;
        for (i, ch) in slice.char_indices() {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return &slice[..=i];
                    }
                }
                _ => {}
            }
        }
        panic!("unclosed array for {key}");
    }

    fn split_objs(s: &str) -> Vec<&str> {
        let mut objs = Vec::new();
        let mut depth = 0usize;
        let mut start = None;
        for (i, ch) in s.char_indices() {
            match ch {
                '{' => {
                    if depth == 0 {
                        start = Some(i);
                    }
                    depth += 1;
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(st) = start {
                            objs.push(&s[st..=i]);
                        }
                    }
                }
                _ => {}
            }
        }
        objs
    }

    let acc_sec = find_json_array(json_str, "\"accessors\":");
    let bv_sec = find_json_array(json_str, "\"bufferViews\":");

    let bvs: Vec<(usize, usize)> = split_objs(bv_sec)
        .into_iter()
        .map(|o| {
            let off = extract_u32_list(o, "byteOffset").first().copied().unwrap_or(0);
            let len = extract_u32_list(o, "byteLength")[0];
            (off, len)
        })
        .collect();

    let accs: Vec<(usize, usize, usize, usize)> = split_objs(acc_sec)
        .into_iter()
        .map(|o| {
            let bv = extract_u32_list(o, "bufferView")[0];
            let off = extract_u32_list(o, "byteOffset").first().copied().unwrap_or(0);
            let count = extract_u32_list(o, "count")[0];
            let comp = extract_u32_list(o, "componentType")[0];
            (bv, off, count, comp)
        })
        .collect();

    // Accessor 0 = indices (5123 u16 or 5125 u32), Accessor 1 = POSITION (5126 VEC3 f32)
    let (idx_bv, idx_off, idx_count, idx_comp) = accs[0];
    let idx_base = bvs[idx_bv].0 + idx_off;
    let raw_indices: Vec<usize> = (0..idx_count)
        .map(|i| {
            if idx_comp == 5123 {
                let b = &bin[idx_base + i * 2..idx_base + i * 2 + 2];
                u16::from_le_bytes([b[0], b[1]]) as usize
            } else {
                let b = &bin[idx_base + i * 4..idx_base + i * 4 + 4];
                u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
            }
        })
        .collect();

    let (pos_bv, pos_off, pos_count, _) = accs[1];
    let pos_base = bvs[pos_bv].0 + pos_off;
    let raw_pos: Vec<Vector3<f32>> = (0..pos_count)
        .map(|i| {
            let o = pos_base + i * 12;
            let x = f32::from_le_bytes(bin[o..o + 4].try_into().unwrap());
            let y = f32::from_le_bytes(bin[o + 4..o + 8].try_into().unwrap());
            let z = f32::from_le_bytes(bin[o + 8..o + 12].try_into().unwrap());
            Vector3::new(x, y, z)
        })
        .collect();

    // Weld split vertices by quantized coordinates
    let mut map: HashMap<(i32, i32, i32), usize> = HashMap::new();
    let mut uniq = Vec::new();
    let mut remap = Vec::with_capacity(raw_pos.len());
    for p in raw_pos {
        let key = (
            (p.x * 1e5).round() as i32,
            (p.y * 1e5).round() as i32,
            (p.z * 1e5).round() as i32,
        );
        let id = *map.entry(key).or_insert_with(|| {
            let id = uniq.len();
            uniq.push(p);
            id
        });
        remap.push(id);
    }

    let tris: Vec<[usize; 3]> = raw_indices
        .chunks_exact(3)
        .map(|c| [remap[c[0]], remap[c[1]], remap[c[2]]])
        .collect();
    (uniq, tris)
}

/// Closest point on triangle `(a, b, c)` to query point `p` (Ericson, Real-Time Collision Detection).
fn closest_point_on_triangle(
    p: Vector3<f32>,
    a: Vector3<f32>,
    b: Vector3<f32>,
    c: Vector3<f32>,
) -> Vector3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + v * ab;
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + w * ac;
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Exact generalized winding number `Omega(p)` (Van Oosterom & Strackee solid angle formula)
/// for closed triangle mesh: `Omega(p) ~ 1.0` inside, `~ 0.0` outside.
fn winding_number(p: Vector3<f32>, verts: &[Vector3<f32>], tris: &[[usize; 3]]) -> f32 {
    let mut sum_omega = 0.0f32;
    for &[ia, ib, ic] in tris {
        let a = verts[ia] - p;
        let b = verts[ib] - p;
        let c = verts[ic] - p;
        let la = a.norm();
        let lb = b.norm();
        let lc = c.norm();
        if la < 1e-8 || lb < 1e-8 || lc < 1e-8 {
            return 0.5;
        }
        let num = a.dot(&b.cross(&c));
        let den = la * lb * lc + a.dot(&b) * lc + b.dot(&c) * la + c.dot(&a) * lb;
        sum_omega += 2.0 * num.atan2(den);
    }
    sum_omega / (4.0 * std::f32::consts::PI)
}

/// Signed distance (`< 0` inside, `> 0` outside) and closest surface point.
fn signed_distance_and_closest(
    p: Vector3<f32>,
    verts: &[Vector3<f32>],
    tris: &[[usize; 3]],
) -> (f32, Vector3<f32>) {
    let mut best_d2 = f32::MAX;
    let mut best_cp = Vector3::zeros();
    for &[ia, ib, ic] in tris {
        let cp = closest_point_on_triangle(p, verts[ia], verts[ib], verts[ic]);
        let d2 = (p - cp).norm_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_cp = cp;
        }
    }
    let dist = best_d2.sqrt();
    let wn = winding_number(p, verts, tris);
    let sign = if wn >= 0.5 { -1.0 } else { 1.0 };
    (sign * dist, best_cp)
}

/// Labelle & Shewchuk Isosurface Stuffing on a Body-Centered Cubic (BCC) tetrahedral lattice.
fn generate_bunny_isosurface_stuffed(
    glb_path: &Path,
    cell_size: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 4]>) {
    let (mut surf_verts, surf_tris) = load_watertight_glb_surface(glb_path);

    // Normalize Bunny to height 1.55m, mild roll tilt, resting height y_min = 0.90m
    let mut bmin = Vector3::repeat(f32::MAX);
    let mut bmax = Vector3::repeat(f32::MIN);
    for v in &surf_verts {
        bmin = bmin.inf(v);
        bmax = bmax.sup(v);
    }
    let center = 0.5 * (bmin + bmax);
    let scale = 1.55 / (bmax.y - bmin.y);
    let tilt = Rotation3::from_euler_angles(0.08, 0.25, -0.12);
    for v in &mut surf_verts {
        *v = tilt * ((*v - center) * scale);
    }
    bmin = Vector3::repeat(f32::MAX);
    bmax = Vector3::repeat(f32::MIN);
    for v in &surf_verts {
        bmin = bmin.inf(v);
        bmax = bmax.sup(v);
    }
    let xz_center = Vector3::new(0.5 * (bmin.x + bmax.x), 0.0, 0.5 * (bmin.z + bmax.z));
    let y_shift = 0.90 - bmin.y;
    for v in &mut surf_verts {
        v.x -= xz_center.x;
        v.z -= xz_center.z;
        v.y += y_shift;
    }

    // Recompute transformed bounds and pad by 2.5 BCC cells
    bmin = Vector3::repeat(f32::MAX);
    bmax = Vector3::repeat(f32::MIN);
    for v in &surf_verts {
        bmin = bmin.inf(v);
        bmax = bmax.sup(v);
    }
    let pad = 2.5 * cell_size;
    let origin = bmin - Vector3::repeat(pad);
    let span = (bmax + Vector3::repeat(pad)) - origin;
    let nx = (span.x / cell_size).ceil() as usize;
    let ny = (span.y / cell_size).ceil() as usize;
    let nz = (span.z / cell_size).ceil() as usize;

    let primal_count = (nx + 1) * (ny + 1) * (nz + 1);
    let dual_count = nx * ny * nz;
    let primal_id = |i: usize, j: usize, k: usize| -> usize { (k * (ny + 1) + j) * (nx + 1) + i };
    let dual_id = |i: usize, j: usize, k: usize| -> usize { primal_count + (k * ny + j) * nx + i };

    let mut lattice_pos = Vec::with_capacity(primal_count + dual_count);
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                lattice_pos.push(
                    origin + Vector3::new(i as f32, j as f32, k as f32) * cell_size,
                );
            }
        }
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                lattice_pos.push(
                    origin
                        + Vector3::new(i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5) * cell_size,
                );
            }
        }
    }

    // Build all BCC tetrahedra around interior X, Y, Z dual edges
    let mut bcc_tets: Vec<[usize; 4]> = Vec::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if i + 1 < nx {
                    let d0 = dual_id(i, j, k);
                    let d1 = dual_id(i + 1, j, k);
                    let p00 = primal_id(i + 1, j, k);
                    let p10 = primal_id(i + 1, j + 1, k);
                    let p11 = primal_id(i + 1, j + 1, k + 1);
                    let p01 = primal_id(i + 1, j, k + 1);
                    for [a, b] in [[p00, p10], [p10, p11], [p11, p01], [p01, p00]] {
                        let mut t = [d0, d1, a, b];
                        t.sort_unstable();
                        bcc_tets.push(t);
                    }
                }
                if j + 1 < ny {
                    let d0 = dual_id(i, j, k);
                    let d1 = dual_id(i, j + 1, k);
                    let p00 = primal_id(i, j + 1, k);
                    let p10 = primal_id(i + 1, j + 1, k);
                    let p11 = primal_id(i + 1, j + 1, k + 1);
                    let p01 = primal_id(i, j + 1, k + 1);
                    for [a, b] in [[p00, p10], [p10, p11], [p11, p01], [p01, p00]] {
                        let mut t = [d0, d1, a, b];
                        t.sort_unstable();
                        bcc_tets.push(t);
                    }
                }
                if k + 1 < nz {
                    let d0 = dual_id(i, j, k);
                    let d1 = dual_id(i, j, k + 1);
                    let p00 = primal_id(i, j, k + 1);
                    let p10 = primal_id(i + 1, j, k + 1);
                    let p11 = primal_id(i + 1, j + 1, k + 1);
                    let p01 = primal_id(i, j + 1, k + 1);
                    for [a, b] in [[p00, p10], [p10, p11], [p11, p01], [p01, p00]] {
                        let mut t = [d0, d1, a, b];
                        t.sort_unstable();
                        bcc_tets.push(t);
                    }
                }
            }
        }
    }

    // Evaluate signed distance & snap lattice vertices within alpha * cell_size onto exact Bunny surface
    let snap_dist = 0.24 * cell_size;
    let mut phi = vec![1.0f32; lattice_pos.len()];
    let mut state = vec![1i8; lattice_pos.len()]; // -1 = inside, 0 = snapped surface, +1 = outside

    for v in 0..lattice_pos.len() {
        let p = lattice_pos[v];
        // Fast AABB reject for outer vertices
        if p.x < bmin.x - snap_dist
            || p.x > bmax.x + snap_dist
            || p.y < bmin.y - snap_dist
            || p.y > bmax.y + snap_dist
            || p.z < bmin.z - snap_dist
            || p.z > bmax.z + snap_dist
        {
            phi[v] = cell_size;
            state[v] = 1;
            continue;
        }
        let (sd, cp) = signed_distance_and_closest(p, &surf_verts, &surf_tris);
        if sd.abs() <= snap_dist {
            lattice_pos[v] = cp;
            phi[v] = 0.0;
            state[v] = 0;
        } else if sd < 0.0 {
            phi[v] = sd;
            state[v] = -1;
        } else {
            phi[v] = sd;
            state[v] = 1;
        }
    }

    // Cut strictly sign-changing edges (-1 <-> +1) with bisection + surface projection
    let mut cut_cache: HashMap<(usize, usize), usize> = HashMap::new();
    let mut get_cut = |u: usize,
                       v: usize,
                       lattice_pos: &mut Vec<Vector3<f32>>,
                       state: &mut Vec<i8>|
     -> usize {
        let key = if u < v { (u, v) } else { (v, u) };
        if let Some(&id) = cut_cache.get(&key) {
            return id;
        }
        let (inside_v, outside_v) = if phi[u] < 0.0 { (u, v) } else { (v, u) };
        let mut lo = lattice_pos[inside_v];
        let mut hi = lattice_pos[outside_v];
        for _ in 0..10 {
            let mid = 0.5 * (lo + hi);
            if winding_number(mid, &surf_verts, &surf_tris) >= 0.5 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let mid = 0.5 * (lo + hi);
        let (_, cp) = signed_distance_and_closest(mid, &surf_verts, &surf_tris);
        let id = lattice_pos.len();
        lattice_pos.push(cp);
        state.push(0);
        cut_cache.insert(key, id);
        id
    };

    // Conformal simplicial split using universal index-ordered diagonal rule
    let mut raw_tets: Vec<[usize; 4]> = Vec::new();
    let bcc_vol = cell_size * cell_size * cell_size / 12.0;

    for tet in bcc_tets {
        let mut ins = Vec::with_capacity(4);
        let mut zeros = Vec::with_capacity(4);
        let mut outs = Vec::with_capacity(4);
        for &vid in &tet {
            match state[vid] {
                -1 => ins.push(vid),
                0 => zeros.push(vid),
                _ => outs.push(vid),
            }
        }
        // Note: `tet` was sorted, so `ins`, `zeros`, `outs` are already sorted in ascending vertex ID!
        match (ins.len(), zeros.len(), outs.len()) {
            (0, 4, 0) => {
                // Keep all-surface tet at thin features (e.g. Bunny ears) iff centroid is inside & non-flat
                let c = 0.25
                    * (lattice_pos[tet[0]]
                        + lattice_pos[tet[1]]
                        + lattice_pos[tet[2]]
                        + lattice_pos[tet[3]]);
                let v = signed_tet_volume(
                    lattice_pos[tet[0]].into(),
                    lattice_pos[tet[1]].into(),
                    lattice_pos[tet[2]].into(),
                    lattice_pos[tet[3]].into(),
                )
                .abs();
                if v > 0.05 * bcc_vol && winding_number(c, &surf_verts, &surf_tris) >= 0.5 {
                    raw_tets.push(tet);
                }
            }
            (0, _, _) => {}
            (_, _, 0) => {
                raw_tets.push(tet);
            }
            (1, 0, 3) => {
                let i0 = ins[0];
                let c0 = get_cut(i0, outs[0], &mut lattice_pos, &mut state);
                let c1 = get_cut(i0, outs[1], &mut lattice_pos, &mut state);
                let c2 = get_cut(i0, outs[2], &mut lattice_pos, &mut state);
                raw_tets.push([i0, c0, c1, c2]);
            }
            (1, 1, 2) => {
                let i0 = ins[0];
                let z0 = zeros[0];
                let c0 = get_cut(i0, outs[0], &mut lattice_pos, &mut state);
                let c1 = get_cut(i0, outs[1], &mut lattice_pos, &mut state);
                raw_tets.push([i0, z0, c0, c1]);
            }
            (1, 2, 1) => {
                let i0 = ins[0];
                let c0 = get_cut(i0, outs[0], &mut lattice_pos, &mut state);
                raw_tets.push([i0, zeros[0], zeros[1], c0]);
            }
            (2, 0, 2) => {
                let (i0, i1) = (ins[0], ins[1]);
                let (o0, o1) = (outs[0], outs[1]);
                let c00 = get_cut(i0, o0, &mut lattice_pos, &mut state);
                let c01 = get_cut(i0, o1, &mut lattice_pos, &mut state);
                let c10 = get_cut(i1, o0, &mut lattice_pos, &mut state);
                let c11 = get_cut(i1, o1, &mut lattice_pos, &mut state);
                raw_tets.push([i0, i1, c10, c11]);
                raw_tets.push([i0, c10, c00, c11]);
                raw_tets.push([i0, c00, c01, c11]);
            }
            (2, 1, 1) => {
                let (i0, i1) = (ins[0], ins[1]);
                let z0 = zeros[0];
                let o0 = outs[0];
                let c0 = get_cut(i0, o0, &mut lattice_pos, &mut state);
                let c1 = get_cut(i1, o0, &mut lattice_pos, &mut state);
                raw_tets.push([z0, i0, i1, c1]);
                raw_tets.push([z0, i0, c1, c0]);
            }
            (3, 0, 1) => {
                let (i0, i1, i2) = (ins[0], ins[1], ins[2]);
                let o0 = outs[0];
                let c0 = get_cut(i0, o0, &mut lattice_pos, &mut state);
                let c1 = get_cut(i1, o0, &mut lattice_pos, &mut state);
                let c2 = get_cut(i2, o0, &mut lattice_pos, &mut state);
                raw_tets.push([i0, i1, i2, c2]);
                raw_tets.push([i0, i1, c2, c1]);
                raw_tets.push([i0, c1, c2, c0]);
            }
            _ => unreachable!(),
        }
    }

    // Keep largest connected face-adjacent component (eliminates any isolated outlier micro-tet)
    let mut face_to_tets: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for (ti, t) in raw_tets.iter().enumerate() {
        for f in [
            [t[1], t[2], t[3]],
            [t[0], t[2], t[3]],
            [t[0], t[1], t[3]],
            [t[0], t[1], t[2]],
        ] {
            let mut key = f;
            key.sort_unstable();
            face_to_tets.entry(key).or_default().push(ti);
        }
    }
    let mut visited = vec![false; raw_tets.len()];
    let mut best_comp = Vec::new();
    for start in 0..raw_tets.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut comp = Vec::new();
        while let Some(cur) = stack.pop() {
            comp.push(cur);
            let t = raw_tets[cur];
            for f in [
                [t[1], t[2], t[3]],
                [t[0], t[2], t[3]],
                [t[0], t[1], t[3]],
                [t[0], t[1], t[2]],
            ] {
                let mut key = f;
                key.sort_unstable();
                for &nb in &face_to_tets[&key] {
                    if !visited[nb] {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                }
            }
        }
        if comp.len() > best_comp.len() {
            best_comp = comp;
        }
    }

    // Compact vertex indices to 0..NV
    let mut v_remap: HashMap<usize, u32> = HashMap::new();
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut is_surface_vert: Vec<bool> = Vec::new();
    let mut tets: Vec<[u32; 4]> = Vec::with_capacity(best_comp.len());

    for ti in best_comp {
        let mut mapped = [0u32; 4];
        for (k, &old_v) in raw_tets[ti].iter().enumerate() {
            let new_v = *v_remap.entry(old_v).or_insert_with(|| {
                let id = positions.len() as u32;
                let p = lattice_pos[old_v];
                positions.push([p.x, p.y, p.z]);
                is_surface_vert.push(state[old_v] == 0);
                id
            });
            mapped[k] = new_v;
        }
        tets.push(mapped);
    }

    canonicalize_tets(&positions, &mut tets);

    // 3 passes of volume-guarded Laplacian smoothing (tangential + surface reprojection on boundary,
    // full 3D relaxation in interior) for smooth surface triangles and high minimum tet quality.
    let mut v_neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); positions.len()];
    let mut v_inc_tets: Vec<Vec<usize>> = vec![Vec::new(); positions.len()];
    for (ti, &[a, b, c, d]) in tets.iter().enumerate() {
        let vs = [a as usize, b as usize, c as usize, d as usize];
        for &u in &vs {
            v_inc_tets[u].push(ti);
            for &w in &vs {
                if u != w {
                    v_neighbors[u].insert(w);
                }
            }
        }
    }
    let min_allowed_vol = 0.02 * bcc_vol;
    for _pass in 0..3 {
        for v in 0..positions.len() {
            if v_neighbors[v].is_empty() {
                continue;
            }
            let cur = Vector3::from(positions[v]);
            let mut avg = Vector3::zeros();
            let mut count = 0.0f32;
            for &nb in &v_neighbors[v] {
                if !is_surface_vert[v] || is_surface_vert[nb] {
                    avg += Vector3::from(positions[nb]);
                    count += 1.0;
                }
            }
            if count == 0.0 {
                continue;
            }
            avg /= count;
            let candidate = if is_surface_vert[v] {
                let blended = cur.lerp(&avg, 0.35);
                let (_, cp) = signed_distance_and_closest(blended, &surf_verts, &surf_tris);
                cp
            } else {
                cur.lerp(&avg, 0.40)
            };

            let old_pos = positions[v];
            let cand_arr = [candidate.x, candidate.y, candidate.z];
            let mut cur_min_v = f32::MAX;
            let mut new_min_v = f32::MAX;
            for &ti in &v_inc_tets[v] {
                let [i0, i1, i2, i3] = tets[ti];
                cur_min_v = cur_min_v.min(signed_tet_volume(
                    positions[i0 as usize],
                    positions[i1 as usize],
                    positions[i2 as usize],
                    positions[i3 as usize],
                ));
            }
            positions[v] = cand_arr;
            for &ti in &v_inc_tets[v] {
                let [i0, i1, i2, i3] = tets[ti];
                new_min_v = new_min_v.min(signed_tet_volume(
                    positions[i0 as usize],
                    positions[i1 as usize],
                    positions[i2 as usize],
                    positions[i3 as usize],
                ));
            }
            if new_min_v < min_allowed_vol.min(cur_min_v * 0.8) {
                positions[v] = old_pos;
            }
        }
    }

    (positions, tets)
}

fn validate_and_write_asset(name: &str, positions: &[[f32; 3]], tets: &[[u32; 4]]) {
    assert!(!positions.is_empty() && !tets.is_empty());
    let mut min_vol = f32::MAX;
    let mut max_vol = 0.0f32;
    let mut total_vol = 0.0f32;
    for &[i0, i1, i2, i3] in tets {
        let v = signed_tet_volume(
            positions[i0 as usize],
            positions[i1 as usize],
            positions[i2 as usize],
            positions[i3 as usize],
        );
        assert!(
            v > 1e-8,
            "Asset `{name}` has degenerate/inverted tet with volume {v}"
        );
        min_vol = min_vol.min(v);
        max_vol = max_vol.max(v);
        total_vol += v;
    }

    let surf_tris = extract_boundary_triangles(tets);
    let mut edge_deg: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in surf_tris.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let k = if a < b { (a, b) } else { (b, a) };
            *edge_deg.entry(k).or_insert(0) += 1;
        }
    }
    let non_manifold = edge_deg.values().filter(|&&d| d != 2).count();
    assert_eq!(
        non_manifold, 0,
        "Asset `{name}` boundary surface has {non_manifold} non-manifold edges!"
    );

    let dir = Path::new("assets/models").join(name);
    fs::create_dir_all(&dir).unwrap();
    let pos_path = dir.join(format!("{name}_positions.vec3f32bin"));
    let tet_path = dir.join(format!("{name}_tet_indices.vec4u32bin"));
    fs::write(&pos_path, bytemuck::cast_slice(positions)).unwrap();
    fs::write(&tet_path, bytemuck::cast_slice(tets)).unwrap();

    println!(
        "[{name:>6}] NV={:<5} NT={:<5} SurfTris={:<5} Vol=[{:.2e}, {:.2e}] TotalVol={:.4} Size={:.1} KB",
        positions.len(),
        tets.len(),
        surf_tris.len() / 3,
        min_vol,
        max_vol,
        total_vol,
        ((positions.len() * 12 + tets.len() * 16) as f32) / 1024.0
    );
}

fn main() {
    let bunny_glb = Path::new("assets/bunny_h.glb");
    if bunny_glb.exists() {
        let (bunny_pos, bunny_tets) = generate_bunny_isosurface_stuffed(bunny_glb, 0.135);
        validate_and_write_asset("bunny", &bunny_pos, &bunny_tets);
    } else {
        println!("[ bunny] skipped (assets/bunny_h.glb not present; keeping existing binary)");
    }

    let (cube_pos, cube_tets) = generate_box_grid_tet(
        6,
        6,
        6,
        [0.52, 0.52, 0.52],
        [0.32, 0.20, 0.24],
        [0.0, 1.75, 0.0],
    );
    validate_and_write_asset("cube", &cube_pos, &cube_tets);

    let (torus_pos, torus_tets) = generate_torus_tet(
        24,
        4,
        0.75,
        0.27,
        [0.65, 0.15, 0.25],
        [0.0, 1.80, 0.0],
    );
    validate_and_write_asset("torus", &torus_pos, &torus_tets);

    let (beam_pos, beam_tets) = generate_box_grid_tet(
        12,
        4,
        4,
        [1.05, 0.26, 0.26],
        [0.0, 0.0, 0.0],
        [0.05, 1.45, 0.0],
    );
    validate_and_write_asset("beam", &beam_pos, &beam_tets);
}
