use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix3, Vector3};
use rand::{Rng, SeedableRng};
use wgpu::util::DeviceExt;

mod utils;

#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct SvdResultStd430 {
    u: [f32; 9],
    _pad0: [f32; 3],
    s: [f32; 3],
    v: [f32; 9],
}

const WGSL: &str = include_str!("../assets/shaders/arap.wgsl");

fn m3_from_row(arr: &[f32; 9]) -> Matrix3<f32> {
    Matrix3::from_row_slice(arr)
}

fn sort_unsigned_svd(
    u: &Matrix3<f32>,
    s: &Vector3<f32>,
    v: &Matrix3<f32>,
) -> (Matrix3<f32>, Vector3<f32>, Matrix3<f32>) {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap());

    let mut s_sorted = Vector3::zeros();
    let mut u_sorted = Matrix3::zeros();
    let mut v_sorted = Matrix3::zeros();

    for (col_new, &col_old) in (0..3).zip(idx.iter()) {
        s_sorted[col_new] = s[col_old];
        u_sorted.set_column(col_new, &u.column(col_old));
        v_sorted.set_column(col_new, &v.column(col_old));
    }
    (u_sorted, s_sorted, v_sorted)
}

fn orth_err(q: &Matrix3<f32>) -> f32 {
    (q.transpose() * q - Matrix3::<f32>::identity()).norm()
}

fn recon_rel_err(a: &Matrix3<f32>, u: &Matrix3<f32>, s: &Vector3<f32>, v: &Matrix3<f32>) -> f32 {
    let s_mat = Matrix3::from_diagonal(s);
    let a_hat = u * s_mat * v.transpose();
    (a - a_hat).norm() / a.norm().max(1e-12)
}

async fn run() {
    let (device, queue) = utils::get_wgpu_device_and_queue().await;

    let th = std::f32::consts::FRAC_PI_4;
    let (c, s) = (th.cos(), th.sin());
    let mut inputs: Vec<[f32; 9]> = vec![
        // Rank-deficient / near-singular
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        // Identity (det > 0)
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        // Inverted reflection (det < 0)
        [1.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 1.0],
        // Diagonal scale
        [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
        // Pure rotation Z 45 deg
        [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0],
        // General non-orthogonal
        [1.0, 2.0, 3.0, 0.5, 1.0, 1.5, -0.2, 0.3, 0.8],
    ];

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..16 {
        let mut mat = [0.0f32; 9];
        for elem in &mut mat {
            *elem = rng.gen_range(-5.0..5.0);
        }
        inputs.push(mat);
    }

    let n = inputs.len() as u32;
    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_mats"),
        contents: bytemuck::cast_slice(&inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_stride = std::mem::size_of::<SvdResultStd430>() as u64;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: out_stride * n as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_stride * n as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("empty"),
        entries: &[],
    });
    let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("empty_bg"),
        layout: &empty_bgl,
        entries: &[],
    });
    let data_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("data_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("layout"),
        bind_group_layouts: &[
            Some(&empty_bgl),
            Some(&empty_bgl),
            Some(&empty_bgl),
            Some(&data_bgl),
        ],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("svd_shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("svd_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("svd_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &data_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("svd_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_bg, &[]);
        pass.set_bind_group(1, &empty_bg, &[]);
        pass.set_bind_group(2, &empty_bg, &[]);
        pass.set_bind_group(3, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, out_stride * n as u64);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("device poll failed");
    let data = slice.get_mapped_range().expect("get_mapped_range failed");
    let gpu_results: &[SvdResultStd430] = bytemuck::cast_slice(&data);

    let tol_s = 1e-3f32;
    let tol_recon = 5e-4f32;
    let tol_orth = 5e-4f32;

    for (i, (a_row, g)) in inputs.iter().zip(gpu_results.iter()).enumerate() {
        let a = m3_from_row(a_row);
        let det_a = a.determinant();

        let svd = nalgebra::linalg::SVD::new(a, true, true);
        let u_cpu = svd.u.unwrap();
        let s_cpu = svd.singular_values;
        let v_cpu = svd.v_t.unwrap().transpose();
        let (_, mut s_expected, _) = sort_unsigned_svd(&u_cpu, &s_cpu, &v_cpu);
        // Signed SVD convention: det(U) = +1, det(V) = +1, sign(S.z) = sign(det(A))
        if det_a < -1e-6 {
            s_expected[2] = -s_expected[2];
        }

        let u_gpu = m3_from_row(&g.u);
        let v_gpu = m3_from_row(&g.v);
        let s_gpu = Vector3::new(g.s[0], g.s[1], g.s[2]);

        let s_err = (s_expected - s_gpu).amax();
        let recon_err = recon_rel_err(&a, &u_gpu, &s_gpu, &v_gpu);
        let orth_u = orth_err(&u_gpu);
        let orth_v = orth_err(&v_gpu);

        assert!(
            (u_gpu.determinant() - 1.0).abs() <= 1e-3,
            "Case {i}: det(U) = {} != +1",
            u_gpu.determinant()
        );
        assert!(
            (v_gpu.determinant() - 1.0).abs() <= 1e-3,
            "Case {i}: det(V) = {} != +1",
            v_gpu.determinant()
        );
        if det_a.abs() > 1e-4 {
            assert!(
                s_err <= tol_s,
                "Case {i}: signed singular values mismatch: expected {s_expected:?}, got {s_gpu:?}"
            );
        }
        assert!(
            recon_err <= tol_recon,
            "Case {i}: reconstruction error {recon_err} > {tol_recon}"
        );
        assert!(
            orth_u <= tol_orth && orth_v <= tol_orth,
            "Case {i}: orthogonality error U={orth_u}, V={orth_v}"
        );
    }

    drop(data);
    readback_buf.unmap();
}

#[test]
fn test_svd_signed_3x3() {
    pollster::block_on(run());
}
