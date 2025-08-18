use bytemuck::{Pod, Zeroable};
use nalgebra::{Matrix3, Vector3};
use slint::wgpu_24::wgpu;
use std::{borrow::Cow, default};
use wgpu::util::DeviceExt;
use rand::Rng;

mod utils;


#[repr(C)]
#[derive(Clone, Copy, Debug, Zeroable, Pod)]
struct SvdResultStd430 {
    U: [f32; 9],
    _pad0: [f32; 3],
    S: [f32; 3],
    V: [f32; 9],
}

const WGSL: &str = include_str!("../assets/shaders/arap.wgsl");

fn m3_from_row(arr: &[f32; 9]) -> Matrix3<f32> {
    Matrix3::from_row_slice(arr)
}
fn arr_from_m3(m: &Matrix3<f32>) -> [f32; 9] {
    let mut a = [0.0f32; 9];
    a.copy_from_slice(m.as_slice()); // nalgebra Matrix3 row_slice returns row-major when constructed via from_row_slice
    a
}

/// 将 (U,S,V) 重排为按 S 降序，U/V 同步重排。
fn sort_svd(
    u: &Matrix3<f32>,
    s: &Vector3<f32>,
    v: &Matrix3<f32>,
) -> (Matrix3<f32>, Vector3<f32>, Matrix3<f32>) {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| s[j].partial_cmp(&s[i]).unwrap()); // desc

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

/// 对齐 U/V 各列的符号，使与参考矩阵的对应列点积为非负；同时对 U/V 同列翻转，保持重建不变。
fn align_sign_to_ref(u: &mut Matrix3<f32>, v: &mut Matrix3<f32>, u_ref: &Matrix3<f32>) {
    for k in 0..3 {
        let dot = u.column(k).dot(&u_ref.column(k));
        if dot < 0.0 {
            u.set_column(k, &(-u.column(k)));
            v.set_column(k, &(-v.column(k)));
        }
    }
}

/// 正交性误差 ||Q^T Q - I||_F
fn orth_err(q: &Matrix3<f32>) -> f32 {
    let i = Matrix3::<f32>::identity();
    (q.transpose() * q - i).norm()
}

/// 以相对 Frobenius 误差衡量重建质量
fn recon_rel_err(a: &Matrix3<f32>, u: &Matrix3<f32>, s: &Vector3<f32>, v: &Matrix3<f32>) -> f32 {
    let s_mat = Matrix3::from_diagonal(s);
    let a_hat = u * s_mat * v.transpose();
    (a - a_hat).norm() / a.norm().max(1e-12)
}

async fn run() {
    // ---------- wgpu 初始化 ----------
    let (device, queue) = utils::get_wgpu_device_and_queue().await;

    // ---------- 测试输入 ----------
    let th = std::f32::consts::FRAC_PI_4;
    let (c, s) = (th.cos(), th.sin());
    let mut inputs: Vec<[f32; 9]> = vec![
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        // Identity
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        // Diagonal scale
        [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
        // Rotation Z 45°
        [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0],
        // 一般非正交矩阵
        [1.0, 2.0, 3.0, 0.5, 1.0, 1.5, -0.2, 0.3, 0.8],
    ];
    // rand input

    let mut rng = rand::thread_rng();

    // 生成 10 个随机 3x3 矩阵
    let rand_inputs: Vec<[f32; 9]> = (0..10)
        .map(|_| {
            let mut mat = [0.0f32; 9];
            for i in 0..9 {
                // 生成范围在 [-5.0, 5.0] 之间的随机数
                mat[i] = rng.gen_range(-5.0..5.0);
            }
            mat
        })
        .collect();

    inputs.extend(rand_inputs);
    let n = inputs.len() as u32;

    // ---------- Buffers ----------
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

    // ---------- BindGroup / Pipeline (注意 @group(3)) ----------
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
        bind_group_layouts: &[&empty_bgl, &empty_bgl, &empty_bgl, &data_bgl],
        push_constant_ranges: &[],
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

    // ---------- Dispatch ----------
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
        let wg = 64u32;
        pass.dispatch_workgroups((n + wg - 1) / wg, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, out_stride * n as u64);
    queue.submit(Some(encoder.finish()));

    // ---------- 读回 ----------
    let slice = readback_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let gpu_results: &[SvdResultStd430] = bytemuck::cast_slice(&data);

    // ---------- 与 nalgebra 验证 ----------
    let tol_s = 1e-4f32; // 奇异值容忍
    let tol_recon = 5e-4f32; // 重建相对误差
    let tol_orth = 5e-4f32; // 正交性

    for (i, (a_row, g)) in inputs.iter().zip(gpu_results.iter()).enumerate() {
        println!("----- Case {i} -----");
        let a = m3_from_row(a_row);
        println!("gpu_results = {:?}", gpu_results[i]);

        // CPU 基准
        let svd = nalgebra::linalg::SVD::new(a.clone(), true, true);
        println!("svd_nalgebra = {:?}", svd);
        let mut u_cpu = svd.u.as_ref().unwrap().clone_owned();
        let s_cpu = svd.singular_values.clone_owned();
        let mut v_cpu = svd.v_t.as_ref().unwrap().transpose();

        // GPU 结果 -> Matrix
        let mut u_gpu = m3_from_row(&g.U);
        let mut v_gpu = m3_from_row(&g.V);
        let mut s_gpu = Vector3::new(g.S[0], g.S[1], g.S[2]);

        // 降序重排
        let (u_cpu, s_cpu, v_cpu) = sort_svd(&u_cpu, &s_cpu, &v_cpu);
        (u_gpu, s_gpu, v_gpu) = sort_svd(&u_gpu, &s_gpu, &v_gpu);

        // 符号对齐：以 U 列为参考（也可用 V）
        align_sign_to_ref(&mut u_gpu, &mut v_gpu, &u_cpu);

        // 误差评估
        let s_err = (s_cpu - s_gpu).amax(); // max-abs diff
        let recon_err_gpu = recon_rel_err(&a, &u_gpu, &s_gpu, &v_gpu);
        let recon_err_cpu = recon_rel_err(&a, &u_cpu, &s_cpu, &v_cpu);
        let orth_u = orth_err(&u_gpu);
        let orth_v = orth_err(&v_gpu);
        println!("A = {:?}", a_row);
        println!("S (cpu) = {:?}", [s_cpu[0], s_cpu[1], s_cpu[2]]);
        println!("S (gpu) = {:?}", [s_gpu[0], s_gpu[1], s_gpu[2]]);
        println!("max |ΔS|  = {:.3e}", s_err);
        println!(
            "recon err (gpu) = {:.3e}, (cpu) = {:.3e}",
            recon_err_gpu, recon_err_cpu
        );
        println!("orth err  U = {:.3e}, V = {:.3e}", orth_u, orth_v);

        // 断言（用于自动化测试）
        // assert!(s_err <= tol_s, "singular values too far");
        assert!(
            recon_err_gpu <= tol_recon,
            "gpu reconstruction error too large"
        );
        assert!(
            orth_u <= tol_orth && orth_v <= tol_orth,
            "gpu U/V not orthogonal enough"
        );
    }

    drop(data);
    readback_buf.unmap();

    println!("All cases passed within tolerances.");
}

#[cfg(test)]
mod test {
    use super::run;

    #[test]
    fn test_svd() {
        pollster::block_on(run());
    }
}
