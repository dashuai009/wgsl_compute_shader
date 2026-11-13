use std::borrow::Cow;

use nalgebra::Vector3;
use wgpu::util::DeviceExt;
use wgsl_compute_shader::mesh_gen::{generate_box_grid_tet, signed_tet_volume};
use wgsl_compute_shader::phys_data::{CpuPhysBuffers, SimParams, StiffnessPreset};

mod utils;

const WGSL: &str = include_str!("../assets/shaders/arap.wgsl");

struct GpuSimHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pos_buf: wgpu::Buffer,
    normals_buf: wgpu::Buffer,
    tet_state_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    predict_pl: wgpu::ComputePipeline,
    reset_pl: wgpu::ComputePipeline,
    copy_mid_pl: wgpu::ComputePipeline,
    arap_pl: wgpu::ComputePipeline,
    update_vel_pl: wgpu::ComputePipeline,
    clear_n_pl: wgpu::ComputePipeline,
    accum_n_pl: wgpu::ComputePipeline,
    norm_n_pl: wgpu::ComputePipeline,
    params: SimParams,
}

impl GpuSimHarness {
    async fn new(cpu: &CpuPhysBuffers, params: SimParams) -> Self {
        let (device, queue) = utils::get_wgpu_device_and_queue().await;

        let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pos"),
            contents: bytemuck::cast_slice(&cpu.pos_bits),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let normals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("normals"),
            contents: bytemuck::cast_slice(&cpu.normal_bits),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let verts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("verts"),
            contents: bytemuck::cast_slice(&cpu.verts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tets"),
            contents: bytemuck::cast_slice(&cpu.tets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tet_state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet_state"),
            contents: bytemuck::cast_slice(&cpu.tet_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let surf_tris_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_tris"),
            contents: bytemuck::cast_slice(&cpu.surface_tris),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let entry = |binding: u32, read_only: bool, uniform: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if uniform {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("xpbd_bgl"),
            entries: &[
                entry(0, false, false),
                entry(1, false, false),
                entry(2, false, false),
                entry(3, true, false),
                entry(4, false, false),
                entry(5, true, false),
                entry(6, true, true),
            ],
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("xpbd_pl_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("arap_wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
        });

        let mk_pl = |ep: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl_layout),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("xpbd_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: normals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: verts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tet_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: surf_tris_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            predict_pl: mk_pl("predict_positions"),
            reset_pl: mk_pl("reset_lagrange"),
            copy_mid_pl: mk_pl("copy_mid_pos"),
            arap_pl: mk_pl("cs_arap"),
            update_vel_pl: mk_pl("update_velocities"),
            clear_n_pl: mk_pl("clear_normals"),
            accum_n_pl: mk_pl("accumulate_normals"),
            norm_n_pl: mk_pl("normalize_normals"),
            device,
            queue,
            pos_buf,
            normals_buf,
            tet_state_buf,
            bind_group,
            params,
        }
    }

    fn step(&self, substeps: u32, jacobi_iters: u32) {
        let wg_v = self.params.nv.div_ceil(64);
        let wg_t = self.params.nt.div_ceil(64);
        let wg_s = self.params.n_surf_tris.div_ceil(64);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..substeps {
                pass.set_pipeline(&self.predict_pl);
                pass.dispatch_workgroups(wg_v, 1, 1);

                pass.set_pipeline(&self.reset_pl);
                pass.dispatch_workgroups(wg_t, 1, 1);

                for _ in 0..jacobi_iters {
                    pass.set_pipeline(&self.copy_mid_pl);
                    pass.dispatch_workgroups(wg_v, 1, 1);

                    pass.set_pipeline(&self.arap_pl);
                    pass.dispatch_workgroups(wg_t, 1, 1);
                }

                pass.set_pipeline(&self.update_vel_pl);
                pass.dispatch_workgroups(wg_v, 1, 1);
            }

            pass.set_pipeline(&self.clear_n_pl);
            pass.dispatch_workgroups(wg_v, 1, 1);
            pass.set_pipeline(&self.accum_n_pl);
            pass.dispatch_workgroups(wg_s, 1, 1);
            pass.set_pipeline(&self.norm_n_pl);
            pass.dispatch_workgroups(wg_v, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn read_vec3_bits(&self, src: &wgpu::Buffer) -> Vec<[f32; 3]> {
        let size = src.size();
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(src, 0, &rb, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
        let data = slice.get_mapped_range().expect("get_mapped_range failed");
        let bits: &[u32] = bytemuck::cast_slice(&data);
        let out = bits
            .chunks_exact(3)
            .map(|c| {
                [
                    f32::from_bits(c[0]),
                    f32::from_bits(c[1]),
                    f32::from_bits(c[2]),
                ]
            })
            .collect();
        drop(data);
        rb.unmap();
        out
    }

    fn read_tet_states(&self) -> Vec<[f32; 4]> {
        let size = self.tet_state_buf.size();
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb_tet"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.tet_state_buf, 0, &rb, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device poll failed");
        let data = slice.get_mapped_range().expect("get_mapped_range failed");
        let out: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        rb.unmap();
        out
    }
}

#[test]
fn test_xpbd_rest_pose_zero_drift() {
    pollster::block_on(async {
        let (positions, tets) =
            generate_box_grid_tet(3, 3, 3, [0.5, 0.5, 0.5], [0.2, 0.3, 0.1], [0.0, 2.0, 0.0]);
        let pinned = vec![false; positions.len()];
        let cpu = CpuPhysBuffers::build(&positions, &tets, &pinned, 1000.0);
        let (mu, lambda) = StiffnessPreset::Medium.lame_parameters();

        let params = SimParams {
            dt: 1.0 / 600.0,
            omega: 1.0,
            mu,
            lambda,
            gravity_y: 0.0,
            damping: 0.0,
            floor_y: 0.0,
            friction: 0.5,
            nt: cpu.canonical_tets.len() as u32,
            nv: positions.len() as u32,
            n_surf_tris: (cpu.surface_tris.len() / 3) as u32,
            enable_floor: 0,
        };

        let sim = GpuSimHarness::new(&cpu, params).await;
        sim.step(20, 8);

        let out_pos = sim.read_vec3_bits(&sim.pos_buf);
        let states = sim.read_tet_states();

        for (i, (p_orig, p_out)) in positions.iter().zip(out_pos.iter()).enumerate() {
            let err = (Vector3::from(*p_orig) - Vector3::from(*p_out)).norm();
            assert!(
                err < 1e-5,
                "Rest pose vertex {i} drifted by {err}: {p_orig:?} -> {p_out:?}"
            );
        }
        for (ti, s) in states.iter().enumerate() {
            assert!(
                s[2].abs() < 1e-5,
                "Rest pose tet {ti} residual {} != 0",
                s[2]
            );
        }
    });
}

#[test]
fn test_xpbd_inverted_recovery() {
    pollster::block_on(async {
        let (rest_pos, tets) =
            generate_box_grid_tet(2, 2, 2, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let pinned = vec![false; rest_pos.len()];
        let mut cpu = CpuPhysBuffers::build(&rest_pos, &tets, &pinned, 1000.0);

        // Invert every tetrahedron along Y (`y -> -0.35 * y`), giving det(F) = -0.35 < 0!
        for (v, p) in rest_pos.iter().enumerate() {
            let inverted_y = -0.35 * p[1];
            cpu.pos_bits[3 * v + 1] = inverted_y.to_bits();
            cpu.verts[v].pos_mid_im[1] = inverted_y;
            cpu.verts[v].pos_prev[1] = inverted_y;
        }

        let (mu, lambda) = StiffnessPreset::Stiff.lame_parameters();
        let params = SimParams {
            dt: 1.0 / 600.0,
            omega: 1.0,
            mu,
            lambda,
            gravity_y: 0.0,
            damping: 2.0,
            floor_y: -10.0,
            friction: 0.0,
            nt: cpu.canonical_tets.len() as u32,
            nv: rest_pos.len() as u32,
            n_surf_tris: (cpu.surface_tris.len() / 3) as u32,
            enable_floor: 0,
        };

        let sim = GpuSimHarness::new(&cpu, params).await;
        sim.step(120, 10);

        let out_pos = sim.read_vec3_bits(&sim.pos_buf);
        let mut total_vol = 0.0f32;
        for (ti, &[i0, i1, i2, i3]) in cpu.canonical_tets.iter().enumerate() {
            let v = signed_tet_volume(
                out_pos[i0 as usize],
                out_pos[i1 as usize],
                out_pos[i2 as usize],
                out_pos[i3 as usize],
            );
            assert!(
                v.is_finite() && v > 0.0,
                "Tet {ti} failed to un-invert: signed volume = {v}"
            );
            total_vol += v;
        }
        assert!(
            (total_vol - 1.0).abs() < 0.03,
            "Recovered cube volume {total_vol} too far from 1.0"
        );
    });
}

#[test]
fn test_xpbd_gravity_floor_and_pinned_beam() {
    pollster::block_on(async {
        let (positions, tets) =
            generate_box_grid_tet(6, 2, 2, [1.0, 0.25, 0.25], [0.0, 0.0, 0.0], [0.0, 0.30, 0.0]);
        let is_pinned: Vec<bool> = positions.iter().map(|p| p[0] <= -0.999).collect();
        let cpu = CpuPhysBuffers::build(&positions, &tets, &is_pinned, 1000.0);
        let (mu, lambda) = StiffnessPreset::Soft.lame_parameters();

        let params = SimParams {
            dt: 1.0 / 600.0,
            omega: 1.0,
            mu,
            lambda,
            gravity_y: -9.81,
            damping: 0.3,
            floor_y: 0.0,
            friction: 0.6,
            nt: cpu.canonical_tets.len() as u32,
            nv: positions.len() as u32,
            n_surf_tris: (cpu.surface_tris.len() / 3) as u32,
            enable_floor: 1,
        };

        let sim = GpuSimHarness::new(&cpu, params).await;
        sim.step(180, 8);

        let out_pos = sim.read_vec3_bits(&sim.pos_buf);
        let out_normals = sim.read_vec3_bits(&sim.normals_buf);

        let mut max_tip_drop: f32 = 0.0;
        for (v, (orig, cur)) in positions.iter().zip(out_pos.iter()).enumerate() {
            assert!(cur[0].is_finite() && cur[1].is_finite() && cur[2].is_finite());
            if is_pinned[v] {
                let drift = (Vector3::from(*orig) - Vector3::from(*cur)).norm();
                assert!(drift < 1e-7, "Pinned vertex {v} moved by {drift}");
            } else {
                assert!(
                    cur[1] >= -1e-5,
                    "Free vertex {v} penetrated floor: y = {}",
                    cur[1]
                );
                max_tip_drop = max_tip_drop.max(orig[1] - cur[1]);
            }

            let n_len = Vector3::from(out_normals[v]).norm();
            assert!(
                (n_len - 1.0).abs() < 1e-3,
                "Vertex {v} normal not unit length: {n_len}"
            );
        }
        assert!(
            max_tip_drop > 0.05,
            "Cantilever tip did not deflect under gravity (max drop = {max_tip_drop})"
        );
    });
}
