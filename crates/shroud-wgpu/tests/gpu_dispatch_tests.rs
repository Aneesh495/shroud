//! End-to-end GPU compute dispatch tests running Shroud WGSL shaders via WGPU.

use bytemuck::{Pod, Zeroable};
use shroud_core::atmospheric::henyey_greenstein;
use shroud_wgpu::buffers::GpuBufferHelper;
use shroud_wgpu::context::GpuContext;
use shroud_wgpu::pipeline::ComputePipelineHelper;
use std::path::PathBuf;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PhaseQueryInputGpu {
    incident_dir: [f32; 3],
    g: f32,
    u_sample: [f32; 2],
    cos_query: f32,
    padding: f32,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_gpu_dispatch_rng_and_colorimetry() {
    let ctx = match pollster::block_on(GpuContext::init_headless()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping GPU dispatch test (no headless adapter): {e}");
            return;
        }
    };

    // 1. Test shaders/common/rng.wgsl
    let rng_path = repo_root().join("shaders/common/rng.wgsl");
    let rng_src = std::fs::read_to_string(&rng_path).expect("Read rng.wgsl");
    let rng_pipe = ComputePipelineHelper::create_compute_pipeline(
        &ctx.device,
        "rng_kernel",
        &rng_src,
        "cs_main",
    )
    .expect("Compile rng.wgsl");

    let count = 256usize;
    let out_bytes = (count * std::mem::size_of::<[f32; 2]>()) as u64;
    let rng_buf = GpuBufferHelper::create_storage_buffer(&ctx.device, "rng_out", out_bytes);
    let bgl = rng_pipe.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rng_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: rng_buf.as_entire_binding(),
        }],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&rng_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((count as u32) / 64, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));

    let samples: Vec<[f32; 2]> = pollster::block_on(GpuBufferHelper::readback_buffer(
        &ctx.device,
        &ctx.queue,
        &rng_buf,
        out_bytes,
    ))
    .expect("Readback rng");

    assert_eq!(samples.len(), count);
    for (i, [u0, u1]) in samples.iter().enumerate() {
        assert!(
            *u0 >= 0.0 && *u0 <= 1.0 && *u1 >= 0.0 && *u1 <= 1.0,
            "Sample {i} out of [0,1]: ({u0}, {u1})"
        );
    }
    assert_ne!(samples[0], samples[1]);
}

#[test]
fn test_gpu_henyey_greenstein_matches_cpu_reference() {
    let ctx = match pollster::block_on(GpuContext::init_headless()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping GPU HG test (no headless adapter): {e}");
            return;
        }
    };

    let hg_path = repo_root().join("shaders/volumetric/phase_henyey_greenstein.wgsl");
    let hg_src = std::fs::read_to_string(&hg_path).expect("Read phase_henyey_greenstein.wgsl");
    let hg_pipe = ComputePipelineHelper::create_compute_pipeline(
        &ctx.device,
        "hg_phase_kernel",
        &hg_src,
        "cs_hg_phase_process",
    )
    .expect("Compile phase_henyey_greenstein.wgsl");

    let mut queries = Vec::with_capacity(128);
    for i in 0..128 {
        let cos_q = -1.0 + 2.0 * (i as f32 / 127.0);
        let g = if i % 2 == 0 { 0.72 } else { -0.25 };
        queries.push(PhaseQueryInputGpu {
            incident_dir: [0.0, 0.0, 1.0],
            g,
            u_sample: [(i as f32 + 0.5) / 128.0, 0.37],
            cos_query: cos_q,
            padding: 0.0,
        });
    }

    let in_buf =
        GpuBufferHelper::create_storage_buffer_init(&ctx.device, "hg_in", &queries, false);
    let out_bytes = (queries.len() * std::mem::size_of::<[f32; 4]>()) as u64;
    let out_buf = GpuBufferHelper::create_storage_buffer(&ctx.device, "hg_out", out_bytes);

    let bgl = hg_pipe.get_bind_group_layout(0);
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hg_bg"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&hg_pipe);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(2, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));

    let gpu_out: Vec<[f32; 4]> = pollster::block_on(GpuBufferHelper::readback_buffer(
        &ctx.device,
        &ctx.queue,
        &out_buf,
        out_bytes,
    ))
    .expect("Readback HG output");

    for (i, q) in queries.iter().enumerate() {
        let [dx, dy, dz, gpu_p] = gpu_out[i];
        let dir_len = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (dir_len - 1.0).abs() < 1e-4,
            "Sampled direction {i} not unit length: {dir_len}"
        );
        let cpu_p = henyey_greenstein(q.cos_query, q.g);
        assert!(
            (gpu_p - cpu_p).abs() < 1e-4,
            "GPU vs CPU HG mismatch at {i}: gpu={gpu_p}, cpu={cpu_p}"
        );
    }
}
