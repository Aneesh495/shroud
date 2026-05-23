//! Compute pipeline construction and bind group binding.

use shroud_validator::validator::validate_wgsl_source;
use wgpu::{ComputePipeline, ComputePipelineDescriptor, Device, PipelineCompilationOptions, ShaderModuleDescriptor, ShaderSource};

pub struct ComputePipelineHelper;

impl ComputePipelineHelper {
    /// Compiles and creates a compute pipeline from WGSL source.
    pub fn create_compute_pipeline(
        device: &Device,
        label: &str,
        wgsl_source: &str,
        entry_point: &str,
    ) -> Result<ComputePipeline, String> {
        // Pre-validate with Naga to catch diagnostics with line numbers
        let val = validate_wgsl_source(label, wgsl_source);
        if !val.is_valid {
            return Err(format!("Shader validation error in {label}: {:?}", val.errors));
        }

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(wgsl_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: None, // Automatic layout derivation from reflection
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(pipeline)
    }

    /// Dispatches a verified 1D storage compute kernel and reads back the resulting `f32` buffer.
    pub async fn dispatch_storage_f32(
        device: &Device,
        queue: &wgpu::Queue,
        label: &str,
        wgsl_source: &str,
        entry_point: &str,
        element_count: usize,
    ) -> Result<Vec<f32>, String> {
        let pipeline = Self::create_compute_pipeline(device, label, wgsl_source, entry_point)?;
        let bytes = (element_count * std::mem::size_of::<f32>()) as u64;
        let storage = crate::buffers::GpuBufferHelper::create_storage_buffer(
            device,
            "compute_storage",
            bytes,
        );
        let bgl = pipeline.get_bind_group_layout(0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = ((element_count as u32) + 63) / 64;
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        crate::buffers::GpuBufferHelper::readback_buffer(device, queue, &storage, bytes).await
    }
}
