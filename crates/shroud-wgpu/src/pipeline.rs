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
}
