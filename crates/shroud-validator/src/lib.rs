//! Static shader validator and reflection inspector for Shroud.

pub mod validator;
pub mod reflection;

pub use validator::{validate_hlsl_file, validate_hlsl_source, validate_wgsl_source, validate_wgsl_directory, ValidationResult};
