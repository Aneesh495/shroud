//! WGSL static validation using the Naga frontend and type checker.

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::path::Path;

#[derive(Debug)]
pub struct ValidationResult {
    pub file_path: String,
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub entry_points: Vec<String>,
}

/// Validates an HLSL source file using glslangValidator.
pub fn validate_hlsl_file(path: &Path, entry_point: &str, stage: &str) -> Result<ValidationResult, String> {
    let output = std::process::Command::new("glslangValidator")
        .arg("-D")
        .arg("-e")
        .arg(entry_point)
        .arg("-V")
        .arg("-S")
        .arg(stage)
        .arg("--hlsl-enable-16bit-types")
        .arg(path)
        .output()
        .map_err(|e| format!("Failed to invoke glslangValidator: {e}"))?;

    let is_valid = output.status.success();
    let mut errors = Vec::new();
    if !is_valid {
        errors.push(String::from_utf8_lossy(&output.stderr).to_string());
        let stdout_err = String::from_utf8_lossy(&output.stdout).to_string();
        if !stdout_err.is_empty() {
            errors.push(stdout_err);
        }
    }

    Ok(ValidationResult {
        file_path: path.display().to_string(),
        is_valid,
        errors,
        entry_points: vec![format!("{stage}:{entry_point}")],
    })
}

/// Validates an in-memory HLSL string by writing to a temporary file.
pub fn validate_hlsl_source(name: &str, source: &str, entry_point: &str, stage: &str) -> ValidationResult {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("shroud_{name}.hlsl"));
    if let Err(e) = std::fs::write(&tmp_path, source) {
        return ValidationResult {
            file_path: name.to_string(),
            is_valid: false,
            errors: vec![format!("Failed to write temporary HLSL file: {e}")],
            entry_points: Vec::new(),
        };
    }
    let res = match validate_hlsl_file(&tmp_path, entry_point, stage) {
        Ok(v) => v,
        Err(e) => ValidationResult {
            file_path: name.to_string(),
            is_valid: false,
            errors: vec![e],
            entry_points: Vec::new(),
        },
    };
    let _ = std::fs::remove_file(tmp_path);
    res
}

/// Validates a single WGSL source string.
pub fn validate_wgsl_source(name: &str, source: &str) -> ValidationResult {
    let mut errors = Vec::new();

    let module = match wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => {
            errors.push(format!("WGSL parse error: {}", e.emit_to_string(source)));
            return ValidationResult {
                file_path: name.to_string(),
                is_valid: false,
                errors,
                entry_points: Vec::new(),
            };
        }
    };

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let _info = match validator.validate(&module) {
        Ok(info) => info,
        Err(e) => {
            errors.push(format!("Naga validation error: {e:?}"));
            return ValidationResult {
                file_path: name.to_string(),
                is_valid: false,
                errors,
                entry_points: Vec::new(),
            };
        }
    };

    let entry_points = module
        .entry_points
        .iter()
        .map(|ep| format!("{:?}:{}", ep.stage, ep.name))
        .collect();

    ValidationResult {
        file_path: name.to_string(),
        is_valid: true,
        errors,
        entry_points,
    }
}

/// Recursively scans and validates all .wgsl files in a directory.
pub fn validate_wgsl_directory(dir: &Path) -> Vec<ValidationResult> {
    let mut results = Vec::new();
    if !dir.exists() {
        return results;
    }

    for entry in walkdir(dir) {
        if entry.extension().and_then(|s| s.to_str()) == Some("wgsl") {
            let path_str = entry.display().to_string();
            match std::fs::read_to_string(&entry) {
                Ok(source) => {
                    results.push(validate_wgsl_source(&path_str, &source));
                }
                Err(e) => {
                    results.push(ValidationResult {
                        file_path: path_str,
                        is_valid: false,
                        errors: vec![format!("Failed to read file: {e}")],
                        entry_points: Vec::new(),
                    });
                }
            }
        }
    }

    results
}

fn walkdir(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(walkdir(&path));
            } else {
                files.push(path);
            }
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_compute_shader() {
        let code = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;

        @compute @workgroup_size(64, 1, 1)
        fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&data)) {
                data[idx] = data[idx] * 2.0;
            }
        }
        "#;
        let res = validate_wgsl_source("test_cs", code);
        assert!(res.is_valid, "Validation failed: {:?}", res.errors);
        assert_eq!(res.entry_points.len(), 1);
    }

    #[test]
    fn test_syntax_error_detection() {
        let bad_code = r#"
        @compute @workgroup_size(64, 1, 1)
        fn cs_broken() {
            let missing_semicolon = 42
        }
        "#;
        let res = validate_wgsl_source("test_broken", bad_code);
        assert!(!res.is_valid);
        assert!(!res.errors.is_empty());
    }
}
