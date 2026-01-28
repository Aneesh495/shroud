//! Naga AST reflection for WGSL uniform buffers, storage bindings, and push constants.

use naga::front::wgsl;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingReflection {
    pub name: String,
    pub group: u32,
    pub binding: u32,
    pub space: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShaderReflection {
    pub entry_points: Vec<String>,
    pub bindings: Vec<BindingReflection>,
}

pub fn reflect_wgsl(source: &str) -> Result<ShaderReflection, String> {
    let module = wgsl::parse_str(source).map_err(|e| format!("Parse error: {e}"))?;

    let mut bindings = Vec::new();

    for (_handle, var) in module.global_variables.iter() {
        if let Some(ref binding) = var.binding {
            let name = var.name.clone().unwrap_or_else(|| "unnamed".to_string());
            let space = format!("{:?}", var.space);
            bindings.push(BindingReflection {
                name,
                group: binding.group,
                binding: binding.binding,
                space,
            });
        }
    }

    let entry_points = module
        .entry_points
        .iter()
        .map(|ep| ep.name.clone())
        .collect();

    Ok(ShaderReflection {
        entry_points,
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_bindings() {
        let code = r#"
        struct Uniforms {
            camera_pos: vec3<f32>,
            time: f32,
        };

        @group(0) @binding(0) var<uniform> u_data: Uniforms;
        @group(0) @binding(1) var<storage, read> s_in: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> s_out: array<vec4<f32>>;

        @compute @workgroup_size(32, 1, 1)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            s_out[gid.x] = s_in[gid.x] + vec4<f32>(u_data.camera_pos, u_data.time);
        }
        "#;
        let refl = reflect_wgsl(code).expect("Reflection failed");
        assert_eq!(refl.entry_points, vec!["main"]);
        assert_eq!(refl.bindings.len(), 3);
        assert_eq!(refl.bindings[0].name, "u_data");
        assert_eq!(refl.bindings[0].group, 0);
        assert_eq!(refl.bindings[0].binding, 0);
    }
}
