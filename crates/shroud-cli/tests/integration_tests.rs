//! Workspace-level integration tests verifying full WGSL corpus, reflection, and Rhai execution.

use shroud_rhai::ShroudRhaiEngine;
use shroud_validator::reflection::reflect_wgsl;
use shroud_validator::validate_wgsl_directory;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_all_repo_wgsl_shaders_compile_under_naga() {
    let shaders_dir = repo_root().join("shaders");
    let results = validate_wgsl_directory(&shaders_dir);
    assert!(
        results.len() >= 380,
        "Expected >= 380 WGSL kernels, found {}",
        results.len()
    );
    let failures: Vec<_> = results.iter().filter(|r| !r.is_valid).collect();
    assert!(
        failures.is_empty(),
        "Found {} failing WGSL shaders: {:?}",
        failures.len(),
        failures
    );
}

#[test]
fn test_shader_reflection_bindings_on_real_kernels() {
    let root = repo_root();
    let hg_src =
        std::fs::read_to_string(root.join("shaders/volumetric/phase_henyey_greenstein.wgsl"))
            .unwrap();
    let refl = reflect_wgsl(&hg_src).expect("Reflect HG kernel");
    assert_eq!(refl.entry_points, vec!["cs_hg_phase_process"]);
    assert_eq!(refl.bindings.len(), 2);
    assert_eq!(refl.bindings[0].group, 0);
    assert_eq!(refl.bindings[0].binding, 0);
    assert_eq!(refl.bindings[1].group, 0);
    assert_eq!(refl.bindings[1].binding, 1);
}

#[test]
fn test_all_repo_rhai_scenarios_pass() {
    let scripts_dir = repo_root().join("scripts");
    let engine = ShroudRhaiEngine::new();
    let results = engine.run_directory(&scripts_dir);
    assert!(
        results.len() >= 400,
        "Expected >= 400 Rhai scripts, found {}",
        results.len()
    );
    let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
    assert!(
        failures.is_empty(),
        "Found {} failing Rhai scripts: {:?}",
        failures.len(),
        failures
    );
}
