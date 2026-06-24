// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_15_nonlinear_slam_and_sar.rhai
// Domain Graph: nonlinear_slam_and_sar

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/nonlinear_optics/second_harmonic_generation_shg.wgsl"), "WGSL valid: shaders/nonlinear_optics/second_harmonic_generation_shg.wgsl");
    assert_true(validate_wgsl("shaders/nonlinear_optics/optical_kerr_self_focusing.wgsl"), "WGSL valid: shaders/nonlinear_optics/optical_kerr_self_focusing.wgsl");
    assert_true(validate_wgsl("shaders/lidar_slam/loam_feature_curvature_extraction.wgsl"), "WGSL valid: shaders/lidar_slam/loam_feature_curvature_extraction.wgsl");
    assert_true(validate_wgsl("shaders/lidar_slam/deskew_motion_distortion_slerp.wgsl"), "WGSL valid: shaders/lidar_slam/deskew_motion_distortion_slerp.wgsl");
    assert_true(validate_wgsl("shaders/sar_interferometry/insar_complex_coherence.wgsl"), "WGSL valid: shaders/sar_interferometry/insar_complex_coherence.wgsl");
    assert_true(validate_wgsl("shaders/sar_interferometry/sar_backprojection_gpu_kernel.wgsl"), "WGSL valid: shaders/sar_interferometry/sar_backprojection_gpu_kernel.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_nonlinear_slam_and_sar");
    sc.set_media(0.004, 0.036, 0.75);
    assert_close(sc.extinction(), 0.040, 1e-6);
    assert_close(sc.albedo(), 0.90, 1e-6);

    let r = ray(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
    let hit = r.intersect_box(vec3(-5.0, -5.0, 25.0), vec3(5.0, 5.0, 35.0));
    assert_close(hit, 25.0, 1e-5);

    let pt = r.at(hit);
    assert_close(pt.z, 25.0, 1e-5);

    let tau = beer_lambert(sc.extinction(), hit);
    assert_true(tau > 0.0 && tau < 1.0, "transmittance strictly bounded");
    true
}

let validated_count = verify_domain_shader_graph();
assert_true(validated_count >= 1, "at least one kernel validated");
assert_true(verify_shared_radiometric_invariants(), "invariants passed");

#{"pipeline": "nonlinear_slam_and_sar", "kernels_validated": validated_count}
