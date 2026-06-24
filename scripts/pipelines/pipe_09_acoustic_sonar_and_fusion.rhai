// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_09_acoustic_sonar_and_fusion.rhai
// Domain Graph: acoustic_sonar_and_fusion

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/acoustic_sonar/sonar_ray_tracing_bellhop.wgsl"), "WGSL valid: shaders/acoustic_sonar/sonar_ray_tracing_bellhop.wgsl");
    assert_true(validate_wgsl("shaders/acoustic_sonar/sonar_transmission_loss_thorp.wgsl"), "WGSL valid: shaders/acoustic_sonar/sonar_transmission_loss_thorp.wgsl");
    assert_true(validate_wgsl("shaders/acoustic_sonar/synthetic_aperture_sonar_backprojection.wgsl"), "WGSL valid: shaders/acoustic_sonar/synthetic_aperture_sonar_backprojection.wgsl");
    assert_true(validate_wgsl("shaders/sensor_fusion/ekf_imu_lidar_state_propagation.wgsl"), "WGSL valid: shaders/sensor_fusion/ekf_imu_lidar_state_propagation.wgsl");
    assert_true(validate_wgsl("shaders/sensor_fusion/normal_distributions_transform_3d.wgsl"), "WGSL valid: shaders/sensor_fusion/normal_distributions_transform_3d.wgsl");
    assert_true(validate_wgsl("shaders/sensor_fusion/occupancy_grid_bayesian_mapping.wgsl"), "WGSL valid: shaders/sensor_fusion/occupancy_grid_bayesian_mapping.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_acoustic_sonar_and_fusion");
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

#{"pipeline": "acoustic_sonar_and_fusion", "kernels_validated": validated_count}
