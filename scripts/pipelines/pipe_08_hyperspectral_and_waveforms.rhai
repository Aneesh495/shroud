// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_08_hyperspectral_and_waveforms.rhai
// Domain Graph: hyperspectral_and_waveforms

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/hyperspectral/hyperspectral_unmixing_fcls.wgsl"), "WGSL valid: shaders/hyperspectral/hyperspectral_unmixing_fcls.wgsl");
    assert_true(validate_wgsl("shaders/hyperspectral/spectral_angle_mapper.wgsl"), "WGSL valid: shaders/hyperspectral/spectral_angle_mapper.wgsl");
    assert_true(validate_wgsl("shaders/hyperspectral/thermal_hyperspectral_emissivity_tes.wgsl"), "WGSL valid: shaders/hyperspectral/thermal_hyperspectral_emissivity_tes.wgsl");
    assert_true(validate_wgsl("shaders/hyperspectral/hyperspectral_anomaly_detection_rx.wgsl"), "WGSL valid: shaders/hyperspectral/hyperspectral_anomaly_detection_rx.wgsl");
    assert_true(validate_wgsl("shaders/lidar_waveforms/full_waveform_gaussian_decomposition.wgsl"), "WGSL valid: shaders/lidar_waveforms/full_waveform_gaussian_decomposition.wgsl");
    assert_true(validate_wgsl("shaders/lidar_waveforms/waveform_wiener_deconvolution.wgsl"), "WGSL valid: shaders/lidar_waveforms/waveform_wiener_deconvolution.wgsl");
    assert_true(validate_wgsl("shaders/lidar_waveforms/bathy_lidar_water_surface_refraction.wgsl"), "WGSL valid: shaders/lidar_waveforms/bathy_lidar_water_surface_refraction.wgsl");
    7
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_hyperspectral_and_waveforms");
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

#{"pipeline": "hyperspectral_and_waveforms", "kernels_validated": validated_count}
