// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_06_radar_and_metrology.rhai
// Domain Graph: radar_and_metrology

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/automotive_radars/fmcw_mimo_virtual_array.wgsl"), "WGSL valid: shaders/automotive_radars/fmcw_mimo_virtual_array.wgsl");
    assert_true(validate_wgsl("shaders/automotive_radars/cfar_ca_os_detector.wgsl"), "WGSL valid: shaders/automotive_radars/cfar_ca_os_detector.wgsl");
    assert_true(validate_wgsl("shaders/automotive_radars/micro_doppler_pedestrian.wgsl"), "WGSL valid: shaders/automotive_radars/micro_doppler_pedestrian.wgsl");
    assert_true(validate_wgsl("shaders/automotive_radars/radar_cross_section_physical_optics.wgsl"), "WGSL valid: shaders/automotive_radars/radar_cross_section_physical_optics.wgsl");
    assert_true(validate_wgsl("shaders/metrology_sensors/structured_light_phase_unwrapping.wgsl"), "WGSL valid: shaders/metrology_sensors/structured_light_phase_unwrapping.wgsl");
    assert_true(validate_wgsl("shaders/metrology_sensors/fizeau_interferometer_fringe.wgsl"), "WGSL valid: shaders/metrology_sensors/fizeau_interferometer_fringe.wgsl");
    assert_true(validate_wgsl("shaders/metrology_sensors/white_light_interferometry_vsi.wgsl"), "WGSL valid: shaders/metrology_sensors/white_light_interferometry_vsi.wgsl");
    7
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_radar_and_metrology");
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

#{"pipeline": "radar_and_metrology", "kernels_validated": validated_count}
