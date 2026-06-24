// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_14_adaptive_and_photonic_ics.rhai
// Domain Graph: adaptive_and_photonic_ics

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/adaptive_optics/shack_hartmann_wavefront_sensor.wgsl"), "WGSL valid: shaders/adaptive_optics/shack_hartmann_wavefront_sensor.wgsl");
    assert_true(validate_wgsl("shaders/adaptive_optics/kolmogorov_phase_screen.wgsl"), "WGSL valid: shaders/adaptive_optics/kolmogorov_phase_screen.wgsl");
    assert_true(validate_wgsl("shaders/adaptive_optics/deformable_mirror_influence.wgsl"), "WGSL valid: shaders/adaptive_optics/deformable_mirror_influence.wgsl");
    assert_true(validate_wgsl("shaders/photonic_integrated_circuits/silicon_rib_waveguide_mode.wgsl"), "WGSL valid: shaders/photonic_integrated_circuits/silicon_rib_waveguide_mode.wgsl");
    assert_true(validate_wgsl("shaders/photonic_integrated_circuits/microring_resonator_transfer.wgsl"), "WGSL valid: shaders/photonic_integrated_circuits/microring_resonator_transfer.wgsl");
    assert_true(validate_wgsl("shaders/photonic_integrated_circuits/mach_zehnder_interferometer_mesh.wgsl"), "WGSL valid: shaders/photonic_integrated_circuits/mach_zehnder_interferometer_mesh.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_adaptive_and_photonic_ics");
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

#{"pipeline": "adaptive_and_photonic_ics", "kernels_validated": validated_count}
