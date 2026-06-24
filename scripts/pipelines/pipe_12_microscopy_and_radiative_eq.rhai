// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_12_microscopy_and_radiative_eq.rhai
// Domain Graph: microscopy_and_radiative_eq

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/computational_microscopy/fourier_ptychographic_microscopy.wgsl"), "WGSL valid: shaders/computational_microscopy/fourier_ptychographic_microscopy.wgsl");
    assert_true(validate_wgsl("shaders/computational_microscopy/optical_coherence_tomography_oct.wgsl"), "WGSL valid: shaders/computational_microscopy/optical_coherence_tomography_oct.wgsl");
    assert_true(validate_wgsl("shaders/computational_microscopy/stimulated_emission_depletion_sted.wgsl"), "WGSL valid: shaders/computational_microscopy/stimulated_emission_depletion_sted.wgsl");
    assert_true(validate_wgsl("shaders/atmospheric_radiative_equilibrium/line_by_line_hitran_voigt_profile.wgsl"), "WGSL valid: shaders/atmospheric_radiative_equilibrium/line_by_line_hitran_voigt_profile.wgsl");
    assert_true(validate_wgsl("shaders/atmospheric_radiative_equilibrium/two_stream_eddington_approximation.wgsl"), "WGSL valid: shaders/atmospheric_radiative_equilibrium/two_stream_eddington_approximation.wgsl");
    assert_true(validate_wgsl("shaders/atmospheric_radiative_equilibrium/greenhouse_gas_radiative_forcing.wgsl"), "WGSL valid: shaders/atmospheric_radiative_equilibrium/greenhouse_gas_radiative_forcing.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_microscopy_and_radiative_eq");
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

#{"pipeline": "microscopy_and_radiative_eq", "kernels_validated": validated_count}
