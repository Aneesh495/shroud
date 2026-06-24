// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_11_geo_and_quantum_optics.rhai
// Domain Graph: geo_and_quantum_optics

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/geophysics_seismology/seismic_acoustic_wave_p_sv_split.wgsl"), "WGSL valid: shaders/geophysics_seismology/seismic_acoustic_wave_p_sv_split.wgsl");
    assert_true(validate_wgsl("shaders/geophysics_seismology/full_waveform_inversion_adjoint.wgsl"), "WGSL valid: shaders/geophysics_seismology/full_waveform_inversion_adjoint.wgsl");
    assert_true(validate_wgsl("shaders/geophysics_seismology/traveltime_tomography_fast_marching.wgsl"), "WGSL valid: shaders/geophysics_seismology/traveltime_tomography_fast_marching.wgsl");
    assert_true(validate_wgsl("shaders/quantum_optics/hong_ou_mandel_two_photon_dip.wgsl"), "WGSL valid: shaders/quantum_optics/hong_ou_mandel_two_photon_dip.wgsl");
    assert_true(validate_wgsl("shaders/quantum_optics/wigner_quasi_probability_distribution.wgsl"), "WGSL valid: shaders/quantum_optics/wigner_quasi_probability_distribution.wgsl");
    assert_true(validate_wgsl("shaders/quantum_optics/bell_chsh_inequality_violation.wgsl"), "WGSL valid: shaders/quantum_optics/bell_chsh_inequality_violation.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_geo_and_quantum_optics");
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

#{"pipeline": "geo_and_quantum_optics", "kernels_validated": validated_count}
