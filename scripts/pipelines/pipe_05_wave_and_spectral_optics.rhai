// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_05_wave_and_spectral_optics.rhai
// Domain Graph: wave_and_spectral_optics

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/wave_optics/airy_disk_psf.wgsl"), "WGSL valid: shaders/wave_optics/airy_disk_psf.wgsl");
    assert_true(validate_wgsl("shaders/wave_optics/fresnel_diffraction_czt.wgsl"), "WGSL valid: shaders/wave_optics/fresnel_diffraction_czt.wgsl");
    assert_true(validate_wgsl("shaders/wave_optics/fraunhofer_farfield_fft.wgsl"), "WGSL valid: shaders/wave_optics/fraunhofer_farfield_fft.wgsl");
    assert_true(validate_wgsl("shaders/wave_optics/polarization_stokes_mueller.wgsl"), "WGSL valid: shaders/wave_optics/polarization_stokes_mueller.wgsl");
    assert_true(validate_wgsl("shaders/wave_optics/birefringence_uniaxial.wgsl"), "WGSL valid: shaders/wave_optics/birefringence_uniaxial.wgsl");
    assert_true(validate_wgsl("shaders/wave_optics/thin_film_transfer_matrix.wgsl"), "WGSL valid: shaders/wave_optics/thin_film_transfer_matrix.wgsl");
    assert_true(validate_wgsl("shaders/spectral_optics/dispersion_chromatic_aberration.wgsl"), "WGSL valid: shaders/spectral_optics/dispersion_chromatic_aberration.wgsl");
    assert_true(validate_wgsl("shaders/spectral_optics/grating_diffraction_rcwa.wgsl"), "WGSL valid: shaders/spectral_optics/grating_diffraction_rcwa.wgsl");
    8
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_wave_and_spectral_optics");
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

#{"pipeline": "wave_and_spectral_optics", "kernels_validated": validated_count}
