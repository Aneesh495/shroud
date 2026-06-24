// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_13_astrophysics_and_optical_comms.rhai
// Domain Graph: astrophysics_and_optical_comms

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/stellar_astrophysics/stellar_atmosphere_kurucz_atlas.wgsl"), "WGSL valid: shaders/stellar_astrophysics/stellar_atmosphere_kurucz_atlas.wgsl");
    assert_true(validate_wgsl("shaders/stellar_astrophysics/exoplanet_transit_light_curve_mandel_agol.wgsl"), "WGSL valid: shaders/stellar_astrophysics/exoplanet_transit_light_curve_mandel_agol.wgsl");
    assert_true(validate_wgsl("shaders/stellar_astrophysics/accretion_disk_shakura_sunyaev.wgsl"), "WGSL valid: shaders/stellar_astrophysics/accretion_disk_shakura_sunyaev.wgsl");
    assert_true(validate_wgsl("shaders/optical_communication/free_space_optical_link_budget.wgsl"), "WGSL valid: shaders/optical_communication/free_space_optical_link_budget.wgsl");
    assert_true(validate_wgsl("shaders/optical_communication/fiber_chromatic_dispersion_nlse.wgsl"), "WGSL valid: shaders/optical_communication/fiber_chromatic_dispersion_nlse.wgsl");
    assert_true(validate_wgsl("shaders/optical_communication/fso_gamma_gamma_atmospheric_turbulence.wgsl"), "WGSL valid: shaders/optical_communication/fso_gamma_gamma_atmospheric_turbulence.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_astrophysics_and_optical_comms");
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

#{"pipeline": "astrophysics_and_optical_comms", "kernels_validated": validated_count}
