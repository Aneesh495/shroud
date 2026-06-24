// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_07_ocean_sky_and_quantum_detectors.rhai
// Domain Graph: ocean_sky_and_quantum_detectors

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/oceanic_hydrology/petzold_ocean_scattering.wgsl"), "WGSL valid: shaders/oceanic_hydrology/petzold_ocean_scattering.wgsl");
    assert_true(validate_wgsl("shaders/oceanic_hydrology/raman_scattering_water.wgsl"), "WGSL valid: shaders/oceanic_hydrology/raman_scattering_water.wgsl");
    assert_true(validate_wgsl("shaders/oceanic_hydrology/sea_surface_cox_munk.wgsl"), "WGSL valid: shaders/oceanic_hydrology/sea_surface_cox_munk.wgsl");
    assert_true(validate_wgsl("shaders/astronomy_sky/bruneton_atmosphere_transmittance.wgsl"), "WGSL valid: shaders/astronomy_sky/bruneton_atmosphere_transmittance.wgsl");
    assert_true(validate_wgsl("shaders/astronomy_sky/bruneton_atmosphere_inscatter.wgsl"), "WGSL valid: shaders/astronomy_sky/bruneton_atmosphere_inscatter.wgsl");
    assert_true(validate_wgsl("shaders/quantum_photodetectors/spad_avalanche_breakdown.wgsl"), "WGSL valid: shaders/quantum_photodetectors/spad_avalanche_breakdown.wgsl");
    assert_true(validate_wgsl("shaders/quantum_photodetectors/sipm_microcell_crosstalk.wgsl"), "WGSL valid: shaders/quantum_photodetectors/sipm_microcell_crosstalk.wgsl");
    7
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_ocean_sky_and_quantum_detectors");
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

#{"pipeline": "ocean_sky_and_quantum_detectors", "kernels_validated": validated_count}
