// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_03_volumetric_and_restir.rhai
// Domain Graph: volumetric_and_restir

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/volumetric/phase_henyey_greenstein.wgsl"), "WGSL valid: shaders/volumetric/phase_henyey_greenstein.wgsl");
    assert_true(validate_wgsl("shaders/volumetric/delta_tracking_woodcock.wgsl"), "WGSL valid: shaders/volumetric/delta_tracking_woodcock.wgsl");
    assert_true(validate_wgsl("shaders/volumetric/ratio_tracking_transmittance.wgsl"), "WGSL valid: shaders/volumetric/ratio_tracking_transmittance.wgsl");
    assert_true(validate_wgsl("shaders/volumetric/spectral_hero_tracking.wgsl"), "WGSL valid: shaders/volumetric/spectral_hero_tracking.wgsl");
    assert_true(validate_wgsl("shaders/restir/reservoir_structures.wgsl"), "WGSL valid: shaders/restir/reservoir_structures.wgsl");
    assert_true(validate_wgsl("shaders/restir/restir_di_candidate.wgsl"), "WGSL valid: shaders/restir/restir_di_candidate.wgsl");
    assert_true(validate_wgsl("shaders/restir/restir_gi_temporal.wgsl"), "WGSL valid: shaders/restir/restir_gi_temporal.wgsl");
    assert_true(validate_wgsl("shaders/restir/visibility_ray_check.wgsl"), "WGSL valid: shaders/restir/visibility_ray_check.wgsl");
    8
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_volumetric_and_restir");
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

#{"pipeline": "volumetric_and_restir", "kernels_validated": validated_count}
