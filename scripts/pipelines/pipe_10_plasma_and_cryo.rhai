// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_10_plasma_and_cryo.rhai
// Domain Graph: plasma_and_cryo

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/plasma_aerodynamics/hypersonic_plasma_sheath_attenuation.wgsl"), "WGSL valid: shaders/plasma_aerodynamics/hypersonic_plasma_sheath_attenuation.wgsl");
    assert_true(validate_wgsl("shaders/plasma_aerodynamics/ionization_saha_equilibrium.wgsl"), "WGSL valid: shaders/plasma_aerodynamics/ionization_saha_equilibrium.wgsl");
    assert_true(validate_wgsl("shaders/plasma_aerodynamics/mhd_hartmann_flow_channel.wgsl"), "WGSL valid: shaders/plasma_aerodynamics/mhd_hartmann_flow_channel.wgsl");
    assert_true(validate_wgsl("shaders/cryogenic_superconducting/superconducting_snspd_detector.wgsl"), "WGSL valid: shaders/cryogenic_superconducting/superconducting_snspd_detector.wgsl");
    assert_true(validate_wgsl("shaders/cryogenic_superconducting/josephson_junction_rscj.wgsl"), "WGSL valid: shaders/cryogenic_superconducting/josephson_junction_rscj.wgsl");
    assert_true(validate_wgsl("shaders/cryogenic_superconducting/squid_magnetometer_flux_voltage.wgsl"), "WGSL valid: shaders/cryogenic_superconducting/squid_magnetometer_flux_voltage.wgsl");
    6
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_plasma_and_cryo");
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

#{"pipeline": "plasma_and_cryo", "kernels_validated": validated_count}
