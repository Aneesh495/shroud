// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_02_materials_and_bsdf.rhai
// Domain Graph: materials_and_bsdf

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/materials/microfacet_ggx.wgsl"), "WGSL valid: shaders/materials/microfacet_ggx.wgsl");
    assert_true(validate_wgsl("shaders/materials/disney_principled.wgsl"), "WGSL valid: shaders/materials/disney_principled.wgsl");
    assert_true(validate_wgsl("shaders/materials/dielectric_transmission.wgsl"), "WGSL valid: shaders/materials/dielectric_transmission.wgsl");
    assert_true(validate_wgsl("shaders/materials/conductor_fresnel.wgsl"), "WGSL valid: shaders/materials/conductor_fresnel.wgsl");
    assert_true(validate_wgsl("shaders/materials/retroreflective_microprism.wgsl"), "WGSL valid: shaders/materials/retroreflective_microprism.wgsl");
    assert_true(validate_wgsl("shaders/materials/bssrdf_dipole_diffusion.wgsl"), "WGSL valid: shaders/materials/bssrdf_dipole_diffusion.wgsl");
    assert_true(validate_wgsl("shaders/materials/thin_film_interference.wgsl"), "WGSL valid: shaders/materials/thin_film_interference.wgsl");
    assert_true(validate_wgsl("shaders/materials/anisotropic_cloth.wgsl"), "WGSL valid: shaders/materials/anisotropic_cloth.wgsl");
    8
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_materials_and_bsdf");
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

#{"pipeline": "materials_and_bsdf", "kernels_validated": validated_count}
