// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_01_common_and_bvh.rhai
// Domain Graph: common_and_bvh

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/common/constants.wgsl"), "WGSL valid: shaders/common/constants.wgsl");
    assert_true(validate_wgsl("shaders/common/types.wgsl"), "WGSL valid: shaders/common/types.wgsl");
    assert_true(validate_wgsl("shaders/common/math.wgsl"), "WGSL valid: shaders/common/math.wgsl");
    assert_true(validate_wgsl("shaders/common/rng.wgsl"), "WGSL valid: shaders/common/rng.wgsl");
    assert_true(validate_wgsl("shaders/common/colorimetry.wgsl"), "WGSL valid: shaders/common/colorimetry.wgsl");
    assert_true(validate_wgsl("shaders/bvh/lbvh_morton.wgsl"), "WGSL valid: shaders/bvh/lbvh_morton.wgsl");
    assert_true(validate_wgsl("shaders/bvh/karras_tree_builder.wgsl"), "WGSL valid: shaders/bvh/karras_tree_builder.wgsl");
    assert_true(validate_wgsl("shaders/bvh/bvh_traversal_short_stack.wgsl"), "WGSL valid: shaders/bvh/bvh_traversal_short_stack.wgsl");
    assert_true(validate_wgsl("shaders/bvh/ray_primitive_intersection.wgsl"), "WGSL valid: shaders/bvh/ray_primitive_intersection.wgsl");
    9
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_common_and_bvh");
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

#{"pipeline": "common_and_bvh", "kernels_validated": validated_count}
