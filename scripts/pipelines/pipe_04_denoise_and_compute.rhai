// Shroud :: Multi-Kernel WebGPU Shader Graph Integrity Suite
// Script: scripts/pipelines/pipe_04_denoise_and_compute.rhai
// Domain Graph: denoise_and_compute

fn verify_domain_shader_graph() {
    assert_true(validate_wgsl("shaders/denoise/svgf_temporal_accumulator.wgsl"), "WGSL valid: shaders/denoise/svgf_temporal_accumulator.wgsl");
    assert_true(validate_wgsl("shaders/denoise/svgf_spatial_variance.wgsl"), "WGSL valid: shaders/denoise/svgf_spatial_variance.wgsl");
    assert_true(validate_wgsl("shaders/denoise/svgf_atrous_wavelet.wgsl"), "WGSL valid: shaders/denoise/svgf_atrous_wavelet.wgsl");
    assert_true(validate_wgsl("shaders/denoise/temporal_anti_aliasing.wgsl"), "WGSL valid: shaders/denoise/temporal_anti_aliasing.wgsl");
    assert_true(validate_wgsl("shaders/compute/workgroup_scan_blelloch.wgsl"), "WGSL valid: shaders/compute/workgroup_scan_blelloch.wgsl");
    assert_true(validate_wgsl("shaders/compute/matrix_gemm_tiled.wgsl"), "WGSL valid: shaders/compute/matrix_gemm_tiled.wgsl");
    assert_true(validate_wgsl("shaders/compute/fft_radix2_cooley_tukey.wgsl"), "WGSL valid: shaders/compute/fft_radix2_cooley_tukey.wgsl");
    7
}

fn verify_shared_radiometric_invariants() {
    let sc = scenario("graph_denoise_and_compute");
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

#{"pipeline": "denoise_and_compute", "kernels_validated": validated_count}
