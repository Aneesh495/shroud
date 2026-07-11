// Shroud :: Radiance Cache, ReSTIR Resampling & Variance Reduction Verification
// Script: scripts/radiance_fields/rad_07_cache_07.rhai

fn verify_mis_power_heuristic(pdf_brdf, pdf_light) {
    // Veach (1995) balance & power heuristic weights (beta = 2)
    let w_bal = pdf_brdf / (pdf_brdf + pdf_light);
    let w_bal_comp = pdf_light / (pdf_brdf + pdf_light);
    assert_close(w_bal + w_bal_comp, 1.0, 1e-6);

    let b2 = pdf_brdf * pdf_brdf;
    let l2 = pdf_light * pdf_light;
    let w_pow = b2 / (b2 + l2);
    let w_pow_comp = l2 / (b2 + l2);
    assert_close(w_pow + w_pow_comp, 1.0, 1e-6);
    w_pow
}

fn verify_spherical_gaussian_lobe(sharpness_lam, dir_a, dir_b) {
    let na = dir_a.normalize();
    let nb = dir_b.normalize();
    let cos_t = na.dot(nb);
    let val = exp(sharpness_lam * (cos_t - 1.0));
    assert_true(val > 0.0 && val <= 1.0, "SG lobe amplitude bounded");
    val
}

fn verify_mc_variance_convergence(samples) {
    let inv_sqrt_n = 1.0 / sqrt(samples);
    let baseline = 1.0 / sqrt(1.0);
    assert_lt(inv_sqrt_n, baseline);
    inv_sqrt_n
}

let mis = verify_mis_power_heuristic(0.65, 0.35);
let sg = verify_spherical_gaussian_lobe(21.5, vec3(0.0, 0.0, 1.0), vec3(0.25, 0.15, 0.95));
let std_err = verify_mc_variance_convergence(128.0);

let sc = scenario("radiance_cache_07");
sc.set_media(0.002, 0.0430, 0.76);
assert_gt(sc.albedo(), 0.80);

assert_true(validate_wgsl("shaders/restir/reservoir_structures.wgsl"), "ReSTIR struct WGSL");
assert_true(validate_wgsl("shaders/restir/restir_gi_temporal.wgsl"), "ReSTIR GI WGSL");
assert_true(validate_wgsl("shaders/denoise/svgf_atrous_wavelet.wgsl"), "SVGF atrous WGSL");
assert_true(validate_wgsl("shaders/monte_carlo_samplers/mc_multiple_importance_sampling_balance.wgsl"), "MIS WGSL");

#{"radiance_case": "cache_07", "mis_w": mis, "sg_val": sg, "std_err": std_err}
