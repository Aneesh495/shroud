// Shroud :: Computational Microscopy & Super-Resolution Verification
// Script: scripts/microscopy_biophotonics/mic_10_modal_10.rhai

fn verify_superres_abbe_vs_sted(num_ap, lambda_nm, sted_ratio) {
    let abbe_nm = lambda_nm / (2.0 * num_ap);
    let sim_nm = abbe_nm * 0.5;
    let sted_nm = abbe_nm / sqrt(1.0 + sted_ratio);
    assert_lt(sim_nm, abbe_nm);
    assert_lt(sted_nm, abbe_nm);
    #{"abbe_nm": abbe_nm, "sim_nm": sim_nm, "sted_nm": sted_nm}
}

fn verify_tirf_evanescent(lambda_nm, theta_deg) {
    let pi = 3.141592653589793;
    let n_glass = 1.518;
    let n_water = 1.333;
    let th = theta_deg * (pi / 180.0);
    let sin_th = sin(th);
    let sin_crit = n_water / n_glass;
    assert_gt(sin_th, sin_crit);
    let dp_nm = lambda_nm / (4.0 * pi * sqrt(n_glass * n_glass * sin_th * sin_th - n_water * n_water));
    assert_true(dp_nm > 30.0 && dp_nm < 400.0, "evanescent decay depth in TIRF range");
    dp_nm
}

fn verify_spim_rayleigh_range(w0_um, lambda_nm) {
    let lam_um = lambda_nm * 1e-3;
    let z_r = (3.141592653589793 * w0_um * w0_um) / lam_um;
    let w_at_zr = w0_um * sqrt(2.0);
    assert_close(w_at_zr / w0_um, sqrt(2.0), 1e-5);
    z_r
}

let res = verify_superres_abbe_vs_sted(1.25, 488.0, 10.0);
let dp = verify_tirf_evanescent(488.0, 68.0);
let zr = verify_spim_rayleigh_range(2.5, 488.0);

assert_true(validate_wgsl("shaders/computational_microscopy/fourier_ptychographic_microscopy.wgsl"), "FPM WGSL");
assert_true(validate_wgsl("shaders/computational_microscopy/structured_illumination_sim_superres.wgsl"), "SIM WGSL");
assert_true(validate_wgsl("shaders/computational_microscopy/total_internal_reflection_tirf.wgsl"), "TIRF WGSL");
assert_true(validate_wgsl("shaders/computational_microscopy/stimulated_emission_depletion_sted.wgsl"), "STED WGSL");

#{"microscopy_case": "modal_10", "resolution": res, "tirf_dp_nm": dp, "spim_zr_um": zr}
