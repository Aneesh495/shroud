// Shroud :: Nonlinear Optics & Coherent Fiber Communications Verification
// Script: scripts/nonlinear_optical_comms/nlc_15_link_15.rhai

fn verify_kerr_b_integral(p_peak_w, eff_len_m) {
    let n2 = 2.6e-20; // fused silica m^2 / W
    let a_eff = 80.0e-12; // 80 um^2 effective mode area
    let intensity = p_peak_w / a_eff;
    let lambda_m = 1.55e-6;
    let b_int = (2.0 * 3.141592653589793 / lambda_m) * n2 * intensity * eff_len_m;
    assert_gt(b_int, 0.0);
    b_int
}

fn verify_edfa_ase_noise(gain_db, nf_db) {
    let g = 10.0 ** (gain_db * 0.1);
    let nf = 10.0 ** (nf_db * 0.1);
    let n_sp = 0.5 * nf;
    assert_gt(g, 1.0);
    assert_gt(n_sp, 0.9);
    let h = 6.62607015e-34;
    let c = speed_of_light();
    let nu = c / 1.55e-6;
    let bw_hz = 50.0e9;
    let p_ase_mw = 2.0 * n_sp * (g - 1.0) * h * nu * bw_hz * 1000.0;
    assert_gt(p_ase_mw, 0.0);
    p_ase_mw
}

fn verify_pmd_accumulation(len_km) {
    let d_pmd = 0.05; // ps / sqrt(km)
    let mean_dgd = d_pmd * sqrt(len_km);
    assert_gt(mean_dgd, 0.0);
    mean_dgd
}

let b_val = verify_kerr_b_integral(230.0, 12.0);
let ase = verify_edfa_ase_noise(21.5, 4.5);
let dgd = verify_pmd_accumulation(235.0);

assert_true(validate_wgsl("shaders/nonlinear_optics/second_harmonic_generation_shg.wgsl"), "SHG WGSL");
assert_true(validate_wgsl("shaders/nonlinear_optics/optical_kerr_self_focusing.wgsl"), "Kerr WGSL");
assert_true(validate_wgsl("shaders/optical_communication/edfa_spontaneous_emission_noise.wgsl"), "EDFA WGSL");
assert_true(validate_wgsl("shaders/optical_communication/polarization_mode_dispersion_pmd.wgsl"), "PMD WGSL");

#{"case": "link_15", "b_integral": b_val, "ase_mw": ase, "dgd_ps": dgd}
