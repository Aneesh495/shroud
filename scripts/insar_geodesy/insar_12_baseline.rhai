// Shroud :: InSAR Geodesy, PolSAR Decomposition & GMTI Verification
// Script: scripts/insar_geodesy/insar_12_pass_12.rhai

fn verify_insar_sensitivity(lam_m, b_perp, inc_deg) {
    let pi = 3.141592653589793;
    let inc = inc_deg * (pi / 180.0);
    let slant_range = 850000.0;
    // Height of ambiguity h_2pi = (lambda * R * sin(theta)) / (2 * B_perp)
    let h_amb = (lam_m * slant_range * sin(inc)) / (2.0 * b_perp);
    assert_gt(h_amb, 0.0);

    // Line-of-sight deformation sensitivity: one 2*pi fringe == lambda / 2
    let disp_per_fringe_mm = (lam_m * 0.5) * 1000.0;
    assert_gt(disp_per_fringe_mm, 1.0);
    #{"h_amb_m": h_amb, "fringe_mm": disp_per_fringe_mm}
}

fn verify_cloude_pottier_entropy() {
    let l1 = 0.72;
    let l2 = 0.20;
    let l3 = 0.08;
    let sum_l = l1 + l2 + l3;
    let p1 = l1 / sum_l;
    let p2 = l2 / sum_l;
    let p3 = l3 / sum_l;
    let h = -(p1 * ln(p1) + p2 * ln(p2) + p3 * ln(p3)) / ln(3.0);
    assert_true(h >= 0.0 && h <= 1.0, "scattering entropy bounded in [0, 1]");
    h
}

let insar = verify_insar_sensitivity(0.0555, 296.0, 41.0);
let entropy = verify_cloude_pottier_entropy();

assert_true(validate_wgsl("shaders/sar_interferometry/insar_complex_coherence.wgsl"), "InSAR coh WGSL");
assert_true(validate_wgsl("shaders/sar_interferometry/insar_topographic_phase_flatten.wgsl"), "InSAR topo WGSL");
assert_true(validate_wgsl("shaders/sar_interferometry/polsar_cloude_pottier_decomp.wgsl"), "Cloude-Pottier WGSL");
assert_true(validate_wgsl("shaders/sar_interferometry/sar_backprojection_gpu_kernel.wgsl"), "SAR BP WGSL");

#{"insar_case": "pass_12", "insar": insar, "entropy": entropy}
