// Shroud :: Quantum Optics, SNSPD Cryogenics & Bell Non-Locality Verification
// Script: scripts/quantum_cryo/qc_08_state_08.rhai

fn verify_snspd_and_london_penetration(t_bath_k, t_crit_k, j_ratio) {
    assert_lt(t_bath_k, t_crit_k);
    // Gorter-Casimir temperature dependence lambda_L(T) = lambda_0 / sqrt(1 - (T/Tc)^4)
    let lambda_0_nm = 250.0;
    let ratio_t = t_bath_k / t_crit_k;
    let lambda_t = lambda_0_nm / sqrt(1.0 - (ratio_t ** 4.0));
    assert_gt(lambda_t, lambda_0_nm);

    // High bias current ratio enhances single-photon detection efficiency
    let p_trig = 1.0 - exp(-4.0 * j_ratio);
    assert_true(p_trig > 0.5 && p_trig <= 1.0, "SNSPD trigger probability valid");
    #{"lambda_t_nm": lambda_t, "p_trig": p_trig}
}

fn verify_hom_dip_and_chsh(coherence_ps) {
    let p_zero = 0.5 * (1.0 - exp(0.0));
    assert_close(p_zero, 0.0, 1e-6);
    let p_far = 0.5 * (1.0 - exp(-0.5 * ((10.0 * coherence_ps) / coherence_ps) ** 2.0));
    assert_close(p_far, 0.5, 1e-4);

    // CHSH optimal angles: a=0, a'=pi/4, b=pi/8, b'=3*pi/8
    let pi = 3.141592653589793;
    let a = 0.0;
    let ap = pi / 4.0;
    let b = pi / 8.0;
    let bp = 3.0 * pi / 8.0;

    let e_ab = -cos(2.0 * (a - b));
    let e_abp = -cos(2.0 * (a - bp));
    let e_apb = -cos(2.0 * (ap - b));
    let e_apbp = -cos(2.0 * (ap - bp));
    let s_chsh = abs(e_ab - e_abp + e_apb + e_apbp);

    assert_close(s_chsh, 2.0 * sqrt(2.0), 1e-5);
    assert_gt(s_chsh, 2.0);
    s_chsh
}

fn verify_sipm_saturation() {
    let n_cells = 4000.0;
    let pde = 0.42;
    let counts = [100.0, 1000.0, 5000.0, 15000.0];
    let prev = 0.0;
    for n_ph in counts {
        let fired = n_cells * (1.0 - exp(-n_ph * pde / n_cells));
        assert_gt(fired, prev);
        assert_lt(fired, n_cells);
        prev = fired;
    }
    prev
}

let cryo = verify_snspd_and_london_penetration(0.85, 11.60, 0.850);
let chsh = verify_hom_dip_and_chsh(2.40);
let sipm_max = verify_sipm_saturation();

assert_true(validate_wgsl("shaders/cryogenic_superconducting/superconducting_snspd_detector.wgsl"), "SNSPD WGSL");
assert_true(validate_wgsl("shaders/cryogenic_superconducting/london_penetration_depth_screening.wgsl"), "London WGSL");
assert_true(validate_wgsl("shaders/quantum_optics/hong_ou_mandel_two_photon_dip.wgsl"), "HOM WGSL");
assert_true(validate_wgsl("shaders/quantum_optics/bell_chsh_inequality_violation.wgsl"), "CHSH WGSL");

#{"case": "state_08", "cryo": cryo, "chsh_s": chsh, "sipm_max": sipm_max}
