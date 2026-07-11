// Shroud :: Stellar Astrophysics, Free-Space Comms & Seismology Verification
// Script: scripts/astrophysics_seismo/as_07_epoch_07.rhai

fn verify_transit_and_reddening(r_planet_star, r_v) {
    // Full transit occlusion depth equals (Rp / R*)^2 before limb darkening
    let depth = r_planet_star * r_planet_star;
    assert_gt(depth, 0.0);
    assert_lt(depth, 0.25);

    // CCM interstellar reddening at V-band (x = 1 / 0.55 um = 1.818 um^-1)
    let ebv = 0.45;
    let a_v = r_v * ebv;
    assert_gt(a_v, 0.0);
    #{"depth": depth, "a_v_mag": a_v}
}

fn verify_seismic_velocities(vp_m_s, vs_m_s) {
    // Stability condition in linear isotropic elasticity: Vp / Vs >= sqrt(2) (Poisson > 0)
    assert_gt(vp_m_s / vs_m_s, 1.41421);
    let c_rayleigh = 0.9194 * vs_m_s;
    assert_lt(c_rayleigh, vs_m_s);

    // Traveltime tomography slowness monotonicity
    let d_km = [10.0, 25.0, 60.0, 120.0];
    let prev_dt = 0.0;
    for dist in d_km {
        let dt_ps = (dist * 1000.0) * (1.0 / vs_m_s - 1.0 / vp_m_s);
        assert_gt(dt_ps, prev_dt);
        prev_dt = dt_ps;
    }
    #{"c_rayleigh": c_rayleigh, "max_s_p_delay_s": prev_dt}
}

fn verify_fso_scintillation_link() {
    let range_m = 12000.0;
    let div_urad = 25.0;
    let footprint_m = range_m * (div_urad * 1e-6);
    let rx_diam_m = 0.25;
    let geom_loss = (rx_diam_m * rx_diam_m) / (footprint_m * footprint_m);
    assert_true(geom_loss > 0.0 && geom_loss <= 1.0, "geometric coupling bounded");
    geom_loss
}

let astro = verify_transit_and_reddening(0.0990, 3.70);
let seismo = verify_seismic_velocities(4800.0, 2526.3);
let fso = verify_fso_scintillation_link();

assert_true(validate_wgsl("shaders/stellar_astrophysics/exoplanet_transit_light_curve_mandel_agol.wgsl"), "Transit WGSL");
assert_true(validate_wgsl("shaders/stellar_astrophysics/interstellar_reddening_cardelli.wgsl"), "CCM WGSL");
assert_true(validate_wgsl("shaders/geophysics_seismology/seismic_acoustic_wave_p_sv_split.wgsl"), "Seismic P/SV WGSL");
assert_true(validate_wgsl("shaders/optical_communication/free_space_optical_link_budget.wgsl"), "FSO WGSL");

#{"case": "epoch_07", "astro": astro, "seismo": seismo, "fso_geom": fso}
