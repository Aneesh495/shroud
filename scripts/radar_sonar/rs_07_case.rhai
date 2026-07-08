// Shroud :: 4D MIMO Radar, CFAR Detection & Acoustic Sonar Verification
// Script: scripts/radar_sonar/rs_07_array_07.rhai

fn verify_mimo_fmcw_resolution(bw_hz, f_center_hz) {
    let c = speed_of_light();
    let range_res_m = c / (2.0 * bw_hz);
    let lambda_m = c / f_center_hz;
    assert_gt(range_res_m, 0.0);
    assert_gt(lambda_m, 0.0);

    let radar = fmcw_config(f_center_hz, bw_hz, 40.0e-6, 0.15);
    let targets = [[18.5, -12.0], [64.0, 0.0], [112.5, 24.0], [175.0, -30.5]];
    for tgt in targets {
        let fu = radar.beat_up(tgt[0], tgt[1]);
        let fd = radar.beat_down(tgt[0], tgt[1]);
        assert_close(radar.solve_range(fu, fd), tgt[0], 0.05);
        assert_close(radar.solve_velocity(fu, fd), tgt[1], 0.05);
    }
    range_res_m
}

fn verify_ca_cfar_threshold(n_ref, p_fa) {
    // Exact CA-CFAR threshold factor alpha = N * (P_fa^(-1/N) - 1)
    let alpha = n_ref * ((p_fa ** (-1.0 / n_ref)) - 1.0);
    assert_gt(alpha, 1.0);
    let noise_floor = 1.5e-6;
    let threshold = alpha * noise_floor;
    assert_gt(threshold, noise_floor);
    alpha
}

fn verify_thorp_ocean_attenuation(f_khz, sal_psu) {
    let f2 = f_khz * f_khz;
    let boric = (0.11 * f2) / (1.0 + f2);
    let mgso4 = (44.0 * f2) / (4100.0 + f2) * (sal_psu / 35.0);
    let pure = 2.75e-4 * f2 + 0.003;
    let alpha_db_km = boric + mgso4 + pure;
    assert_gt(alpha_db_km, 0.0);

    let ranges = [100.0, 500.0, 1500.0, 4000.0];
    let prev_tl = 0.0;
    for r in ranges {
        let tl = 20.0 * (ln(r) / ln(10.0)) + alpha_db_km * (r * 1e-3);
        assert_gt(tl, prev_tl);
        prev_tl = tl;
    }
    #{"alpha_db_km": alpha_db_km, "tl_4km": prev_tl}
}

fn verify_janus_adcp_inversion(beam_deg) {
    let th = beam_deg * (3.141592653589793 / 180.0);
    let u_true = 1.45;
    let v_true = -0.85;
    let w_true = 0.12;
    let v1 = u_true * sin(th) + w_true * cos(th);
    let v2 = -u_true * sin(th) + w_true * cos(th);
    let v3 = v_true * sin(th) + w_true * cos(th);
    let v4 = -v_true * sin(th) + w_true * cos(th);

    let u_est = (v1 - v2) / (2.0 * sin(th));
    let v_est = (v3 - v4) / (2.0 * sin(th));
    let w_est = (v1 + v2 + v3 + v4) / (4.0 * cos(th));
    let err_v = (v1 + v2) - (v3 + v4);
    assert_close(u_est, u_true, 1e-6);
    assert_close(v_est, v_true, 1e-6);
    assert_close(w_est, w_true, 1e-6);
    assert_close(err_v, 0.0, 1e-6);
    true
}

let dr = verify_mimo_fmcw_resolution(1.30 * 1.0e9, 79.00 * 1.0e9);
let alpha = verify_ca_cfar_threshold(32.0, 1.0e-5);
let sonar = verify_thorp_ocean_attenuation(54.0, 30.0);
verify_janus_adcp_inversion(20.0);

assert_true(validate_wgsl("shaders/automotive_radars/fmcw_mimo_virtual_array.wgsl"), "MIMO WGSL");
assert_true(validate_wgsl("shaders/automotive_radars/cfar_ca_os_detector.wgsl"), "CFAR WGSL");
assert_true(validate_wgsl("shaders/acoustic_sonar/sonar_transmission_loss_thorp.wgsl"), "Thorp WGSL");
assert_true(validate_wgsl("shaders/acoustic_sonar/acoustic_doppler_current_profiler.wgsl"), "ADCP WGSL");

#{"case": "array_07", "range_res_m": dr, "cfar_alpha": alpha, "sonar": sonar}
