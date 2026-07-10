// Shroud :: Adaptive Optics Control & Silicon Photonics Verification
// Script: scripts/adaptive_photonics/ap_14_mesh_14.rhai

fn verify_ao_turbulence_scales(r0_m, v_wind) {
    let greenwood_hz = 0.427 * (v_wind / r0_m);
    let tau0_ms = 0.314 * (r0_m / v_wind) * 1000.0;
    assert_gt(greenwood_hz, 0.0);
    assert_gt(tau0_ms, 0.0);

    // Anisoplanatic Strehl falloff with off-axis angle
    let theta0_arcsec = 2.8;
    let angles = [0.5, 1.5, 3.0, 6.0];
    let prev_s = 2.0;
    for th in angles {
        let var_rad2 = (th / theta0_arcsec) ** (5.0 / 3.0);
        let strehl = exp(-var_rad2);
        assert_lt(strehl, prev_s);
        prev_s = strehl;
    }
    #{"f_greenwood_hz": greenwood_hz, "tau0_ms": tau0_ms, "off_axis_strehl": prev_s}
}

fn verify_silicon_microring_resonance(radius_um, n_eff) {
    let lambda_nm = 1550.0;
    let n_g = 4.25;
    let fsr_nm = (lambda_nm * lambda_nm) / (n_g * 2.0 * 3.141592653589793 * radius_um * 1000.0);
    assert_gt(fsr_nm, 0.5);

    // Thermo-optic resonance shift for Si (dn/dT = 1.86e-4 K^-1)
    let dn_dt = 1.86e-4;
    let delta_t_k = 12.0;
    let d_lambda_nm = (lambda_nm / n_g) * dn_dt * delta_t_k;
    assert_gt(d_lambda_nm, 0.0);
    #{"fsr_nm": fsr_nm, "thermo_shift_nm": d_lambda_nm, "n_eff": n_eff}
}

fn verify_opa_grating_beam_width(num_emitters, pitch_um, lambda_um) {
    let aperture_um = num_emitters * pitch_um;
    let fwhm_rad = 0.886 * (lambda_um / aperture_um);
    let fwhm_deg = fwhm_rad * (180.0 / 3.141592653589793);
    assert_gt(fwhm_deg, 0.0);
    assert_lt(fwhm_deg, 5.0);
    fwhm_deg
}

let ao = verify_ao_turbulence_scales(0.210, 11.0);
let ring = verify_silicon_microring_resonance(8.0, 2.42);
let beam_deg = verify_opa_grating_beam_width(320.0, 2.0, 1.55);

assert_true(validate_wgsl("shaders/adaptive_optics/shack_hartmann_wavefront_sensor.wgsl"), "SHWFS WGSL");
assert_true(validate_wgsl("shaders/adaptive_optics/kolmogorov_phase_screen.wgsl"), "Kolmogorov WGSL");
assert_true(validate_wgsl("shaders/photonic_integrated_circuits/microring_resonator_transfer.wgsl"), "Microring WGSL");
assert_true(validate_wgsl("shaders/photonic_integrated_circuits/optical_phased_array_lidar_beamsteer.wgsl"), "OPA WGSL");

#{"case": "mesh_14", "ao": ao, "ring": ring, "opa_fwhm_deg": beam_deg}
