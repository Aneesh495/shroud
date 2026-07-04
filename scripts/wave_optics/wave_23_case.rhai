// Shroud :: Wave Optics, Zernike Aberration & Polarization Verification
// Script: scripts/wave_optics/wave_23_profile_23.rhai

fn evaluate_airy_and_strehl(wavelength_nm, num_aperture, w_rms_um) {
    let lambda_um = wavelength_nm * 1e-3;
    let airy_radius_um = 1.22 * lambda_um / (2.0 * num_aperture);
    assert_gt(airy_radius_um, 0.0);

    let phase_rms = 2.0 * 3.141592653589793 * (w_rms_um / lambda_um);
    let marechal_strehl = exp(-(phase_rms * phase_rms));
    assert_true(marechal_strehl > 0.0 && marechal_strehl <= 1.0, "Strehl ratio in (0, 1]");

    // Diffraction-limited threshold requires Strehl >= 0.80
    let is_diff_limited = marechal_strehl >= 0.80;
    #{"airy_um": airy_radius_um, "strehl": marechal_strehl, "diff_limited": is_diff_limited}
}

fn evaluate_stokes_mueller_cascade(pol_angle_deg) {
    let rad = pol_angle_deg * (3.141592653589793 / 180.0);
    // Horizontal polarized input S = [1, 1, 0, 0] passing ideal linear polarizer at theta
    let s0_in = 1.0;
    let s1_in = 1.0;
    let c2 = cos(2.0 * rad);
    let s2 = sin(2.0 * rad);

    let i_out = 0.5 * (s0_in + s1_in * c2);
    let q_out = 0.5 * (s0_in * c2 + s1_in * c2 * c2);
    let u_out = 0.5 * (s0_in * s2 + s1_in * c2 * s2);

    // Malus's law verification: I_out == cos^2(theta)
    let malus_expected = cos(rad) * cos(rad);
    assert_close(i_out, malus_expected, 1e-5);

    if i_out > 1e-4 {
        let dolp = sqrt(q_out * q_out + u_out * u_out) / i_out;
        assert_close(dolp, 1.0, 1e-4);
    }
    i_out
}

fn evaluate_uniaxial_birefringence(n_o, n_e, thickness_mm, wavelength_nm) {
    let theta_list = [15.0, 30.0, 45.0, 60.0, 75.0];
    let results = [];
    for th_deg in theta_list {
        let th = th_deg * (3.141592653589793 / 180.0);
        let cos_t = cos(th);
        let sin_t = sin(th);
        let inv_ne2 = (cos_t * cos_t) / (n_o * n_o) + (sin_t * sin_t) / (n_e * n_e);
        let n_eff = 1.0 / sqrt(inv_ne2);
        let opd_nm = (thickness_mm * 1.0e6) * abs(n_eff - n_o);
        let retard_waves = opd_nm / wavelength_nm;
        assert_gt(retard_waves, 0.0);
        results.push(#{"theta": th_deg, "n_eff": n_eff, "waves": retard_waves});
    }
    results
}

let psf = evaluate_airy_and_strehl(780.0, 0.800, 0.0510);
let malus = evaluate_stokes_mueller_cascade(30.0 + 23 * 2.0);
let biref = evaluate_uniaxial_birefringence(1.6440, 1.8130, 1.5, 780.0);
assert_true(biref.len() == 5, "5 crystal incidence angles evaluated");

assert_true(validate_wgsl("shaders/wave_optics/airy_disk_psf.wgsl"), "airy PSF WGSL");
assert_true(validate_wgsl("shaders/wave_optics/polarization_stokes_mueller.wgsl"), "stokes WGSL");
assert_true(validate_wgsl("shaders/wave_optics/birefringence_uniaxial.wgsl"), "birefringence WGSL");
assert_true(validate_wgsl("shaders/wave_optics/thin_film_transfer_matrix.wgsl"), "TMM WGSL");

#{"case": "profile_23", "psf": psf, "malus": malus, "angles": biref.len()}
