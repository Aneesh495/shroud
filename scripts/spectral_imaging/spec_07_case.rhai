// Shroud :: Hyperspectral Unmixing, Emissivity & Water Vapor Retrieval
// Script: scripts/spectral_imaging/spec_07_bandset_07.rhai

fn verify_simplex_unmixing(w0, w1, w2) {
    let total = w0 + w1 + w2;
    let a = w0 / total;
    let b = w1 / total;
    let c = w2 / total;
    assert_close(a + b + c, 1.0, 1e-6);
    assert_true(a >= 0.0 && b >= 0.0 && c >= 0.0, "abundance non-negativity holds");

    // Endmember pure signatures in 4 VNIR-SWIR bands
    let e_veg = [0.05, 0.04, 0.48, 0.25];
    let e_soil = [0.22, 0.28, 0.32, 0.38];
    let e_water = [0.04, 0.03, 0.01, 0.005];

    let mixed = [];
    let i = 0;
    while i < 4 {
        mixed.push(a * e_veg[i] + b * e_soil[i] + c * e_water[i]);
        i += 1;
    }
    let ndvi = (mixed[2] - mixed[1]) / (mixed[2] + mixed[1]);
    assert_true(ndvi >= -1.0 && ndvi <= 1.0, "NDVI strictly bounded");
    #{"abundances": [a, b, c], "ndvi": ndvi}
}

fn verify_spectral_angle_mapper() {
    let v1 = vec3(0.12, 0.45, 0.30);
    let v2 = v1 * 2.5; // Pure illuminance scaling
    let cos_th = v1.dot(v2) / (v1.length() * v2.length());
    assert_close(cos_th, 1.0, 1e-5);

    let v_orth = vec3(-0.45, 0.12, 0.0);
    let cos_orth = v1.dot(v_orth) / (v1.length() * v_orth.length());
    assert_close(cos_orth, 0.0, 1e-5);
    true
}

fn verify_tes_lwir_emissivity(surface_k, sky_k, eps_band) {
    let bb_surf = eval_lwir_radiance(surface_k);
    let bb_sky = eval_lwir_radiance(sky_k);
    let l_sensor = eps_band * bb_surf + (1.0 - eps_band) * bb_sky;
    let eps_recovered = (l_sensor - bb_sky) / (bb_surf - bb_sky);
    assert_close(eps_recovered, eps_band, 1e-5);
    l_sensor
}

fn verify_water_vapor_apda(pwv_g_cm2) {
    let airmasses = [1.0, 1.4, 2.0, 3.0];
    let prev_tau = 2.0;
    for m in airmasses {
        let optical_depth = 0.38 * ((pwv_g_cm2 * m) ** 0.625);
        let trans = exp(-optical_depth);
        assert_lt(trans, prev_tau);
        assert_gt(trans, 0.0);
        prev_tau = trans;
    }
    prev_tau
}

let unmix = verify_simplex_unmixing(0.390, 0.500, 0.180);
verify_spectral_angle_mapper();
let lwir_rad = verify_tes_lwir_emissivity(305.0, 260.0, 0.94);
let wv_trans = verify_water_vapor_apda(3.75);

assert_true(validate_wgsl("shaders/hyperspectral/hyperspectral_unmixing_fcls.wgsl"), "FCLS WGSL");
assert_true(validate_wgsl("shaders/hyperspectral/spectral_angle_mapper.wgsl"), "SAM WGSL");
assert_true(validate_wgsl("shaders/hyperspectral/thermal_hyperspectral_emissivity_tes.wgsl"), "TES WGSL");
assert_true(validate_wgsl("shaders/hyperspectral/vegetation_indices_ndvi_pri.wgsl"), "NDVI WGSL");

#{"spectral_case": "bandset_07", "ndvi": unmix.ndvi, "lwir": lwir_rad, "wv_trans": wv_trans}
