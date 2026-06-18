// Shroud :: Material Radiometric & Optical Calibration Script
// Script: scripts/materials/mat_15_pedestrian_skin_fitzpatrick6.rhai
// Material Target: pedestrian_skin_fitzpatrick6

fn evaluate_material_radiometry(refl, emiss) {
    let lidar = lidar_905nm();
    let lidar_swir = lidar_config(1550.0, 10.0e-6, 0.030, 0.0009);
    let lidar_green = lidar_config(532.0, 18.0e-6, 0.045, 0.0010);

    let angles_cos = [1.0, 0.940, 0.866, 0.707, 0.500, 0.258, 0.100];
    let prev_ph = 1.0e30;
    let angular_curve = [];

    for c_th in angles_cos {
        let tau_905 = beer_lambert(0.002, 50.0);
        let tau_1550 = beer_lambert(0.0015, 50.0);
        let tau_532 = beer_lambert(0.003, 50.0);
        let p905 = lidar.expected_photons(50.0, refl, c_th, tau_905);
        let p1550 = lidar_swir.expected_photons(50.0, refl * 0.92, c_th, tau_1550);
        let p532 = lidar_green.expected_photons(50.0, refl * 0.85, c_th, tau_532);
        assert_lt(p905, prev_ph);
        prev_ph = p905;
        angular_curve.push(#{"cos_theta": c_th, "p905": p905, "p1550": p1550, "p532": p532});
    }

    // Check distance scaling at fixed normal incidence
    let ranges = [15.0, 30.0, 60.0, 120.0, 200.0];
    let prev_range_ph = 1.0e30;
    for r in ranges {
        let tau = beer_lambert(0.002, r);
        let pr = lidar.expected_photons(r, refl, 1.0, tau);
        assert_lt(pr, prev_range_ph);
        prev_range_ph = pr;
    }

    // Thermal blackbody emission check across operational surface temperatures
    let temps = [253.15, 268.15, 283.15, 298.15, 313.15, 328.15, 343.15];
    let prev_rad = 0.0;
    let sky_downwelling = eval_lwir_radiance(260.0);
    let thermal_curve = [];
    for t_k in temps {
        let bb = eval_lwir_radiance(t_k);
        let apparent = bb * emiss + (1.0 - emiss) * sky_downwelling;
        assert_gt(bb, prev_rad);
        assert_gt(apparent, 0.0);
        prev_rad = bb;
        thermal_curve.push(#{"temp_k": t_k, "apparent_lwir": apparent});
    }

    #{"angular": angular_curve, "thermal": thermal_curve}
}

let rad_metrics = evaluate_material_radiometry(0.310, 0.980);
assert_true(rad_metrics.angular.len() == 7, "7 incident angles evaluated");
assert_true(rad_metrics.thermal.len() == 7, "7 surface temperatures evaluated");

assert_true(validate_wgsl("shaders/materials_library/mat_pedestrian_skin_fitzpatrick6.wgsl"), "material shader valid");
assert_true(validate_wgsl("shaders/spectral_materials/spec_pedestrian_skin_fitzpatrick6.wgsl"), "spectral dispersion shader valid");
assert_true(validate_wgsl("shaders/materials/microfacet_ggx.wgsl"), "GGX BRDF shader valid");
assert_true(validate_wgsl("shaders/materials/disney_principled.wgsl"), "Disney BSDF shader valid");

#{"material": "pedestrian_skin_fitzpatrick6", "reflectance_905": 0.310, "emissivity_lwir": 0.980}
