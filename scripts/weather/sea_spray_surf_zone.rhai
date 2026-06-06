// Shroud :: Atmospheric & Weather Radiative Transfer Scenario
// Script: scripts/weather/sea_spray_surf_zone.rhai
// Condition: Surf-zone breaking wave sea salt spray
//
// Evaluates Beer-Lambert optical depth, double Henyey-Greenstein phase asymmetry,
// Rayleigh background scattering, US Standard Atmosphere density scaling,
// multi-wavelength LiDAR link budgets, LWIR thermal contrast, and FMCW kinematics.

// ---------------------------------------------------------------------------
// 1. Scenario Initialization & Atmospheric Media Configuration
// ---------------------------------------------------------------------------

fn configure_environment() {
    let sc = scenario("sea_spray_surf_zone");
    sc.set_fog(0.009200, 0.8400);
    sc.set_solar_zenith(32.00);
    sc.set_ambient_temp(294.15);

    // Populate calibrated targets across horizontal visibility corridor
    sc.add_target("near_spectralon_90", vec3(14.0, -0.5, 0.0), vec3(14.2, 0.5, 1.5), 0.90, 294.15);
    sc.add_target("near_diffuse_10", vec3(18.0, 1.5, 0.0), vec3(18.2, 2.5, 1.5), 0.10, 296.15);
    sc.add_target("mid_vehicle_sedan", vec3(45.0, -1.0, 0.0), vec3(49.5, 1.0, 1.55), 0.440, 306.15);
    sc.add_target("mid_pedestrian", vec3(58.0, -2.5, 0.0), vec3(58.6, -1.9, 1.75), 0.32, 306.15);
    sc.add_target("far_retroreflective_sign", vec3(110.0, 2.5, 1.2), vec3(110.1, 3.3, 2.4), 0.95, 292.65);
    sc.add_target("distant_barrier", vec3(380.0, -4.0, 0.0), vec3(381.0, 4.0, 1.2), 0.42, 294.15);
    sc
}

// ---------------------------------------------------------------------------
// 2. Phase Function Angular Distribution Sweep & Quadrature Verification
// ---------------------------------------------------------------------------

fn verify_phase_functions(gf, gb, wf) {
    let forward_peak = eval_double_hg(1.0, gf, gb, wf);
    let forward_15deg = eval_double_hg(0.9659, gf, gb, wf);
    let side_scatter = eval_double_hg(0.0, gf, gb, wf);
    let back_scatter = eval_double_hg(-1.0, gf, gb, wf);

    assert_gt(forward_peak, forward_15deg);
    assert_gt(forward_15deg, side_scatter);
    assert_true(forward_peak > 0.0, "forward scatter phase must be positive");
    assert_true(back_scatter > 0.0, "backward scatter phase must be positive");

    // Check Rayleigh molecular dipole symmetry: p(1.0) == p(-1.0) > p(0.0)
    let ray_fwd = eval_rayleigh_phase(1.0);
    let ray_back = eval_rayleigh_phase(-1.0);
    let ray_perp = eval_rayleigh_phase(0.0);
    assert_close(ray_fwd, ray_back, 1e-6);
    assert_gt(ray_fwd, ray_perp);

    // Numerically integrate normalized Henyey-Greenstein across 128 midpoint bins
    // High forward asymmetry (g > 0.85) requires sufficient quadrature resolution
    let bins = 128;
    let d_mu = 2.0 / bins;
    let integral = 0.0;
    let i = 0;
    while i < bins {
        let mu = -1.0 + (i + 0.5) * d_mu;
        let p_val = eval_henyey_greenstein(mu, gf);
        integral += 2.0 * 3.141592653589793 * p_val * d_mu;
        i += 1;
    }
    assert_true(integral > 0.85 && integral < 1.15, "HG quadrature normalized");

    #{"forward": forward_peak, "side": side_scatter, "back": back_scatter, "integral": integral}
}

// ---------------------------------------------------------------------------
// 3. Spectral Transmittance & Multi-Sensor Attenuation Profile
// ---------------------------------------------------------------------------

fn sweep_range_attenuation(sigma_base, alt_m, refl) {
    let density_ratio = us_std_density(alt_m);
    assert_true(density_ratio > 0.0 && density_ratio <= 1.05, "valid atmospheric density ratio");

    let lidar_905 = lidar_905nm();
    let lidar_1550 = lidar_config(1550.0, 12.0e-6, 0.035, 0.0008);
    let lidar_532 = lidar_config(532.0, 20.0e-6, 0.050, 0.0010);
    assert_close(lidar_905.wavelength_nm(), 905.0, 1e-4);
    assert_close(lidar_1550.wavelength_nm(), 1550.0, 1e-4);
    assert_close(lidar_532.wavelength_nm(), 532.0, 1e-4);

    let prev_photons = 1.0e30;
    let ranges = [10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 135.0, 175.0, 220.0];
    let records = [];

    for r in ranges {
        let tau_905 = beer_lambert(sigma_base * density_ratio, r);
        let tau_1550 = beer_lambert(sigma_base * density_ratio * 0.78, r);
        let tau_532 = beer_lambert(sigma_base * density_ratio * 1.18, r);

        assert_true(tau_905 > 0.0 && tau_905 <= 1.0, "transmittance in (0, 1]");
        assert_true(tau_1550 >= tau_905, "1550 nm penetrates equal or better in haze");
        assert_true(tau_905 >= tau_532, "905 nm penetrates equal or better than 532 nm");

        let ph_905 = lidar_905.expected_photons(r, refl, 1.0, tau_905);
        let ph_1550 = lidar_1550.expected_photons(r, refl, 1.0, tau_1550);
        let ph_532 = lidar_532.expected_photons(r, refl, 1.0, tau_532);
        assert_lt(ph_905, prev_photons);
        prev_photons = ph_905;

        records.push(#{
            "range_m": r,
            "tau_905": tau_905,
            "tau_1550": tau_1550,
            "ph_905": ph_905,
            "ph_1550": ph_1550,
            "ph_532": ph_532
        });
    }
    records
}

// ---------------------------------------------------------------------------
// 4. Thermal LWIR Contrast & Ray-Box Occlusion Verification
// ---------------------------------------------------------------------------

fn verify_thermal_and_occlusion(ambient_k) {
    let rad_bg = eval_lwir_radiance(ambient_k);
    let rad_warm = eval_lwir_radiance(ambient_k + 15.0);
    let rad_hot = eval_lwir_radiance(ambient_k + 45.0);
    let rad_ped = eval_lwir_radiance(306.15);

    assert_gt(rad_warm, rad_bg);
    assert_gt(rad_hot, rad_warm);
    assert_gt(rad_ped, 10.0);

    let sensor_origin = vec3(0.0, 0.0, 1.4);
    let forward_dir = vec3(1.0, 0.0, -0.005);
    let probe_ray = ray(sensor_origin, forward_dir);

    let hit_sedan = probe_ray.intersect_box(vec3(45.0, -1.0, 0.0), vec3(49.5, 1.0, 1.55));
    assert_true(hit_sedan > 44.0 && hit_sedan < 46.0, "sedan forward ray intersection");

    let hit_point = probe_ray.at(hit_sedan);
    assert_gt(hit_point.x, 44.5);

    // Check pedestrian off-axis ray
    let ped_dir = (vec3(58.3, -2.2, 1.0) - sensor_origin).normalize();
    let ped_ray = ray(sensor_origin, ped_dir);
    let hit_ped = ped_ray.intersect_box(vec3(58.0, -2.5, 0.0), vec3(58.6, -1.9, 1.75));
    assert_true(hit_ped > 57.0 && hit_ped < 60.0, "pedestrian off-axis ray intersection");

    let miss_ray = ray(sensor_origin, vec3(0.0, 1.0, 0.0));
    let miss_t = miss_ray.intersect_box(vec3(45.0, -1.0, 0.0), vec3(49.5, 1.0, 1.55));
    assert_close(miss_t, -1.0, 1e-6);

    #{"lwir_bg": rad_bg, "lwir_hot": rad_hot, "hit_sedan": hit_sedan, "hit_ped": hit_ped}
}

// ---------------------------------------------------------------------------
// 5. Coupled FMCW Velocity Discrimination Under Adverse Extinction
// ---------------------------------------------------------------------------

fn verify_fmcw_velocity_pipeline() {
    let fmcw_optical = fmcw_config(193.4e12, 600.0e6, 25.0e-6, 0.25);
    let fmcw_radar = fmcw_config(77.0e9, 1.0e9, 40.0e-6, 0.15);
    let test_cases = [
        [25.0, 12.5],
        [68.4, -22.0],
        [120.0, 0.0],
        [185.2, -31.5],
        [240.0, 18.75],
    ];

    for tc in test_cases {
        let r_true = tc[0];
        let v_true = tc[1];

        let fu_opt = fmcw_optical.beat_up(r_true, v_true);
        let fd_opt = fmcw_optical.beat_down(r_true, v_true);
        assert_close(fmcw_optical.solve_range(fu_opt, fd_opt), r_true, 0.05);
        assert_close(fmcw_optical.solve_velocity(fu_opt, fd_opt), v_true, 0.05);

        let fu_rad = fmcw_radar.beat_up(r_true, v_true);
        let fd_rad = fmcw_radar.beat_down(r_true, v_true);
        assert_close(fmcw_radar.solve_range(fu_rad, fd_rad), r_true, 0.05);
        assert_close(fmcw_radar.solve_velocity(fu_rad, fd_rad), v_true, 0.05);
    }
}

// ---------------------------------------------------------------------------
// 6. Camera Distortion & Spatial Morton Ordering Across Scene Targets
// ---------------------------------------------------------------------------

fn verify_camera_and_bvh_coherency() {
    let uv0 = brown_conrady_distort(0.25, -0.15, -0.18, 0.04, 0.001, -0.001, 0.002);
    assert_true(uv0.len() == 2, "Brown-Conrady returns 2D coordinates");
    let uv1 = brown_conrady_distort(0.50, -0.30, -0.18, 0.04, 0.001, -0.001, 0.002);
    let r0 = sqrt(uv0[0] * uv0[0] + uv0[1] * uv0[1]);
    let r1 = sqrt(uv1[0] * uv1[0] + uv1[1] * uv1[1]);
    assert_gt(r1, r0);

    let m_near = morton_3d(vec3(0.05, 0.45, 0.10));
    let m_far = morton_3d(vec3(0.95, 0.65, 0.85));
    assert_lt(m_near, m_far);
    true
}

// ---------------------------------------------------------------------------
// 7. Execute Scenario & Validate Referenced WGSL Modules
// ---------------------------------------------------------------------------

let env = configure_environment();
assert_true(env.target_count() == 6, "scenario populated with 6 targets");
assert_close(env.extinction(), 0.009200, 1e-6);
assert_true(env.albedo() > 0.85, "scattering-dominated atmospheric medium");

let phase_stats = verify_phase_functions(0.8400, -0.3200, 0.8900);
let range_sweep = sweep_range_attenuation(0.009200, 2.0, 0.440);
let thermal_res = verify_thermal_and_occlusion(294.15);
verify_fmcw_velocity_pipeline();
verify_camera_and_bvh_coherency();

// Cross-check volumetric & sensor WGSL shader compilation status
assert_true(validate_wgsl("shaders/volumetric/phase_henyey_greenstein.wgsl"), "HG WGSL valid");
assert_true(validate_wgsl("shaders/volumetric/delta_tracking_woodcock.wgsl"), "Woodcock WGSL valid");
assert_true(validate_wgsl("shaders/volumetric/ratio_tracking_transmittance.wgsl"), "Ratio WGSL valid");
assert_true(validate_wgsl("shaders/sensors/lidar_tof_pulsed.wgsl"), "ToF LiDAR WGSL valid");
assert_true(validate_wgsl("shaders/sensors/lidar_fmcw_heterodyne.wgsl"), "FMCW LiDAR WGSL valid");

#{"scenario": "sea_spray_surf_zone", "targets": env.target_count(), "phase": phase_stats, "samples": range_sweep.len(), "thermal": thermal_res}
