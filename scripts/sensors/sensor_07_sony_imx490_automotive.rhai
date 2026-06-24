// Shroud :: Industrial Sensor Pipeline Verification
// Script: scripts/sensors/sensor_07_sony_imx490_automotive.rhai
// Sensor Profile: sony_imx490_automotive

fn verify_sensor_link_budget() {
    let lidar = lidar_config(550.0, 1.800e-06, 0.0140, 0.0012);
    assert_close(lidar.wavelength_nm(), 550.0, 1e-3);

    let sc = scenario("sony_imx490_automotive_range_test");
    sc.set_fog(0.012, 0.78);
    sc.add_target("cal_target_10pct", vec3(60.0, -1.0, 0.0), vec3(60.2, 1.0, 2.0), 0.10, 295.0);
    sc.add_target("cal_target_80pct", vec3(60.0, 2.0, 0.0), vec3(60.2, 4.0, 2.0), 0.80, 295.0);

    let tau_60 = beer_lambert(sc.extinction(), 60.0);
    let ph_10 = lidar.expected_photons(60.0, 0.10, 1.0, tau_60);
    let ph_80 = lidar.expected_photons(60.0, 0.80, 1.0, tau_60);
    assert_close(ph_80 / ph_10, 8.0, 1e-4);

    // Verify inverse square geometric propagation in clear air
    let ph_30 = lidar.expected_photons(30.0, 0.50, 1.0, 1.0);
    let ph_120 = lidar.expected_photons(120.0, 0.50, 1.0, 1.0);
    assert_close(ph_30 / ph_120, 16.0, 1e-3);

    // Sweep oblique target orientation cosines
    let ph_normal = lidar.expected_photons(60.0, 0.50, 1.0, tau_60);
    let ph_60deg = lidar.expected_photons(60.0, 0.50, 0.5, tau_60);
    assert_close(ph_normal / ph_60deg, 2.0, 1e-4);

    // Check companion heterodyne beat recovery across velocities
    let fmcw = fmcw_config(7.700e+10, 750.0e6, 30.0e-6, 0.20);
    let velocities = [-32.0, -14.0, 0.0, 14.0, 28.5];
    for vel in velocities {
        let fu = fmcw.beat_up(60.0, vel);
        let fd = fmcw.beat_down(60.0, vel);
        assert_close(fmcw.solve_range(fu, fd), 60.0, 0.05);
        assert_close(fmcw.solve_velocity(fu, fd), vel, 0.05);
    }

    #{"ph_10pct": ph_10, "ph_80pct": ph_80, "tau_60": tau_60}
}

let link = verify_sensor_link_budget();
assert_true(validate_wgsl("shaders/sensor_pipelines/sensor_sony_imx490_automotive.wgsl"), "sensor pipeline WGSL compiles cleanly");
assert_true(validate_wgsl("shaders/sensors/sensor_noise_pipeline.wgsl"), "sensor noise WGSL valid");
assert_true(validate_wgsl("shaders/sensors/camera_brown_conrady.wgsl"), "lens distortion WGSL valid");

#{"sensor": "sony_imx490_automotive", "link": link}
