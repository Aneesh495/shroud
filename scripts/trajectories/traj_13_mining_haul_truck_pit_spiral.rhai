// Shroud :: Autonomous Vehicle & Aerial Sensor Trajectory Simulation
// Script: scripts/trajectories/traj_13_mining_haul_truck_pit_spiral.rhai
// Simulates multi-epoch ego kinematics, dynamic obstacle intersection, and FMCW Doppler.

fn compute_sensor_mounts(ego_pos, heading) {
    let fwd = vec3(cos(heading), sin(heading), 0.0);
    let left = vec3(-sin(heading), cos(heading), 0.0);
    let up = vec3(0.0, 0.0, 1.0);
    let roof_lidar = ego_pos + up * 0.45 + fwd * 0.15;
    let bumper_radar = ego_pos - up * 0.95 + fwd * 2.15;
    let left_mirror_cam = ego_pos + left * 0.92 + fwd * 0.55;
    #{"lidar": roof_lidar, "radar": bumper_radar, "cam": left_mirror_cam, "fwd": fwd}
}

fn simulate_ego_trajectory(v_ego, omega_yaw, sigma_t) {
    let dt = 0.1;
    let steps = 40;
    let x = 0.0;
    let y = 0.0;
    let heading = 0.0;

    let lidar = lidar_905nm();
    let lidar_1550 = lidar_config(1550.0, 10.0e-6, 0.032, 0.0009);
    let radar = fmcw_config(77.0e9, 1.0e9, 40.0e-6, 0.15);
    let opt_fmcw = fmcw_config(193.4e12, 600.0e6, 25.0e-6, 0.25);

    let lead_x0 = 42.0;
    let lead_v = 18.0;

    let telemetry = [];
    let k = 0;
    while k < steps {
        let t = k * dt;
        x += v_ego * cos(heading) * dt;
        y += v_ego * sin(heading) * dt;
        heading += omega_yaw * dt;

        let lead_x = lead_x0 + lead_v * t;
        let ego_pos = vec3(x, y, 1.5);
        let mounts = compute_sensor_mounts(ego_pos, heading);
        let sensor_ray = ray(mounts.lidar, mounts.fwd);

        let bmin = vec3(lead_x, -1.0, 0.0);
        let bmax = vec3(lead_x + 4.4, 1.0, 1.6);
        let hit_range = sensor_ray.intersect_box(bmin, bmax);

        let true_gap = abs(lead_x - x) + 1.0;
        let rel_vel = lead_v - v_ego;
        let eval_r = if hit_range > 0.0 { hit_range } else { true_gap };

        let tau = beer_lambert(sigma_t, eval_r);
        let ph_905 = lidar.expected_photons(eval_r, 0.40, 1.0, tau);
        let ph_1550 = lidar_1550.expected_photons(eval_r, 0.35, 1.0, beer_lambert(sigma_t * 0.78, eval_r));

        let fu_r = radar.beat_up(eval_r, rel_vel);
        let fd_r = radar.beat_down(eval_r, rel_vel);
        assert_close(radar.solve_range(fu_r, fd_r), eval_r, 0.15);
        assert_close(radar.solve_velocity(fu_r, fd_r), rel_vel, 0.15);

        let fu_o = opt_fmcw.beat_up(eval_r, rel_vel);
        let fd_o = opt_fmcw.beat_down(eval_r, rel_vel);
        assert_close(opt_fmcw.solve_range(fu_o, fd_o), eval_r, 0.15);
        assert_close(opt_fmcw.solve_velocity(fu_o, fd_o), rel_vel, 0.15);

        assert_true(ph_905 >= 0.0 && ph_1550 >= 0.0, "non-negative lidar returns");
        telemetry.push(#{"t": t, "x": x, "y": y, "range": eval_r, "ph_905": ph_905, "ph_1550": ph_1550});
        k += 1;
    }
    telemetry
}

fn validate_slam_shaders() {
    assert_true(validate_wgsl("shaders/lidar_slam/deskew_motion_distortion_slerp.wgsl"), "deskew WGSL");
    assert_true(validate_wgsl("shaders/lidar_slam/loam_feature_curvature_extraction.wgsl"), "LOAM WGSL");
    assert_true(validate_wgsl("shaders/lidar_slam/point_to_edge_line_residual.wgsl"), "edge residual WGSL");
    assert_true(validate_wgsl("shaders/sensor_fusion/ekf_imu_lidar_state_propagation.wgsl"), "EKF WGSL");
    assert_true(validate_wgsl("shaders/sensor_fusion/iterative_closest_point_point_to_plane.wgsl"), "ICP WGSL");
    true
}

let log_data = simulate_ego_trajectory(8.50, -0.0550, 0.02200);
assert_true(log_data.len() == 40, "40 trajectory epochs simulated");
assert_true(validate_slam_shaders(), "SLAM WGSL kernels verified");

#{"trajectory": "mining_haul_truck_pit_spiral", "epochs": log_data.len(), "final_state": log_data[39]}
