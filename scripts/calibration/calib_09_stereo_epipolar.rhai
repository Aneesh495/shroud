// Shroud :: Sensor Calibration & Radiometric Correction Script
// Script: scripts/calibration/calib_09_stereo_epipolar.rhai
// Target Calibration Kernel: shaders/sensor_calibration/calib_stereo_epipolar_rectification.wgsl

fn evaluate_radiometric_linearity() {
    let lidar = lidar_config(550.0, 6.5e-6, 0.028, 0.0011);
    let reflectances = [0.05, 0.15, 0.30, 0.50, 0.75, 0.95];
    let prev = 0.0;
    let curve = [];
    for refl in reflectances {
        let ph = lidar.expected_photons(40.0, refl, 1.0, 0.92);
        assert_gt(ph, prev);
        prev = ph;
        curve.push(ph);
    }
    assert_close(curve[5] / curve[0], 19.0, 1e-3);
    curve
}

fn evaluate_thermal_nuc_drift(base_temp_k) {
    let offsets = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0];
    let prev_l = 0.0;
    for d in offsets {
        let rad = eval_lwir_radiance(base_temp_k + d);
        assert_gt(rad, prev_l);
        prev_l = rad;
    }
    prev_l
}

let lin = evaluate_radiometric_linearity();
let nuc = evaluate_thermal_nuc_drift(290.00);
assert_true(validate_wgsl("shaders/sensor_calibration/calib_stereo_epipolar_rectification.wgsl"), "calibration WGSL kernel valid");

#{"calibration": "stereo_epipolar", "linearity_samples": lin.len(), "max_lwir": nuc}
