// Shroud :: Frequency-Modulated Continuous Wave (FMCW) Heterodyne LiDAR
// Computes optical beat frequencies to measure range and radial Doppler velocity simultaneously.

const SPEED_OF_LIGHT: f32 = 299792458.0;

struct FmcwLaserUniforms {
    center_frequency_hz: f32,
    sweep_bandwidth_hz: f32,
    chirp_duration_s: f32,
    power_watts: f32,
};

struct FmcwMeasurement {
    f_beat_up_hz: f32,
    f_beat_down_hz: f32,
    estimated_range_m: f32,
    estimated_velocity_m_s: f32,
};

@group(0) @binding(0) var<uniform> u_fmcw: FmcwLaserUniforms;
@group(0) @binding(1) var<storage, read> s_target_ranges: array<f32>;
@group(0) @binding(2) var<storage, read> s_target_velocities: array<f32>;
@group(0) @binding(3) var<storage, read_write> s_fmcw_results: array<FmcwMeasurement>;

@compute @workgroup_size(64, 1, 1)
fn cs_simulate_fmcw_lidar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_fmcw_results)) {
        return;
    }

    let true_range = s_target_ranges[idx];
    let true_velocity = s_target_velocities[idx];

    let c = SPEED_OF_LIGHT;
    let b = u_fmcw.sweep_bandwidth_hz;
    let t_c = u_fmcw.chirp_duration_s;
    let lambda = c / u_fmcw.center_frequency_hz;

    let f_range = (2.0 * b * true_range) / (c * t_c);
    let f_doppler = (2.0 * true_velocity) / lambda;

    let f_up = abs(f_range - f_doppler);
    let f_down = abs(f_range + f_doppler);

    // Invert measurements to recover range and velocity
    let est_f_range = (f_down + f_up) * 0.5;
    let est_f_doppler = (f_down - f_up) * 0.5;

    let est_range = (c * t_c * est_f_range) / (2.0 * b);
    let est_velocity = (lambda * est_f_doppler) / 2.0;

    var out_m: FmcwMeasurement;
    out_m.f_beat_up_hz = f_up;
    out_m.f_beat_down_hz = f_down;
    out_m.estimated_range_m = est_range;
    out_m.estimated_velocity_m_s = est_velocity;

    s_fmcw_results[idx] = out_m;
}
