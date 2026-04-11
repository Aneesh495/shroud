// Shroud :: FMCW Radar/LiDAR Spectral Windowing & Doppler Estimator
// Implements 4-term Blackman-Harris windowing with sub-bin quadratic interpolation.

const PI: f32 = 3.14159265358979323846;

fn blackman_harris_4term(n: f32, n_total: f32) -> f32 {
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;
    let f = 2.0 * PI * n / (n_total - 1.0);
    return a0 - a1 * cos(f) + a2 * cos(2.0 * f) - a3 * cos(3.0 * f);
}

@group(0) @binding(0) var<storage, read> s_raw_beat_signal: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_windowed_signal: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_apply_fmcw_window(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_windowed_signal)) {
        return;
    }
    let n_total = f32(arrayLength(&s_windowed_signal));
    let w = blackman_harris_4term(f32(idx), n_total);
    s_windowed_signal[idx] = s_raw_beat_signal[idx] * w;
}
