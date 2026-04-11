// Shroud :: Constant Fraction Discriminator (CFD) LiDAR Peak Extraction
// Sub-sample timing extraction independent of pulse amplitude.

struct CfdParams {
    delay_bins: u32,
    fraction: f32,
    noise_threshold: f32,
    time_bin_width_s: f32,
};

struct DetectedEcho {
    tof_s: f32,
    peak_amplitude: f32,
    pulse_width_s: f32,
    quality_flag: u32,
};

@group(0) @binding(0) var<uniform> u_cfd: CfdParams;
@group(0) @binding(1) var<storage, read> s_raw_waveform: array<f32>; // 256 bins per pulse
@group(0) @binding(2) var<storage, read_write> s_detected_echoes: array<DetectedEcho>;

@compute @workgroup_size(64, 1, 1)
fn cs_cfd_peak_detect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pulse_idx = gid.x;
    if (pulse_idx >= arrayLength(&s_detected_echoes)) {
        return;
    }
    let base_idx = pulse_idx * 256u;
    var max_amp = 0.0;
    var peak_bin = 0u;

    // 1. Find absolute peak
    for (var b = 0u; b < 256u; b = b + 1u) {
        let val = s_raw_waveform[base_idx + b];
        if (val > max_amp) {
            max_amp = val;
            peak_bin = b;
        }
    }

    var echo: DetectedEcho;
    if (max_amp < u_cfd.noise_threshold) {
        echo.tof_s = 0.0;
        echo.peak_amplitude = 0.0;
        echo.pulse_width_s = 0.0;
        echo.quality_flag = 0u;
    } else {
        // 2. CFD bipolar zero-crossing timing extraction
        let d = u_cfd.delay_bins;
        let f = u_cfd.fraction;
        var zero_cross_t = f32(peak_bin) * u_cfd.time_bin_width_s;

        for (var b = d; b < peak_bin; b = b + 1u) {
            let sig_delayed = s_raw_waveform[base_idx + b - d];
            let sig_inverted = -f * s_raw_waveform[base_idx + b];
            let cfd_signal = sig_delayed + sig_inverted;
            if (cfd_signal >= 0.0) {
                zero_cross_t = f32(b) * u_cfd.time_bin_width_s;
                break;
            }
        }
        echo.tof_s = zero_cross_t;
        echo.peak_amplitude = max_amp;
        echo.pulse_width_s = f32(d) * u_cfd.time_bin_width_s * 2.0;
        echo.quality_flag = 1u;
    }
    s_detected_echoes[pulse_idx] = echo;
}
