// Shroud :: Solid-State Flash LiDAR Single-Photon Avalanche Diode (SPAD)
// Simulates photon time-histogram binning, dead time, and pile-up distortion.

struct SpadPixelParams {
    dead_time_ns: f32,
    time_bin_width_ps: f32,
    dark_count_rate_cps: f32,
    pde_efficiency: f32,
};

struct SpadHistogramBin {
    count: atomic<u32>,
};

@group(0) @binding(0) var<uniform> u_spad: SpadPixelParams;
@group(0) @binding(1) var<storage, read> s_incoming_photons_tof: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_histogram_bins: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_spad_histogram_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incoming_photons_tof)) {
        return;
    }

    let photon_tof_s = s_incoming_photons_tof[idx];
    let bin_width_s = u_spad.time_bin_width_ps * 1e-12;

    if (bin_width_s > 1e-15) {
        let bin_idx = u32(photon_tof_s / bin_width_s);
        if (bin_idx < arrayLength(&s_histogram_bins)) {
            // Simplified increment
            s_histogram_bins[bin_idx] += 1u;
        }
    }
}
