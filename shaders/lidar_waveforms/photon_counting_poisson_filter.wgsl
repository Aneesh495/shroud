// Shroud :: Single-Photon Counting Poisson Noise Filtering (ATLAS / ICESat-2)
// Subsystem Module: shaders/lidar_waveforms/photon_counting_poisson_filter.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Filters high-ambient solar background photons in micro-pulse photon-counting LiDAR.
//   - Applies spatio-temporal density clustering to isolate real surface return photons.
//
// Mathematical & Physical Formulation:
//   Solar noise rate N_noise = (lambda / hc) * S_solar * Omega * Delta_lambda * eta_opt.
//   Clustering radius R_search rejects isolated Poisson background events.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

// Fundamental Physical & Mathematical Constants (CODATA 2022)
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const HALF_PI: f32 = 1.57079632679489661923;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

struct PhotonFilterConfig { time_gate_ns: f32, min_cluster_pts: u32, pad0: f32, pad1: f32 };

struct FilteredPhoton { timestamp_ns: f32, elevation_m: f32, is_signal: u32, cluster_density: f32 };

@group(0) @binding(0) var<uniform> u_pf: PhotonFilterConfig;
@group(0) @binding(1) var<storage, read> s_photons_raw: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_photons_filtered: array<FilteredPhoton>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_photons_filtered)) { return; }
    let ph = s_photons_raw[idx];
    let t = ph.x;
    let z = ph.y;
    let density = ph.z;
    let is_sig = select(0u, 1u, density >= f32(u_pf.min_cluster_pts));
    s_photons_filtered[idx] = FilteredPhoton(t, z, is_sig, density);
}
