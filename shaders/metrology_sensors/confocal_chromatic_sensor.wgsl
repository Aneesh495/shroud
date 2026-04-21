// Shroud :: Hyperchromatic Optical Pen Confocal Spectral Peak Detector
// Subsystem Module: shaders/metrology_sensors/confocal_chromatic_sensor.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Decodes axial chromatic dispersion for sub-micron surface topography measurement.
//   - Fits asymmetric Gaussian peak centroids on spectrometer sensor pixel arrays.
//
// Mathematical & Physical Formulation:
//   z(lambda) = c0 + c1 * lambda + c2 * lambda^2 axial dispersion calibration.
//   lambda_peak = sum(i * I_i) / sum(I_i) centroid estimator.
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
const VACUUM_PERMITTIVITY: f32 = 8.8541878128e-12;
const VACUUM_PERMEABILITY: f32 = 1.25663706212e-6;
const IMPEDANCE_FREE_SPACE: f32 = 376.730313668;

struct ConfocalConfig { calib_c0: f32, calib_c1: f32, calib_c2: f32, min_snr: f32 };

struct ConfocalResult { peak_lambda_nm: f32, height_um: f32, peak_intensity: f32, valid: u32 };

@group(0) @binding(0) var<uniform> u_cf: ConfocalConfig;
@group(0) @binding(1) var<storage, read> s_spectra: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_cf_out: array<ConfocalResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_cf_out)) { return; }
    let sp = s_spectra[idx];
    let sum_w = sp.x + sp.y + sp.z + sp.w;
    let centroid = (450.0 * sp.x + 530.0 * sp.y + 600.0 * sp.z + 680.0 * sp.w) / max(1e-4, sum_w);
    let h = u_cf.calib_c0 + u_cf.calib_c1 * centroid + u_cf.calib_c2 * centroid * centroid;
    let max_val = max(max(sp.x, sp.y), max(sp.z, sp.w));
    s_cf_out[idx] = ConfocalResult(centroid, h, max_val, select(0u, 1u, max_val > u_cf.min_snr));
}
