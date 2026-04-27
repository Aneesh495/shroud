// Shroud :: Dual-Wavelength (1064 nm / 532 nm) Canopy Chlorophyll Probing
// Subsystem Module: shaders/lidar_waveforms/dual_wavelength_vegetation_lidar.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates normalized differential vegetation ratio from concurrent NIR and Green laser pulses.
//   - Separates photosynthetic foliage from woody woody debris and bark.
//
// Mathematical & Physical Formulation:
//   NDR = (R_1064 - R_532) / (R_1064 + R_532) active laser reflectance ratio.
//   Foliage exhibits high NDR > 0.6 while bark and soil yield NDR < 0.2.
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

struct DualWaveParams { calib_ratio: f32, pad0: f32, pad1: f32, pad2: f32 };

struct DualEchoResult { ndr: f32, is_foliage: u32, green_photons: f32, nir_photons: f32 };

@group(0) @binding(0) var<uniform> u_dw: DualWaveParams;
@group(0) @binding(1) var<storage, read> s_dual_pulses: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_dual_out: array<DualEchoResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_dual_out)) { return; }
    let p = s_dual_pulses[idx];
    let g = p.x;
    let nir = p.y * u_dw.calib_ratio;
    let ndr = (nir - g) / max(1e-4, nir + g);
    s_dual_out[idx] = DualEchoResult(ndr, select(0u, 1u, ndr > 0.45), g, nir);
}
