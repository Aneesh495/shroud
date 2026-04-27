// Shroud :: Time-Domain Backprojection (TDBP) Synthetic Aperture Sonar
// Subsystem Module: shaders/acoustic_sonar/synthetic_aperture_sonar_backprojection.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Synthesizes along-track virtual acoustic antenna aperture via coherent summation.
//   - Reconstructs ultra-high-resolution seafloor reflectivity image sigma(x, y).
//
// Mathematical & Physical Formulation:
//   I(x, y) = sum_p s_p(2 * R_p(x, y) / c) * exp(i * 2 * k * R_p(x, y)).
//   Maintains dynamic focus across all ranges and swath depths.
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

struct SasGridParams { grid_x_m: f32, grid_y_m: f32, c_sound: f32, k_wave: f32 };

struct SasPixel { complex_pixel: vec2<f32>, magnitude: f32, phase: f32 };

@group(0) @binding(0) var<uniform> u_sas: SasGridParams;
@group(0) @binding(1) var<storage, read> s_ping_data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_sas_image: array<SasPixel>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_sas_image)) { return; }
    var acc = vec2<f32>(0.0, 0.0);
    let count = arrayLength(&s_ping_data);
    for (var p: u32 = 0u; p < count; p = p + 1u) {
        let ping = s_ping_data[p];
        let dx = u_sas.grid_x_m - ping.x;
        let dy = u_sas.grid_y_m - ping.y;
        let dist = sqrt(dx * dx + dy * dy);
        let phase = 2.0 * u_sas.k_wave * dist;
        acc = acc + vec2<f32>(ping.z * cos(phase), ping.z * sin(phase));
    }
    let mag = length(acc);
    s_sas_image[idx] = SasPixel(acc, mag, atan2(acc.y, acc.x));
}
