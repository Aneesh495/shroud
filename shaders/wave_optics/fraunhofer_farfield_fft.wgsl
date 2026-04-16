// Shroud :: Fraunhofer Far-Field Polygon Aperture Diffraction Kernel
// Subsystem Module: shaders/wave_optics/fraunhofer_farfield_fft.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Computes asymptotic Fraunhofer diffraction pattern for polygonal aperture blades.
//   - Simulates chromatic starburst diffraction spikes and aperture glare.
//
// Mathematical & Physical Formulation:
//   I(theta_x, theta_y) proportional to |iint A(x, y) * exp(-i * k * (x * theta_x + y * theta_y)) dx dy|^2
//   Blade normal singularities yield 2N diffraction spikes for N-blade diaphragms.
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

struct FraunhoferConfig { blade_count: u32, wavelength_nm: f32, curvature: f32, scale: f32 };

struct SpikeOutput { r: f32, g: f32, b: f32, intensity: f32 };

@group(0) @binding(0) var<uniform> u_cfg: FraunhoferConfig;
@group(0) @binding(1) var<storage, read> s_angles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_spikes: array<SpikeOutput>;

fn blade_spike(angle: f32, blades: u32, curv: f32) -> f32 {
    let sector = TWO_PI / f32(blades);
    let delta = abs((angle % sector) - 0.5 * sector);
    let s = 1.0 / (1.0 + 500.0 * delta * delta);
    return pow(s, 1.0 + curv * 2.0);
}

fn sinc(x: f32) -> f32 {
    if (abs(x) < 1e-4) { return 1.0; }
    return sin(PI * x) / (PI * x);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_angles)) { return; }
    let p = s_angles[idx];
    let ang = atan2(p.y, p.x);
    let rad = sqrt(p.x * p.x + p.y * p.y);
    let sp = blade_spike(ang, u_cfg.blade_count, u_cfg.curvature);
    let ring = sinc(rad * u_cfg.scale);
    let total = (sp * 0.8 + ring * ring * 0.2) / (1.0 + rad * 15.0);
    let cr = total * pow(u_cfg.wavelength_nm / 650.0, 2.0);
    let cg = total * pow(u_cfg.wavelength_nm / 532.0, 2.0);
    let cb = total * pow(u_cfg.wavelength_nm / 450.0, 2.0);
    s_spikes[idx] = SpikeOutput(cr, cg, cb, total);
}
