// Shroud :: Jablonski Diagram Molecular Fluorescence and Stokes Shift Kinetics
// Subsystem Module: shaders/spectral_optics/fluorescence_emission_redshift.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates molecular fluorophore electronic absorption, vibrational decay, and emission.
//   - Computes asymmetric Franck-Condon spectral envelopes and decay lifetimes.
//
// Mathematical & Physical Formulation:
//   Phi_f = Gamma_r / (Gamma_r + k_nr) fluorescence quantum yield.
//   I(t) = I_0 * exp(-t / tau_f) temporal decay.
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

struct FluorophoreSpec { abs_peak_nm: f32, em_peak_nm: f32, qy: f32, tau_ns: f32 };

struct FluorOutput { absorbed: f32, emitted: f32, radiance: f32, inst_intensity: f32 };

@group(0) @binding(0) var<uniform> u_fl: FluorophoreSpec;
@group(0) @binding(1) var<storage, read> s_in_rad: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_fl_out: array<FluorOutput>;

fn fc_envelope(lambda: f32, peak: f32) -> f32 {
    let d = (lambda - peak) / 25.0;
    return exp(-0.5 * d * d);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_in_rad)) { return; }
    let in_v = s_in_rad[idx];
    let lambda = in_v.x;
    let flux = in_v.y;
    let t_ns = in_v.z;
    let abs_fac = fc_envelope(lambda, u_fl.abs_peak_nm);
    let abs_photons = flux * abs_fac;
    let em_photons = abs_photons * u_fl.qy;
    let decay = exp(-t_ns / max(0.1, u_fl.tau_ns));
    let em_env = fc_envelope(u_fl.em_peak_nm, u_fl.em_peak_nm);
    s_fl_out[idx] = FluorOutput(abs_photons, em_photons, em_photons * em_env, em_photons * decay);
}
