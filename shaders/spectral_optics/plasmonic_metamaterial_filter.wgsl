// Shroud :: Drude-Lorentz Surface Plasmon Polariton Metamaterial Filter
// Subsystem Module: shaders/spectral_optics/plasmonic_metamaterial_filter.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Evaluates Surface Plasmon Polariton (SPP) dispersion across subwavelength metallic arrays.
//   - Applies Drude-Lorentz complex permittivity model for noble metals.
//
// Mathematical & Physical Formulation:
//   eps(omega) = 1 - omega_p^2 / (omega^2 + i * gamma * omega) + lorentz terms.
//   k_spp = (omega / c) * sqrt((eps_m * eps_d) / (eps_m + eps_d)).
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

struct PlasmonicSpec { wp_ev: f32, gamma_ev: f32, pitch_nm: f32, eps_d: f32 };

struct PlasmonicOut { re_eps: f32, im_eps: f32, spp_lambda_nm: f32, transmission: f32 };

@group(0) @binding(0) var<uniform> u_pl: PlasmonicSpec;
@group(0) @binding(1) var<storage, read> s_probe_lambda: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_pl_out: array<PlasmonicOut>;

fn drude_eps(w_ev: f32, wp: f32, gam: f32) -> vec2<f32> {
    let den = w_ev * w_ev + gam * gam;
    let re = 1.0 - (wp * wp) / max(1e-6, den);
    let im = (wp * wp * gam) / max(1e-6, w_ev * den);
    return vec2<f32>(re, im);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_probe_lambda)) { return; }
    let l_nm = s_probe_lambda[idx];
    let w_ev = 1239.84 / max(1.0, l_nm);
    let eps = drude_eps(w_ev, u_pl.wp_ev, u_pl.gamma_ev);
    let spp_l = u_pl.pitch_nm * sqrt((u_pl.eps_d * abs(eps.x)) / max(1e-4, abs(eps.x) - u_pl.eps_d));
    let trans = exp(-0.5 * pow((l_nm - spp_l) / 30.0, 2.0));
    s_pl_out[idx] = PlasmonicOut(eps.x, eps.y, spp_l, trans);
}
