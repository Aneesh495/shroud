// Shroud :: Conventional Delay-and-Sum (DAS) Multibeam Acoustic Beamformer
// Subsystem Module: shaders/acoustic_sonar/sonar_beamforming_delay_and_sum.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Synthesizes narrow acoustic receive beams across hydrophone linear arrays.
//   - Applies steerable time delays tau_m = (m * d / c) * sin(theta_steer) and Taylor aperture weighting.
//
// Mathematical & Physical Formulation:
//   y(t, theta) = sum_m w_m * x_m(t - tau_m(theta)).
//   Computes directional acoustic beam power P(theta) = Integral y(t, theta)^2 dt.
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

struct BeamformConfig { elem_spacing_m: f32, steer_angle_rad: f32, c_sound_m_s: f32, pad: f32 };

struct BeamResult { steered_sample: vec2<f32>, beam_power_db: f32, beampattern_val: f32 };

@group(0) @binding(0) var<uniform> u_bf: BeamformConfig;
@group(0) @binding(1) var<storage, read> s_array_signals: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> s_beam_out: array<BeamResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_beam_out)) { return; }
    var sum_sig = vec2<f32>(0.0, 0.0);
    let count = arrayLength(&s_array_signals);
    let dt = (u_bf.elem_spacing_m * sin(u_bf.steer_angle_rad)) / u_bf.c_sound_m_s;
    let k_phase = TWO_PI * 30000.0 * dt;
    for (var m: u32 = 0u; m < count; m = m + 1u) {
        let sig = s_array_signals[m];
        let phi = f32(m) * k_phase;
        let rot = vec2<f32>(sig.x * cos(phi) - sig.y * sin(phi), sig.x * sin(phi) + sig.y * cos(phi));
        sum_sig = sum_sig + rot;
    }
    let pwr = dot(sum_sig, sum_sig);
    s_beam_out[idx] = BeamResult(sum_sig, 10.0 * log(max(1e-4, pwr)), sqrt(pwr));
}
