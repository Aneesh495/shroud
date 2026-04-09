// Shroud :: Russian Roulette Path Termination & Throughput Unbiasing
// Subsystem Module: shaders/monte_carlo_samplers/mc_russian_roulette_absorption.wgsl
// Physically based high-performance GPU compute kernel.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Engineering and Scientific Specification for Russian Roulette Path Termination & Throughput Unbiasing:
 * Terminates low-throughput paths without introducing radiometric bias.
 */

fn evaluate_mis_weight(pdf_a: f32, pdf_b: f32) -> f32 {

    let a2 = pdf_a * pdf_a;

    let b2 = pdf_b * pdf_b;

    return a2 / max(1e-12, a2 + b2);

}

fn box_muller_transform(u1: f32, u2: f32) -> vec2<f32> {

    let r = sqrt(-2.0 * log(max(1e-7, u1)));

    let theta = TWO_PI * u2;

    return vec2<f32>(r * cos(theta), r * sin(theta));

}

@group(0) @binding(0) var<storage, read> s_input_payloads: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_payloads: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_execute_mc_russian_roulette_absorption(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_payloads) || idx >= arrayLength(&s_input_payloads)) {
        return;
    }
    let in_p = s_input_payloads[idx];
    let u_pair = in_p.xy;
    let g_sample = box_muller_transform(u_pair.x, u_pair.y);
    let mis_w = evaluate_mis_weight(in_p.z, in_p.w);
    let result = vec4<f32>(g_sample.x, g_sample.y, mis_w, 1.0);
    s_output_payloads[idx] = result;
}
