// Shroud :: BB84 Quantum Key Distribution (QKD) Basis Sifting and QBER
// Subsystem Module: shaders/quantum_optics/quantum_key_distribution_bb84_sifting.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Simulates Bennett-Brassard 1984 quantum key distribution protocol between Alice and Bob.
//   - Computes basis match sifting: Keeps bits where Alice and Bob randomly chose identical bases (Rectilinear vs Diagonal).
//
// Mathematical & Physical Formulation:
//   Calculates Quantum Bit Error Rate (QBER) = E / N_sifted.
//   QBER > 11% triggers abort due to eavesdropper (Eve) intercept-resend attack.
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
const HBAR: f32 = 1.054571817e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;

struct QkdConfig { qber_threshold: f32, pad0: f32, pad1: f32, pad2: f32 };

struct QkdBitResult { alice_bit: u32, bob_bit: u32, basis_match: u32, is_error: u32 };

@group(0) @binding(0) var<uniform> u_qkd: QkdConfig;
@group(0) @binding(1) var<storage, read> s_alice_raw: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> s_bob_raw: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> s_qkd_out: array<QkdBitResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_qkd_out)) { return; }
    let a = s_alice_raw[idx];
    let b = s_bob_raw[idx];
    let a_bit = a.x;
    let a_bas = a.y;
    let b_bit = b.x;
    let b_bas = b.y;
    let match_b = select(0u, 1u, a_bas == b_bas);
    let err = select(0u, select(0u, 1u, a_bit != b_bit), match_b == 1u);
    s_qkd_out[idx] = QkdBitResult(a_bit, b_bit, match_b, err);
}
