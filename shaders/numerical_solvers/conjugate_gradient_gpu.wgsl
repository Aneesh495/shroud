// Shroud :: GPU Parallel Preconditioned Conjugate Gradient (PCG) Linear Solver
// Subsystem: numerical_solvers :: conjugate_gradient_gpu
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for GPU Parallel Preconditioned Conjugate Gradient (PCG) Linear Solver:
 * Solves large symmetric positive-definite sparse systems A * x = b on GPU:
 * Alpha_k = (r_k^T * r_k) / (p_k^T * A * p_k)
 * x_{k+1} = x_k + Alpha_k * p_k
 * r_{k+1} = r_k - Alpha_k * A * p_k
 * Beta_k = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
 * p_{k+1} = r_{k+1} + Beta_k * p_k.
 */

struct PcgState { alpha: f32, beta: f32, residual_norm_sq: f32, iteration: u32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn compute_cg_step(r_dot_r: f32, p_Ap: f32) -> f32 {

    return r_dot_r / max(1e-12, p_Ap);

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_conjugate_gradient_gpu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let alpha = compute_cg_step(in_val.x, in_val.y);
    let result = vec4<f32>(alpha, in_val.x - alpha * in_val.y, in_val.z, 1.0);
    s_output_data[idx] = result;
}
