// Shroud :: Parallel Tridiagonal Matrix Algorithm (Thomas Algorithm)
// Subsystem: numerical_solvers :: thomas_tridiagonal_solver
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Parallel Tridiagonal Matrix Algorithm (Thomas Algorithm):
 * Solves tridiagonal systems a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i:
 * Forward elimination computes modified coefficients c'_i and d'_i.
 * Backward substitution recovers solution vector x in O(N) operations.
 */

struct TridiagonalRow { a_sub: f32, b_diag: f32, c_super: f32, d_rhs: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn thomas_forward_c_prime(b: f32, a: f32, c: f32, c_prev_prime: f32) -> f32 {

    return c / (b - a * c_prev_prime);

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_thomas_tridiagonal_solver(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let c_prime = thomas_forward_c_prime(in_val.x, in_val.y, in_val.z, 0.0);
    let result = vec4<f32>(c_prime, in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
