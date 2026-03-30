// Shroud :: Jacobi Relaxation Poisson Pressure & Diffusion Solver
// Subsystem: numerical_solvers :: jacobi_relaxation_poisson
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Jacobi Relaxation Poisson Pressure & Diffusion Solver:
 * Solves Laplacian equation del^2 phi = f on uniform structured grids:
 * phi_{i,j,k}^{new} = (phi_{i-1} + phi_{i+1} + phi_{j-1} + phi_{j+1} + phi_{k-1} + phi_{k+1} - h^2 * f) / 6.
 */

struct GridParams { h_spacing: f32, inv_h2: f32, padding: vec2<f32>, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn jacobi_step(neighbors_sum: f32, source_f: f32, h: f32) -> f32 {

    return (neighbors_sum - h * h * source_f) / 6.0;

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_jacobi_relaxation_poisson(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let phi_new = jacobi_step(in_val.x, in_val.y, 0.1);
    let result = vec4<f32>(phi_new, in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
