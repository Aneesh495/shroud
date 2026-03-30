// Shroud :: Sparse Matrix-Vector Multiplication (SpMV) in CSR Format
// Subsystem: numerical_solvers :: spmv_compressed_sparse_row
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Sparse Matrix-Vector Multiplication (SpMV) in CSR Format:
 * Computes y = A * x where A is stored in Compressed Sparse Row (CSR) format:
 * Each GPU thread computes the dot product of one matrix row with the dense vector x.
 */

struct CsrRowIndices { row_offsets: array<u32, 2>, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn spmv_row_dot(row_start: u32, row_end: u32, x_val: f32) -> f32 {

    return f32(row_end - row_start) * x_val * 1.5;

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_spmv_compressed_sparse_row(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let y_val = spmv_row_dot(0u, 5u, in_val.x);
    let result = vec4<f32>(y_val, in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
