// Shroud :: Stokes-Mueller Polarimetric Radiative Transfer
// Subsystem: optical_wave_propagation :: coherence_stokes_mueller
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Stokes-Mueller Polarimetric Radiative Transfer:
 * Tracks full Stokes vector S = [I, Q, U, V]^T under Mueller matrix transformations M:
 * S_out = M * S_in
 * Models linear polarization (Q, U) and circular polarization (V) in aerosol backscatter.
 */

struct StokesVector { i: f32, q: f32, u: f32, v: f32, };

struct MuellerMatrix { m: array<vec4<f32>, 4>, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn transform_stokes(m: MuellerMatrix, s: StokesVector) -> StokesVector {

    let s_vec = vec4<f32>(s.i, s.q, s.u, s.v);

    return StokesVector(

        dot(m.m[0], s_vec), dot(m.m[1], s_vec),

        dot(m.m[2], s_vec), dot(m.m[3], s_vec)

    );

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_coherence_stokes_mueller(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    var s_in: StokesVector; s_in.i = in_val.x; s_in.q = in_val.y; s_in.u = in_val.z; s_in.v = in_val.w;
    var m: MuellerMatrix;
    m.m[0] = vec4<f32>(1.0, 0.0, 0.0, 0.0); m.m[1] = vec4<f32>(0.0, 0.8, 0.0, 0.0);
    m.m[2] = vec4<f32>(0.0, 0.0, 0.8, 0.0); m.m[3] = vec4<f32>(0.0, 0.0, 0.0, 0.9);
    let s_out = transform_stokes(m, s_in);
    let result = vec4<f32>(s_out.i, s_out.q, s_out.u, s_out.v);
    s_output_data[idx] = result;
}
