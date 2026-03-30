// Shroud :: Optical Birefringence & Retardation in Stressed Glass
// Subsystem: optical_wave_propagation :: polarization_birefringence
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Optical Birefringence & Retardation in Stressed Glass:
 * Computes optical phase retardation delta = (2 * pi / lambda) * (n_e - n_o) * d:
 * Models stress-induced photoelastic fringes in vehicle windshields and polycarbonate covers.
 */

struct RetardationParams { lambda_m: f32, delta_n: f32, thickness_m: f32, angle_rad: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn evaluate_birefringence_phase(d_n: f32, d: f32, wl: f32) -> f32 {

    return (TWO_PI / max(1e-9, wl)) * d_n * d;

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_polarization_birefringence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let phase = evaluate_birefringence_phase(0.005, in_val.z, 0.550e-6);
    let transmission = sin(phase * 0.5) * sin(phase * 0.5);
    let result = vec4<f32>(transmission, phase, in_val.z, 1.0);
    s_output_data[idx] = result;
}
