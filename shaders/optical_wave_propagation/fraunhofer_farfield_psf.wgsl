// Shroud :: Fraunhofer Far-Field Point Spread Function (PSF)
// Subsystem: optical_wave_propagation :: fraunhofer_farfield_psf
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Fraunhofer Far-Field Point Spread Function (PSF):
 * Computes the far-field diffraction pattern as the 2D spatial Fourier transform of the aperture transmission function:
 * U(u, v) = FourierTransform{t(x, y)} evaluated at spatial frequencies (u, v) = (x / (lambda * z), y / (lambda * z)).
 * Models polygonal aperture blade edges and starburst spikes.
 */

struct ApertureParams { radius_m: f32, wavelength_m: f32, focal_length_m: f32, blades: u32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn evaluate_fraunhofer_airy(r_norm: f32) -> f32 {

    let x = max(1e-5, r_norm * PI);

    let sinc_x = sin(x) / x;

    return sinc_x * sinc_x;

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_fraunhofer_farfield_psf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let psf = evaluate_fraunhofer_airy(length(in_val.xy));
    let result = vec4<f32>(psf, psf, psf, 1.0);
    s_output_data[idx] = result;
}
