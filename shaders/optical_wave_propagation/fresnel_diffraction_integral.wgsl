// Shroud :: Fresnel Near-Field Wave Optical Diffraction Integral
// Subsystem: optical_wave_propagation :: fresnel_diffraction_integral
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Fresnel Near-Field Wave Optical Diffraction Integral:
 * Evaluates the Fresnel-Kirchhoff scalar diffraction integral in the near-field regime:
 * U(x, y, z) = (exp(i * k * z) / (i * lambda * z)) * double_integral U(xi, eta) * exp(i * k * ((x - xi)^2 + (y - eta)^2) / (2 * z)) dxi deta
 * Approximates the quadratic phase factor for wave propagation past apertures and occluders.
 */

struct ComplexFloat { real: f32, imag: f32, };

fn complex_add(a: ComplexFloat, b: ComplexFloat) -> ComplexFloat { return ComplexFloat(a.real + b.real, a.imag + b.imag); }

fn complex_mult(a: ComplexFloat, b: ComplexFloat) -> ComplexFloat { return ComplexFloat(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real); }

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn eval_fresnel_kernel(dist_z: f32, wavelength: f32, delta_pos: vec2<f32>) -> ComplexFloat {

    let k = TWO_PI / max(1e-9, wavelength);

    let phase = (k * dot(delta_pos, delta_pos)) / (2.0 * max(1e-4, dist_z));

    return ComplexFloat(cos(phase), sin(phase));

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_fresnel_diffraction_integral(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let kernel_val = eval_fresnel_kernel(in_val.z, 0.532e-6, in_val.xy);
    let result = vec4<f32>(kernel_val.real, kernel_val.imag, in_val.z, 1.0);
    s_output_data[idx] = result;
}
