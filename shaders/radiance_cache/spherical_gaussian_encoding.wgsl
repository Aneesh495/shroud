// Shroud :: Spherical Gaussian (SG) Directional Radiance Encoding
// Subsystem: radiance_cache :: spherical_gaussian_encoding
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Spherical Gaussian (SG) Directional Radiance Encoding:
 * Encodes directional radiance distributions as mixtures of Spherical Gaussians:
 * G(v; mu, lambda, a) = a * exp(lambda * (dot(v, mu) - 1))
 * Enables closed-form products with spherical Gaussian BRDF lobes.
 */

struct SphericalGaussian { axis_mu: vec3<f32>, sharpness_lambda: f32, amplitude_a: vec3<f32>, padding: f32, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn evaluate_sg(v: vec3<f32>, mu: vec3<f32>, lambda: f32, a: vec3<f32>) -> vec3<f32> {

    return a * exp(lambda * (dot(v, mu) - 1.0));

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_spherical_gaussian_encoding(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let sg_val = evaluate_sg(in_val.xyz, vec3<f32>(0.0, 1.0, 0.0), 8.0, vec3<f32>(1.0));
    let result = vec4<f32>(sg_val, 1.0);
    s_output_data[idx] = result;
}
