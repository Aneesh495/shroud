// Shroud :: Angular Spectrum Wave Propagation in Homogeneous Media
// Subsystem: optical_wave_propagation :: angular_spectrum_method
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Angular Spectrum Wave Propagation in Homogeneous Media:
 * Propagates complex wavefields via transfer function H(kx, ky) = exp(i * z * sqrt(k^2 - kx^2 - ky^2)):
 * Evanescent waves decay exponentially when kx^2 + ky^2 > k^2.
 * Propagating waves undergo pure phase shift without amplitude loss.
 */

struct WaveParams { k_wave_number: f32, prop_distance_z: f32, padding: vec2<f32>, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn angular_spectrum_transfer(k: f32, z: f32, k_transverse_sq: f32) -> vec2<f32> {

    let k_sq = k * k;

    if (k_transverse_sq <= k_sq) {

        let kz = sqrt(k_sq - k_transverse_sq);

        return vec2<f32>(cos(kz * z), sin(kz * z));

    } else {

        let alpha = sqrt(k_transverse_sq - k_sq);

        return vec2<f32>(exp(-alpha * z), 0.0);

    }

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_angular_spectrum_method(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let tf = angular_spectrum_transfer(TWO_PI / 0.905e-6, in_val.z, dot(in_val.xy, in_val.xy));
    let result = vec4<f32>(tf.x, tf.y, in_val.z, 1.0);
    s_output_data[idx] = result;
}
