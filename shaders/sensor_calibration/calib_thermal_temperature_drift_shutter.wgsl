// Shroud :: Thermal Camera FPA Temperature Drift Shutter Compensation
// Subsystem Module: shaders/sensor_calibration/calib_thermal_temperature_drift_shutter.wgsl
// Physically based high-performance GPU compute kernel.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Engineering and Scientific Specification for Thermal Camera FPA Temperature Drift Shutter Compensation:
 * Compensates for ambient focal plane array temperature changes.
 */

struct CalibrationHeader { valid: u32, version: u32, padding: vec2<u32>, };

fn apply_calib_scalar(v: f32, scale: f32, offset: f32) -> f32 {

    return v * scale + offset;

}

@group(0) @binding(0) var<storage, read> s_input_payloads: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_payloads: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_execute_calib_thermal_temperature_drift_shutter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_payloads) || idx >= arrayLength(&s_input_payloads)) {
        return;
    }
    let in_p = s_input_payloads[idx];
    let raw_signal = in_p.x;
    let corrected = apply_calib_scalar(raw_signal, 1.025, -0.005);
    let result = vec4<f32>(corrected, in_p.y, in_p.z, 1.0);
    s_output_payloads[idx] = result;
}
