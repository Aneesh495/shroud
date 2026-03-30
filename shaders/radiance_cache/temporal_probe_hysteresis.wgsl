// Shroud :: Temporal Probe Hysteresis & Disocclusion Reset
// Subsystem: radiance_cache :: temporal_probe_hysteresis
// Physically derived WebGPU compute kernel with numerical stability bounds.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

/*
 * Theoretical and Mathematical Formulation for Temporal Probe Hysteresis & Disocclusion Reset:
 * Detects sudden lighting changes (shadow disocclusion) and resets probe hysteresis:
 * Prevents ghosting when dynamic vehicles or headlights move through the scene.
 */

struct HysteresisParams { threshold: f32, reset_weight: f32, padding: vec2<f32>, };

@group(0) @binding(0) var<storage, read> s_input_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_output_data: array<vec4<f32>>; // Output payload

fn detect_disocclusion(old_val: f32, new_val: f32, thresh: f32) -> bool {

    return abs(old_val - new_val) / max(1e-4, old_val) > thresh;

}

@compute @workgroup_size(64, 1, 1)
fn cs_main_temporal_probe_hysteresis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_output_data) || idx >= arrayLength(&s_input_data)) {
        return;
    }
    let in_val = s_input_data[idx];
    let reset = detect_disocclusion(in_val.x, in_val.y, 0.5);
    let result = vec4<f32>(select(0.05, 1.0, reset), in_val.x, in_val.y, 1.0);
    s_output_data[idx] = result;
}
