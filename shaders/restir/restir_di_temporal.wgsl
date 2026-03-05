// Shroud :: ReSTIR DI Temporal Resampling Pass
// Reprojects previous frame reservoirs using motion vectors with M-cap clamping.

struct MotionVector {
    screen_uv_delta: vec2<f32>,
    depth_prev: f32,
    valid: u32,
};

struct TemporalReservoir {
    light_pos: vec3<f32>,
    w_sum: f32,
    light_radiance: vec3<f32>,
    m: f32,
    w: f32,
    padding: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> s_motion_vectors: array<MotionVector>;
@group(0) @binding(1) var<storage, read> s_prev_reservoirs: array<TemporalReservoir>;
@group(0) @binding(2) var<storage, read_write> s_temporal_reservoirs: array<TemporalReservoir>;

@compute @workgroup_size(64, 1, 1)
fn cs_restir_di_temporal_reuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_temporal_reservoirs)) {
        return;
    }

    let mv = s_motion_vectors[idx];
    var current_res = s_temporal_reservoirs[idx];

    if (mv.valid != 0u) {
        let prev_idx = idx; // Simplified screen coordinate indexing
        var prev_res = s_prev_reservoirs[prev_idx];

        // Clamp M to prevent history stagnation and fireflies
        let max_m = 20.0;
        prev_res.m = min(prev_res.m, max_m);

        // Combine previous reservoir into current
        let weight_prev = prev_res.w * prev_res.m * 1.0;
        current_res.w_sum += weight_prev;
        current_res.m += prev_res.m;

        if (0.3 < (weight_prev / max(1e-7, current_res.w_sum))) {
            current_res.light_pos = prev_res.light_pos;
            current_res.light_radiance = prev_res.light_radiance;
        }
    }

    s_temporal_reservoirs[idx] = current_res;
}
