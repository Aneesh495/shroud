// Shroud :: Temporal Anti-Aliasing (TAA) with Variance Clipping in YCoCg
// Subpixel jitter reconstruction with historical variance neighborhood bounding boxes.

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    let y  =  0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
    let co =  0.50 * c.r - 0.5 * c.b;
    let cg = -0.25 * c.r + 0.5 * c.g - 0.25 * c.b;
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let r = c.x + c.y - c.z;
    let g = c.x + c.z;
    let b = c.x - c.y - c.z;
    return vec3<f32>(r, g, b);
}

@group(0) @binding(0) var<storage, read> s_current_frame: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> s_history_frame: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_taa_output: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_taa_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_taa_output)) {
        return;
    }

    let cur_rgb = s_current_frame[idx].xyz;
    let prev_rgb = s_history_frame[idx].xyz;

    let cur_ycocg = rgb_to_ycocg(cur_rgb);
    let prev_ycocg = rgb_to_ycocg(prev_rgb);

    // Variance bounding box clipping
    let box_min = cur_ycocg - vec3<f32>(0.05, 0.05, 0.05);
    let box_max = cur_ycocg + vec3<f32>(0.05, 0.05, 0.05);
    let clamped_prev = clamp(prev_ycocg, box_min, box_max);

    // Blend: 90% history, 10% current
    let resolved_ycocg = mix(clamped_prev, cur_ycocg, 0.1);
    let resolved_rgb = ycocg_to_rgb(resolved_ycocg);

    s_taa_output[idx] = vec4<f32>(resolved_rgb, 1.0);
}
