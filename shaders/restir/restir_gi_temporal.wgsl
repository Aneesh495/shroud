// Shroud :: ReSTIR GI Temporal Path Resampling
// Resamples indirect path reservoirs across frames with Jacobian determinant motion correction.

struct GiReservoir {
    sample_pos: vec3<f32>,
    w_sum: f32,
    sample_normal: vec3<f32>,
    m: f32,
    radiance: vec3<f32>,
    w: f32,
};

@group(0) @binding(0) var<storage, read> s_prev_gi_reservoirs: array<GiReservoir>;
@group(0) @binding(1) var<storage, read_write> s_curr_gi_reservoirs: array<GiReservoir>;

// Computes Jacobian determinant for path reconnection motion correction
fn compute_path_jacobian(
    prev_primary_pos: vec3<f32>,
    curr_primary_pos: vec3<f32>,
    sample_pos: vec3<f32>,
    sample_normal: vec3<f32>
) -> f32 {
    let to_prev = prev_primary_pos - sample_pos;
    let to_curr = curr_primary_pos - sample_pos;

    let dist_prev = length(to_prev);
    let dist_curr = length(to_curr);

    let cos_prev = max(0.0, dot(normalize(to_prev), sample_normal));
    let cos_curr = max(0.0, dot(normalize(to_curr), sample_normal));

    if (dist_curr < 1e-4 || cos_prev < 1e-4) {
        return 1.0;
    }

    let jacobian = (cos_curr / cos_prev) * (dist_prev * dist_prev) / (dist_curr * dist_curr);
    return clamp(jacobian, 0.1, 10.0);
}

@compute @workgroup_size(64, 1, 1)
fn cs_restir_gi_temporal_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_curr_gi_reservoirs)) {
        return;
    }

    var curr_res = s_curr_gi_reservoirs[idx];
    let prev_res = s_prev_gi_reservoirs[idx];

    let jacobian = compute_path_jacobian(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.02, 0.0, 0.0),
        prev_res.sample_pos,
        prev_res.sample_normal
    );

    let target_weight = prev_res.w * prev_res.m * jacobian;
    curr_res.w_sum += target_weight;
    curr_res.m += prev_res.m;

    if (0.4 < (target_weight / max(1e-7, curr_res.w_sum))) {
        curr_res.sample_pos = prev_res.sample_pos;
        curr_res.sample_normal = prev_res.sample_normal;
        curr_res.radiance = prev_res.radiance;
    }

    s_curr_gi_reservoirs[idx] = curr_res;
}
