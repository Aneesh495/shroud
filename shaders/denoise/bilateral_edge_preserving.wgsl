// Shroud :: Cross-Bilateral Edge-Preserving Spatial Filter
// Preserves depth discontinuities and object silhouette boundaries.

struct CrossBilateralParams {
    sigma_spatial: f32,
    sigma_range: f32,
    window_radius: i32,
    padding: f32,
};

@group(0) @binding(0) var<uniform> u_bilateral: CrossBilateralParams;
@group(0) @binding(1) var<storage, read> s_raw_signal: array<vec4<f32>>; // rgb, depth
@group(0) @binding(2) var<storage, read_write> s_filtered_signal: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_cross_bilateral_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_filtered_signal)) {
        return;
    }

    let center = s_raw_signal[idx];
    let center_color = center.xyz;
    let center_depth = center.w;

    var accum_color = vec3<f32>(0.0);
    var accum_weight = 0.0;

    let rad = u_bilateral.window_radius;
    for (var d = -rad; d <= rad; d = d + 1) {
        let sample_idx = u32(clamp(i32(idx) + d, 0, i32(arrayLength(&s_raw_signal)) - 1));
        let sample_val = s_raw_signal[sample_idx];

        let spatial_dist2 = f32(d * d);
        let range_dist2 = pow(center_depth - sample_val.w, 2.0);

        let w_spatial = exp(-spatial_dist2 / (2.0 * u_bilateral.sigma_spatial * u_bilateral.sigma_spatial));
        let w_range = exp(-range_dist2 / (2.0 * u_bilateral.sigma_range * u_bilateral.sigma_range));
        let w = w_spatial * w_range;

        accum_color += sample_val.xyz * w;
        accum_weight += w;
    }

    s_filtered_signal[idx] = vec4<f32>(accum_color / max(1e-5, accum_weight), center_depth);
}
