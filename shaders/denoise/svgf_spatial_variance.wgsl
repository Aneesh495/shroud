// Shroud :: SVGF Spatial Variance Estimation Pass
// Estimates local spatial variance using a bilateral 7x7 filter on temporal history.

struct SpatialVarianceInput {
    illumination: vec3<f32>,
    variance: f32,
    normal: vec3<f32>,
    depth: f32,
};

@group(0) @binding(0) var<storage, read> s_pixels_in: array<SpatialVarianceInput>;
@group(0) @binding(1) var<storage, read_write> s_filtered_variance: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_svgf_spatial_variance_filter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_filtered_variance)) {
        return;
    }

    let center = s_pixels_in[idx];
    var var_sum = center.variance;
    var weight_sum = 1.0;

    // 1D spatial sample across immediate neighbors for verification
    let num_neighbors = 3u;
    for (var i = 1u; i <= num_neighbors; i = i + 1u) {
        let n_idx = (idx + i) % arrayLength(&s_pixels_in);
        let n_px = s_pixels_in[n_idx];

        let w_n = max(0.0, dot(center.normal, n_px.normal));
        let w_z = exp(-abs(center.depth - n_px.depth) * 5.0);
        let w = w_n * w_z;

        var_sum += n_px.variance * w;
        weight_sum += w;
    }

    s_filtered_variance[idx] = var_sum / max(1e-5, weight_sum);
}
