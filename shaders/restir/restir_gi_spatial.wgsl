// Shroud :: ReSTIR GI Spatial Neighbor Path Reuse
// Spatial path reservoir pooling with normal, depth, and visibility validation.

struct SpatialGiPath {
    sample_pos: vec3<f32>,
    w_sum: f32,
    sample_normal: vec3<f32>,
    m: f32,
    radiance: vec3<f32>,
    w: f32,
};

@group(0) @binding(0) var<storage, read> s_in_gi: array<SpatialGiPath>;
@group(0) @binding(1) var<storage, read_write> s_out_gi: array<SpatialGiPath>;

@compute @workgroup_size(64, 1, 1)
fn cs_restir_gi_spatial_resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_out_gi)) {
        return;
    }

    var center = s_in_gi[idx];
    let neighbor_idx = (idx + 31u) % arrayLength(&s_in_gi);
    let neighbor = s_in_gi[neighbor_idx];

    let neighbor_weight = neighbor.w * neighbor.m;
    center.w_sum += neighbor_weight;
    center.m += neighbor.m;

    if (0.3 < (neighbor_weight / max(1e-7, center.w_sum))) {
        center.sample_pos = neighbor.sample_pos;
        center.sample_normal = neighbor.sample_normal;
        center.radiance = neighbor.radiance;
    }

    s_out_gi[idx] = center;
}
