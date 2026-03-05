// Shroud :: ReSTIR DI Spatial Resampling Pass
// Gathers spatial neighbor reservoirs within screen-space radius with geometry rejection.

struct SpatialSurfaceInfo {
    world_pos: vec3<f32>,
    depth: f32,
    world_normal: vec3<f32>,
    material_id: u32,
};

struct SpatialReservoir {
    light_pos: vec3<f32>,
    w_sum: f32,
    light_radiance: vec3<f32>,
    m: f32,
    w: f32,
    padding: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> s_geometry: array<SpatialSurfaceInfo>;
@group(0) @binding(1) var<storage, read> s_in_reservoirs: array<SpatialReservoir>;
@group(0) @binding(2) var<storage, read_write> s_out_reservoirs: array<SpatialReservoir>;

@compute @workgroup_size(64, 1, 1)
fn cs_restir_di_spatial_reuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_out_reservoirs)) {
        return;
    }

    let center_geom = s_geometry[idx];
    var center_res = s_in_reservoirs[idx];

    let num_neighbors = 4u;
    for (var i = 1u; i <= num_neighbors; i = i + 1u) {
        let neighbor_idx = (idx + i * 17u) % arrayLength(&s_in_reservoirs);
        let n_geom = s_geometry[neighbor_idx];

        // Geometric similarity validation
        let normal_sim = dot(center_geom.world_normal, n_geom.world_normal);
        let depth_diff = abs(center_geom.depth - n_geom.depth);

        if (normal_sim > 0.85 && depth_diff < 0.2) {
            let n_res = s_in_reservoirs[neighbor_idx];
            let neighbor_weight = n_res.w * n_res.m * 1.0;

            center_res.w_sum += neighbor_weight;
            center_res.m += n_res.m;

            if (0.25 < (neighbor_weight / max(1e-7, center_res.w_sum))) {
                center_res.light_pos = n_res.light_pos;
                center_res.light_radiance = n_res.light_radiance;
            }
        }
    }

    s_out_reservoirs[idx] = center_res;
}
