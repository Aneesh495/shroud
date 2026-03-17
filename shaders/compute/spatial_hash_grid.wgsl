// Shroud :: 3D Spatial Hashing Grid for Fast Neighbor Searches
// Maps continuous 3D world coordinates to bounded hash buckets for proximity queries.

struct SpatialHashParams {
    cell_size: f32,
    table_size: u32,
    padding: vec2<f32>,
};

fn hash_coords(cx: i32, cy: i32, cz: i32, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(cx) * p1) ^ (u32(cy) * p2) ^ (u32(cz) * p3);
    return n % table_size;
}

@group(0) @binding(0) var<uniform> u_hash_params: SpatialHashParams;
@group(0) @binding(1) var<storage, read> s_positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_bucket_indices: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_compute_spatial_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_positions)) {
        return;
    }

    let p = s_positions[idx].xyz;
    let cs = max(1e-4, u_hash_params.cell_size);

    let cx = i32(floor(p.x / cs));
    let cy = i32(floor(p.y / cs));
    let cz = i32(floor(p.z / cs));

    let bucket = hash_coords(cx, cy, cz, u_hash_params.table_size);
    s_bucket_indices[idx] = bucket;
}
