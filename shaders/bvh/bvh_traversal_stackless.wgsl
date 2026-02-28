// Shroud :: Stackless BVH Traversal via Rope (Escape) Pointers
// Zero-divergence GPU ray tracing traversal without dynamic stack allocations.

struct RayQueryGpu {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
};

struct RopeBvhNode {
    aabb_min: vec3<f32>,
    next_node_on_miss: u32,
    aabb_max: vec3<f32>,
    first_child_or_prim: u32,
    prim_count: u32,
    padding: vec3<u32>,
};

struct HitResultOutput {
    hit_t: f32,
    prim_id: u32,
    padding: vec2<u32>,
};

fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>, t_max: f32) -> bool {
    let t0 = (b_min - origin) * inv_dir;
    let t1 = (b_max - origin) * inv_dir;
    let tmin_v = min(t0, t1);
    let tmax_v = max(t0, t1);

    let t_enter = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    let t_exit = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    return t_exit >= max(0.0, t_enter) && t_enter <= t_max;
}

@group(0) @binding(0) var<storage, read> s_rays: array<RayQueryGpu>;
@group(0) @binding(1) var<storage, read> s_rope_nodes: array<RopeBvhNode>;
@group(0) @binding(2) var<storage, read_write> s_hit_results: array<HitResultOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_stackless_bvh_traverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_idx = gid.x;
    if (ray_idx >= arrayLength(&s_rays)) {
        return;
    }

    let ray = s_rays[ray_idx];
    let inv_dir = vec3<f32>(
        select(1.0 / ray.direction.x, 1e8, abs(ray.direction.x) < 1e-8),
        select(1.0 / ray.direction.y, 1e8, abs(ray.direction.y) < 1e-8),
        select(1.0 / ray.direction.z, 1e8, abs(ray.direction.z) < 1e-8)
    );

    var current_node_idx = 0u;
    var closest_t = ray.t_max;
    var hit_prim = 0xFFFFFFFFu;

    while (current_node_idx != 0xFFFFFFFFu) {
        let node = s_rope_nodes[current_node_idx];
        let hit_box = intersect_aabb(ray.origin, inv_dir, node.aabb_min, node.aabb_max, closest_t);

        if (hit_box) {
            if (node.prim_count > 0u) {
                // Leaf primitive intersection
                hit_prim = node.first_child_or_prim;
                closest_t = closest_t * 0.9; // Simulated intersection distance
                current_node_idx = node.next_node_on_miss;
            } else {
                // Advance to left child
                current_node_idx = node.first_child_or_prim;
            }
        } else {
            // Traverse escape link on bounding box miss
            current_node_idx = node.next_node_on_miss;
        }
    }

    var result: HitResultOutput;
    result.hit_t = closest_t;
    result.prim_id = hit_prim;
    result.padding = vec2<u32>(0u, 0u);
    s_hit_results[ray_idx] = result;
}
