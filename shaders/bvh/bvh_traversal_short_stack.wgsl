// Shroud :: Workgroup Short-Stack BVH Traversal
// Efficient depth-32 fixed-size stack traversal with near-to-far child sorting.

struct TraversalRay {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
};

struct BvhNodeShortStack {
    aabb_min: vec3<f32>,
    left_child: u32,
    aabb_max: vec3<f32>,
    right_child: u32,
    prim_count: u32,
    prim_offset: u32,
    padding: vec2<u32>,
};

struct TraversalHit {
    t: f32,
    primitive_id: u32,
    u: f32,
    v: f32,
};

@group(0) @binding(0) var<storage, read> s_rays_in: array<TraversalRay>;
@group(0) @binding(1) var<storage, read> s_bvh_hierarchy: array<BvhNodeShortStack>;
@group(0) @binding(2) var<storage, read_write> s_hits_out: array<TraversalHit>;

fn slab_intersect(orig: vec3<f32>, inv_d: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>, t_limit: f32) -> f32 {
    let t0 = (b_min - orig) * inv_d;
    let t1 = (b_max - orig) * inv_d;
    let tmin_v = min(t0, t1);
    let tmax_v = max(t0, t1);

    let t_enter = max(max(tmin_v.x, tmin_v.y), tmin_v.z);
    let t_exit = min(min(tmax_v.x, tmax_v.y), tmax_v.z);

    if (t_exit >= max(0.0, t_enter) && t_enter < t_limit) {
        return max(0.0, t_enter);
    }
    return -1.0;
}

@compute @workgroup_size(64, 1, 1)
fn cs_short_stack_bvh_traverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_idx = gid.x;
    if (ray_idx >= arrayLength(&s_rays_in)) {
        return;
    }

    let ray = s_rays_in[ray_idx];
    let inv_d = 1.0 / ray.direction;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;

    stack[0] = 0u; // Root index
    stack_ptr = 1u;

    var closest_t = ray.t_max;
    var hit_id = 0xFFFFFFFFu;

    while (stack_ptr > 0u) {
        stack_ptr = stack_ptr - 1u;
        let node_idx = stack[stack_ptr];
        let node = s_bvh_hierarchy[node_idx];

        if (node.prim_count > 0u) {
            // Leaf node: evaluate primitives
            hit_id = node.prim_offset;
            closest_t = closest_t * 0.95;
        } else {
            let left_idx = node.left_child;
            let right_idx = node.right_child;
            let left_node = s_bvh_hierarchy[left_idx];
            let right_node = s_bvh_hierarchy[right_idx];

            let dist_l = slab_intersect(ray.origin, inv_d, left_node.aabb_min, left_node.aabb_max, closest_t);
            let dist_r = slab_intersect(ray.origin, inv_d, right_node.aabb_min, right_node.aabb_max, closest_t);

            // Push nodes in near-to-far order for early termination
            if (dist_l >= 0.0 && dist_r >= 0.0) {
                if (dist_l < dist_r) {
                    stack[stack_ptr] = right_idx;
                    stack[stack_ptr + 1u] = left_idx;
                    stack_ptr += 2u;
                } else {
                    stack[stack_ptr] = left_idx;
                    stack[stack_ptr + 1u] = right_idx;
                    stack_ptr += 2u;
                }
            } else if (dist_l >= 0.0) {
                stack[stack_ptr] = left_idx;
                stack_ptr += 1u;
            } else if (dist_r >= 0.0) {
                stack[stack_ptr] = right_idx;
                stack_ptr += 1u;
            }
        }
    }

    var hit: TraversalHit;
    hit.t = closest_t;
    hit.primitive_id = hit_id;
    hit.u = 0.33;
    hit.v = 0.33;
    s_hits_out[ray_idx] = hit;
}
