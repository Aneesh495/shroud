// Shroud :: Bottom-Up BVH Bounding Box Aggregation
// Uses atomic flags to propagate tight bounding boxes from leaves up to the root.

struct BvhNodeFit {
    aabb_min: vec3<f32>,
    left_child: u32,
    aabb_max: vec3<f32>,
    right_child: u32,
    parent: u32,
    flag: u32,
    prim_count: u32,
    prim_offset: u32,
};

struct BoundsFitUniforms {
    num_primitives: u32,
    padding: vec3<u32>,
};

@group(0) @binding(0) var<uniform> u_fit_params: BoundsFitUniforms;
@group(0) @binding(1) var<storage, read_write> s_nodes: array<BvhNodeFit>;
@group(0) @binding(2) var<storage, read_write> s_atomic_counters: array<atomic<u32>>;

@compute @workgroup_size(128, 1, 1)
fn cs_fit_bvh_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let prim_idx = gid.x;
    let n = u_fit_params.num_primitives;
    if (prim_idx >= n) {
        return;
    }

    // Leaf index is offset by (n - 1)
    let leaf_idx = (n - 1u) + prim_idx;
    var current_node = s_nodes[leaf_idx].parent;

    while (current_node != 0xFFFFFFFFu) {
        // Atomic increment on arrival counter
        let arrival = atomicAdd(&s_atomic_counters[current_node], 1u);

        // First thread to reach this node terminates; second thread computes union and continues up
        if (arrival == 0u) {
            return;
        }

        let left = s_nodes[current_node].left_child;
        let right = s_nodes[current_node].right_child;

        let left_min = s_nodes[left].aabb_min;
        let left_max = s_nodes[left].aabb_max;
        let right_min = s_nodes[right].aabb_min;
        let right_max = s_nodes[right].aabb_max;

        s_nodes[current_node].aabb_min = min(left_min, right_min);
        s_nodes[current_node].aabb_max = max(left_max, right_max);

        current_node = s_nodes[current_node].parent;
    }
}
