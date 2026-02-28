// Shroud :: Surface Area Heuristic (SAH) Cost Evaluation & Optimization
// Evaluates partition costs and determines optimal leaf clustering.

struct SahNodeData {
    aabb_min: vec3<f32>,
    left_child: u32,
    aabb_max: vec3<f32>,
    right_child: u32,
    sah_cost: f32,
    prim_count: u32,
    padding: vec2<u32>,
};

fn aabb_surface_area(b_min: vec3<f32>, b_max: vec3<f32>) -> f32 {
    let d = max(vec3<f32>(0.0), b_max - b_min);
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

@group(0) @binding(0) var<storage, read_write> s_sah_nodes: array<SahNodeData>;

@compute @workgroup_size(64, 1, 1)
fn cs_evaluate_sah_cost(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_sah_nodes)) {
        return;
    }

    let node = s_sah_nodes[idx];
    let parent_area = aabb_surface_area(node.aabb_min, node.aabb_max);

    if (node.prim_count > 0u) {
        // Leaf cost = C_isect * prim_count
        s_sah_nodes[idx].sah_cost = 1.0 * f32(node.prim_count);
    } else {
        // Interior cost = C_trav + (SA_L / SA_P) * Cost_L + (SA_R / SA_P) * Cost_R
        let l_idx = node.left_child;
        let r_idx = node.right_child;
        let left_node = s_sah_nodes[l_idx];
        let right_node = s_sah_nodes[r_idx];

        let left_area = aabb_surface_area(left_node.aabb_min, left_node.aabb_max);
        let right_area = aabb_surface_area(right_node.aabb_min, right_node.aabb_max);

        var cost = 1.2; // Traversal constant
        if (parent_area > 1e-6) {
            cost += (left_area / parent_area) * left_node.sah_cost;
            cost += (right_area / parent_area) * right_node.sah_cost;
        }
        s_sah_nodes[idx].sah_cost = cost;
    }
}
