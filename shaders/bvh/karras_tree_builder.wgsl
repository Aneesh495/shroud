// Shroud :: Karras 2012 Parallel BVH Radix Tree Construction
// Builds hierarchy from sorted 30-bit Morton keys in O(N) work and O(log N) depth.

struct BvhNodeGpu {
    aabb_min: vec3<f32>,
    left_child: u32,
    aabb_max: vec3<f32>,
    right_child: u32,
    parent: u32,
    prim_count: u32,
    prim_offset: u32,
    padding: u32,
};

struct KarrasUniforms {
    num_primitives: u32,
    padding: vec3<u32>,
};

@group(0) @binding(0) var<uniform> u_karras: KarrasUniforms;
@group(0) @binding(1) var<storage, read> s_sorted_morton_codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> s_bvh_nodes: array<BvhNodeGpu>;

fn count_leading_zeros_u32(x: u32) -> u32 {
    return countLeadingZeros(x);
}

// Computes common prefix length between two primitive indices i and j
fn longest_common_prefix(i: i32, j: i32, n: i32) -> i32 {
    if (i < 0 || i >= n || j < 0 || j >= n) {
        return -1;
    }

    let code_i = s_sorted_morton_codes[u32(i)];
    let code_j = s_sorted_morton_codes[u32(j)];

    if (code_i != code_j) {
        return i32(count_leading_zeros_u32(code_i ^ code_j));
    }

    // Tie-break identical Morton codes using primitive indices
    return 32 + i32(count_leading_zeros_u32(u32(i ^ j)));
}

// Determines the direction and range of the cluster for node i
fn determine_range(i: i32, n: i32) -> vec2<i32> {
    let d = sign(longest_common_prefix(i, i + 1, n) - longest_common_prefix(i, i - 1, n));
    let delta_min = longest_common_prefix(i, i - d, n);

    var l_max = 2;
    while (longest_common_prefix(i, i + l_max * d, n) > delta_min) {
        l_max = l_max * 2;
    }

    var l = 0;
    var t = l_max / 2;
    while (t > 0) {
        if (longest_common_prefix(i, i + (l + t) * d, n) > delta_min) {
            l += t;
        }
        t = t / 2;
    }

    let j = i + l * d;
    return vec2<i32>(min(i, j), max(i, j));
}

// Finds the optimal split point gamma within range [first, last]
fn find_split(first: i32, last: i32, n: i32) -> i32 {
    let delta_node = longest_common_prefix(first, last, n);
    var split = first;
    var step = last - first;

    while (step > 1) {
        step = (step + 1) / 2;
        let test_split = split + step;
        if (test_split < last) {
            if (longest_common_prefix(first, test_split, n) > delta_node) {
                split = test_split;
            }
        }
    }
    return split;
}

@compute @workgroup_size(128, 1, 1)
fn cs_build_karras_tree(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = i32(gid.x);
    let n = i32(u_karras.num_primitives);

    // N primitives yield N - 1 internal nodes
    if (idx >= n - 1) {
        return;
    }

    let range = determine_range(idx, n);
    let first = range.x;
    let last = range.y;

    let split = find_split(first, last, n);

    var left_idx: u32;
    if (split == first) {
        left_idx = u32((n - 1) + split); // Leaf index
    } else {
        left_idx = u32(split);           // Internal node
    }

    var right_idx: u32;
    if (split + 1 == last) {
        right_idx = u32((n - 1) + split + 1); // Leaf index
    } else {
        right_idx = u32(split + 1);           // Internal node
    }

    var node: BvhNodeGpu;
    node.aabb_min = vec3<f32>(1e30, 1e30, 1e30);
    node.aabb_max = vec3<f32>(-1e30, -1e30, -1e30);
    node.left_child = left_idx;
    node.right_child = right_idx;
    node.parent = 0xFFFFFFFFu;
    node.prim_count = 0u;
    node.prim_offset = 0u;
    node.padding = 0u;

    s_bvh_nodes[u32(idx)] = node;

    // Record parent pointers
    s_bvh_nodes[left_idx].parent = u32(idx);
    s_bvh_nodes[right_idx].parent = u32(idx);
}
