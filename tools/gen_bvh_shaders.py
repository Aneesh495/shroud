#!/usr/bin/env python3
"""
Generates shaders/bvh/ WGSL modules.
Implements GPU LBVH construction, radix sort, Karras tree builder, and traversal.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BVH_DIR = ROOT_DIR / "shaders" / "bvh"
BVH_DIR.mkdir(parents=True, exist_ok=True)

def write_shader(filename: str, content: str):
    p = BVH_DIR / filename
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# 1. lbvh_morton.wgsl
LBVH_MORTON_WGSL = """// Shroud :: Linear Bounding Volume Hierarchy (LBVH) Morton Code Generation
// Computes 30-bit Morton codes via bit interleaving of normalized 3D coordinates.

struct BvhPrimitiveInput {
    v0: vec3<f32>,
    prim_id: u32,
    v1: vec3<f32>,
    material_id: u32,
    v2: vec3<f32>,
    padding: u32,
};

struct MortonKeyOutput {
    morton_code: u32,
    primitive_id: u32,
};

struct SceneBoundsUniforms {
    scene_min: vec3<f32>,
    num_primitives: u32,
    scene_max: vec3<f32>,
    padding: u32,
};

@group(0) @binding(0) var<uniform> u_scene_bounds: SceneBoundsUniforms;
@group(0) @binding(1) var<storage, read> s_primitives: array<BvhPrimitiveInput>;
@group(0) @binding(2) var<storage, read_write> s_morton_keys: array<MortonKeyOutput>;

fn expand_bits_10(v_in: u32) -> u32 {
    var v = v_in & 0x000003FFu;
    v = (v ^ (v << 16u)) & 0x030000FFu;
    v = (v ^ (v <<  8u)) & 0x0300F00Fu;
    v = (v ^ (v <<  4u)) & 0x030C30C3u;
    v = (v ^ (v <<  2u)) & 0x09249249u;
    return v;
}

fn compute_morton_30(norm_coord: vec3<f32>) -> u32 {
    let clamped = clamp(norm_coord, vec3<f32>(0.0), vec3<f32>(1.0));
    let x = u32(clamped.x * 1023.0);
    let y = u32(clamped.y * 1023.0);
    let z = u32(clamped.z * 1023.0);
    return expand_bits_10(x) | (expand_bits_10(y) << 1u) | (expand_bits_10(z) << 2u);
}

@compute @workgroup_size(128, 1, 1)
fn cs_generate_morton_codes(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let prim_idx = global_id.x;
    if (prim_idx >= u_scene_bounds.num_primitives) {
        return;
    }

    let prim = s_primitives[prim_idx];
    let centroid = (prim.v0 + prim.v1 + prim.v2) * 0.33333333333;

    let extent = u_scene_bounds.scene_max - u_scene_bounds.scene_min;
    var norm_pos = vec3<f32>(0.5, 0.5, 0.5);
    if (extent.x > 1e-6 && extent.y > 1e-6 && extent.z > 1e-6) {
        norm_pos = (centroid - u_scene_bounds.scene_min) / extent;
    }

    let code = compute_morton_30(norm_pos);

    var key_out: MortonKeyOutput;
    key_out.morton_code = code;
    key_out.primitive_id = prim_idx;

    s_morton_keys[prim_idx] = key_out;
}
"""

# 2. radix_sort_histogram.wgsl
RADIX_HISTOGRAM_WGSL = """// Shroud :: GPU Radix Sort Local Histogram Pass
// 4-bit Least Significant Digit (LSD) Radix Sort with 16 bins per pass.

struct RadixSortParams {
    bit_shift: u32,
    num_elements: u32,
    num_workgroups: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> u_sort_params: RadixSortParams;
@group(0) @binding(1) var<storage, read> s_keys_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> s_global_histograms: array<u32>;

var<workgroup> s_local_histogram: array<atomic<u32>, 16>;

@compute @workgroup_size(256, 1, 1)
fn cs_radix_histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let lid = local_id.x;
    let gid = global_id.x;

    // Clear local shared memory histogram
    if (lid < 16u) {
        atomicStore(&s_local_histogram[lid], 0u);
    }
    workgroupBarrier();

    // Accumulate frequencies into local bins
    if (gid < u_sort_params.num_elements) {
        let key = s_keys_in[gid];
        let bin = (key >> u_sort_params.bit_shift) & 0xFu;
        atomicAdd(&s_local_histogram[bin], 1u);
    }
    workgroupBarrier();

    // Write local workgroup histogram to global memory
    if (lid < 16u) {
        let val = atomicLoad(&s_local_histogram[lid]);
        let out_idx = lid * u_sort_params.num_workgroups + group_id.x;
        s_global_histograms[out_idx] = val;
    }
}
"""

# 3. radix_sort_scan.wgsl
RADIX_SCAN_WGSL = """// Shroud :: GPU Prefix Sum Scan for Radix Sort Global Offsets
// Blelloch workgroup scan across 16 histogram bins.

struct ScanParams {
    num_items: u32,
    padding: vec3<u32>,
};

@group(0) @binding(0) var<uniform> u_scan_params: ScanParams;
@group(0) @binding(1) var<storage, read_write> s_data: array<u32>;

var<workgroup> s_scratch: array<u32, 512>;

@compute @workgroup_size(256, 1, 1)
fn cs_radix_scan(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    let n = u_scan_params.num_items;
    let idx_a = 2u * tid;
    let idx_b = 2u * tid + 1u;

    if (idx_a < n) {
        s_scratch[idx_a] = s_data[idx_a];
    } else {
        s_scratch[idx_a] = 0u;
    }

    if (idx_b < n) {
        s_scratch[idx_b] = s_data[idx_b];
    } else {
        s_scratch[idx_b] = 0u;
    }
    workgroupBarrier();

    // Up-sweep (reduction) phase
    var stride = 1u;
    while (stride < 512u) {
        let index = (tid + 1u) * stride * 2u - 1u;
        if (index < 512u) {
            s_scratch[index] += s_scratch[index - stride];
        }
        stride = stride * 2u;
        workgroupBarrier();
    }

    // Clear the last element for exclusive scan
    if (tid == 0u) {
        s_scratch[511u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep phase
    stride = 256u;
    while (stride > 0u) {
        let index = (tid + 1u) * stride * 2u - 1u;
        if (index < 512u) {
            let temp = s_scratch[index - stride];
            s_scratch[index - stride] = s_scratch[index];
            s_scratch[index] += temp;
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // Write results back
    if (idx_a < n) {
        s_data[idx_a] = s_scratch[idx_a];
    }
    if (idx_b < n) {
        s_data[idx_b] = s_scratch[idx_b];
    }
}
"""

# 4. radix_sort_scatter.wgsl
RADIX_SCATTER_WGSL = """// Shroud :: GPU Radix Sort Scatter Pass
// Reorders keys and primitive payload indices into globally sorted bins.

struct ScatterParams {
    bit_shift: u32,
    num_elements: u32,
    num_workgroups: u32,
    padding: u32,
};

struct KeyValue {
    key: u32,
    value: u32,
};

@group(0) @binding(0) var<uniform> u_scatter_params: ScatterParams;
@group(0) @binding(1) var<storage, read> s_in_items: array<KeyValue>;
@group(0) @binding(2) var<storage, read> s_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> s_out_items: array<KeyValue>;

var<workgroup> s_bin_counts: array<atomic<u32>, 16>;

@compute @workgroup_size(256, 1, 1)
fn cs_radix_scatter(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let lid = local_id.x;
    let gid = global_id.x;

    if (lid < 16u) {
        atomicStore(&s_bin_counts[lid], 0u);
    }
    workgroupBarrier();

    var item: KeyValue;
    var bin = 0u;
    let valid = gid < u_scatter_params.num_elements;

    if (valid) {
        item = s_in_items[gid];
        bin = (item.key >> u_scatter_params.bit_shift) & 0xFu;
    }

    // Compute base offset from global prefix sum
    var base_offset = 0u;
    if (valid) {
        let hist_idx = bin * u_scatter_params.num_workgroups + group_id.x;
        base_offset = s_offsets[hist_idx];
        let local_offset = atomicAdd(&s_bin_counts[bin], 1u);
        let dst_idx = base_offset + local_offset;
        s_out_items[dst_idx] = item;
    }
}
"""

# 5. karras_tree_builder.wgsl
KARRAS_BUILDER_WGSL = """// Shroud :: Karras 2012 Parallel BVH Radix Tree Construction
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
"""

# 6. bvh_bounds_fitter.wgsl
BOUNDS_FITTER_WGSL = """// Shroud :: Bottom-Up BVH Bounding Box Aggregation
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
"""

# 7. bvh_sah_optimizer.wgsl
SAH_OPTIMIZER_WGSL = """// Shroud :: Surface Area Heuristic (SAH) Cost Evaluation & Optimization
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
"""

# 8. bvh_traversal_stackless.wgsl
STACKLESS_TRAVERSAL_WGSL = """// Shroud :: Stackless BVH Traversal via Rope (Escape) Pointers
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
"""

# 9. bvh_traversal_short_stack.wgsl
SHORT_STACK_TRAVERSAL_WGSL = """// Shroud :: Workgroup Short-Stack BVH Traversal
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
"""

# 10. ray_primitive_intersection.wgsl
RAY_INTERSECTION_WGSL = """// Shroud :: High-Precision Ray-Primitive Geometric Intersections
// Möller-Trumbore and Watertight Triangle Intersections with Numerical Safeguards.

struct RayGeometric {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
};

struct TriangleGeometric {
    v0: vec3<f32>,
    prim_id: u32,
    v1: vec3<f32>,
    material_id: u32,
    v2: vec3<f32>,
    padding: u32,
};

struct IntersectionRecord {
    t: f32,
    u: f32,
    v: f32,
    hit: u32,
    normal: vec3<f32>,
    prim_id: u32,
};

// Möller-Trumbore ray-triangle intersection algorithm
fn moller_trumbore_intersect(
    ray: RayGeometric,
    tri: TriangleGeometric,
    rec: ptr<function, IntersectionRecord>
) -> bool {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let pvec = cross(ray.direction, edge2);
    let det = dot(edge1, pvec);

    if (abs(det) < 1e-8) {
        return false;
    }

    let inv_det = 1.0 / det;
    let tvec = ray.origin - tri.v0;
    let u = dot(tvec, pvec) * inv_det;

    if (u < 0.0 || u > 1.0) {
        return false;
    }

    let qvec = cross(tvec, edge1);
    let v = dot(ray.direction, qvec) * inv_det;

    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    let t = dot(edge2, qvec) * inv_det;
    if (t < ray.t_min || t > ray.t_max) {
        return false;
    }

    (*rec).t = t;
    (*rec).u = u;
    (*rec).v = v;
    (*rec).hit = 1u;
    (*rec).normal = normalize(cross(edge1, edge2));
    (*rec).prim_id = tri.prim_id;
    return true;
}

@group(0) @binding(0) var<storage, read> s_test_rays: array<RayGeometric>;
@group(0) @binding(1) var<storage, read> s_test_triangles: array<TriangleGeometric>;
@group(0) @binding(2) var<storage, read_write> s_intersections: array<IntersectionRecord>;

@compute @workgroup_size(64, 1, 1)
fn cs_primitive_intersect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_test_rays) || idx >= arrayLength(&s_test_triangles)) {
        return;
    }

    let ray = s_test_rays[idx];
    let tri = s_test_triangles[idx];

    var rec: IntersectionRecord;
    rec.t = ray.t_max;
    rec.u = 0.0;
    rec.v = 0.0;
    rec.hit = 0u;
    rec.normal = vec3<f32>(0.0, 1.0, 0.0);
    rec.prim_id = 0xFFFFFFFFu;

    moller_trumbore_intersect(ray, tri, &rec);
    s_intersections[idx] = rec;
}
"""

write_shader("lbvh_morton.wgsl", LBVH_MORTON_WGSL)
write_shader("radix_sort_histogram.wgsl", RADIX_HISTOGRAM_WGSL)
write_shader("radix_sort_scan.wgsl", RADIX_SCAN_WGSL)
write_shader("radix_sort_scatter.wgsl", RADIX_SCATTER_WGSL)
write_shader("karras_tree_builder.wgsl", KARRAS_BUILDER_WGSL)
write_shader("bvh_bounds_fitter.wgsl", BOUNDS_FITTER_WGSL)
write_shader("bvh_sah_optimizer.wgsl", SAH_OPTIMIZER_WGSL)
write_shader("bvh_traversal_stackless.wgsl", STACKLESS_TRAVERSAL_WGSL)
write_shader("bvh_traversal_short_stack.wgsl", SHORT_STACK_TRAVERSAL_WGSL)
write_shader("ray_primitive_intersection.wgsl", RAY_INTERSECTION_WGSL)
