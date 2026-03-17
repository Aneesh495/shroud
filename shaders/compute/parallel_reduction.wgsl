// Shroud :: High-Throughput Workgroup Tree Reduction
// Computes sum and maximum across large global arrays with workgroup coalescing.

@group(0) @binding(0) var<storage, read> s_input_elements: array<f32>;
@group(0) @binding(1) var<storage, read_write> s_workgroup_sums: array<f32>;

var<workgroup> s_tree: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn cs_parallel_sum_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let idx = gid.x;

    let n = arrayLength(&s_input_elements);
    if (idx < n) {
        s_tree[tid] = s_input_elements[idx];
    } else {
        s_tree[tid] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction in shared memory
    var s = 128u;
    while (s > 0u) {
        if (tid < s) {
            s_tree[tid] += s_tree[tid + s];
        }
        s = s >> 1u;
        workgroupBarrier();
    }

    if (tid == 0u) {
        s_workgroup_sums[wid.x] = s_tree[0];
    }
}
