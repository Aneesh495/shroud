// Shroud :: Workgroup Blelloch Exclusive Prefix Sum Scan
// Bank-conflict free parallel scan using local workgroup shared memory.

const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> s_in_array: array<u32>;
@group(0) @binding(1) var<storage, read_write> s_out_prefix_sum: array<u32>;

var<workgroup> s_shared_scan: array<u32, 512>;

@compute @workgroup_size(256, 1, 1)
fn cs_blelloch_scan(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = lid.x;
    let idx_a = 2u * tid;
    let idx_b = 2u * tid + 1u;

    let n = arrayLength(&s_in_array);

    if (idx_a < n) {
        s_shared_scan[idx_a] = s_in_array[idx_a];
    } else {
        s_shared_scan[idx_a] = 0u;
    }

    if (idx_b < n) {
        s_shared_scan[idx_b] = s_in_array[idx_b];
    } else {
        s_shared_scan[idx_b] = 0u;
    }
    workgroupBarrier();

    // Up-sweep (reduce)
    var stride = 1u;
    while (stride < 512u) {
        let index = (tid + 1u) * stride * 2u - 1u;
        if (index < 512u) {
            s_shared_scan[index] += s_shared_scan[index - stride];
        }
        stride = stride * 2u;
        workgroupBarrier();
    }

    // Set root to 0 for exclusive scan
    if (tid == 0u) {
        s_shared_scan[511u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep
    stride = 256u;
    while (stride > 0u) {
        let index = (tid + 1u) * stride * 2u - 1u;
        if (index < 512u) {
            let temp = s_shared_scan[index - stride];
            s_shared_scan[index - stride] = s_shared_scan[index];
            s_shared_scan[index] += temp;
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    if (idx_a < n) {
        s_out_prefix_sum[idx_a] = s_shared_scan[idx_a];
    }
    if (idx_b < n) {
        s_out_prefix_sum[idx_b] = s_shared_scan[idx_b];
    }
}
