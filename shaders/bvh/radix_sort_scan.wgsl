// Shroud :: GPU Prefix Sum Scan for Radix Sort Global Offsets
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
