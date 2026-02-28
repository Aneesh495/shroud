// Shroud :: GPU Radix Sort Scatter Pass
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
