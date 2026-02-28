// Shroud :: GPU Radix Sort Local Histogram Pass
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
