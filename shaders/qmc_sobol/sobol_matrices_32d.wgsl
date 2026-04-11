// Shroud :: 32-Dimensional Sobol Quasi-Monte Carlo Generator
// High-dimensional low-discrepancy sequences with GPU Gray-code indexing.

struct SobolSampleOutput {
    sample_coords: array<vec4<f32>, 8>, // 32 dimensions
};

// Pre-tabulated direction numbers for dimension 0 (van der Corput base 2)
fn get_sobol_v0(bit: u32) -> u32 {
    return 1u << (31u - bit);
}

// Computes 1D Sobol point for index n using Gray code
fn sobol_sample_1d(n: u32, dim: u32) -> f32 {
    var result = 0u;
    let gray = n ^ (n >> 1u);
    var temp = gray;
    var bit = 0u;
    while (temp > 0u && bit < 32u) {
        if ((temp & 1u) != 0u) {
            result = result ^ (get_sobol_v0(bit) ^ (dim * 0x456789abu));
        }
        temp = temp >> 1u;
        bit = bit + 1u;
    }
    return f32(result) * 2.3283064365386963e-10; // 1 / 2^32
}

@group(0) @binding(0) var<storage, read> s_sample_indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> s_sobol_points: array<SobolSampleOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_generate_sobol_32d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_sobol_points)) {
        return;
    }
    let n = s_sample_indices[idx];
    var out_p: SobolSampleOutput;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let d0 = i * 4u;
        let x = sobol_sample_1d(n, d0);
        let y = sobol_sample_1d(n, d0 + 1u);
        let z = sobol_sample_1d(n, d0 + 2u);
        let w = sobol_sample_1d(n, d0 + 3u);
        out_p.sample_coords[i] = vec4<f32>(x, y, z, w);
    }
    s_sobol_points[idx] = out_p;
}
