// Shroud :: Tiled General Matrix Multiply (GEMM) Compute Kernel
// 16x16 shared-memory tile caching for Kalman filter and calibration updates.

const TILE_SIZE: u32 = 16u;

struct GemmParams {
    dim_m: u32,
    dim_k: u32,
    dim_n: u32,
    padding: u32,
};

@group(0) @binding(0) var<uniform> u_gemm: GemmParams;
@group(0) @binding(1) var<storage, read> s_matrix_a: array<f32>; // M x K
@group(0) @binding(2) var<storage, read> s_matrix_b: array<f32>; // K x N
@group(0) @binding(3) var<storage, read_write> s_matrix_c: array<f32>; // M x N

var<workgroup> s_tile_a: array<array<f32, 16>, 16>;
var<workgroup> s_tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn cs_gemm_tiled(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;

    var sum = 0.0;
    let num_tiles = (u_gemm.dim_k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let tiled_k_a = t * TILE_SIZE + lid.x;
        let tiled_k_b = t * TILE_SIZE + lid.y;

        if (row < u_gemm.dim_m && tiled_k_a < u_gemm.dim_k) {
            s_tile_a[lid.y][lid.x] = s_matrix_a[row * u_gemm.dim_k + tiled_k_a];
        } else {
            s_tile_a[lid.y][lid.x] = 0.0;
        }

        if (col < u_gemm.dim_n && tiled_k_b < u_gemm.dim_k) {
            s_tile_b[lid.y][lid.x] = s_matrix_b[tiled_k_b * u_gemm.dim_n + col];
        } else {
            s_tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum += s_tile_a[lid.y][k] * s_tile_b[k][lid.x];
        }
        workgroupBarrier();
    }

    if (row < u_gemm.dim_m && col < u_gemm.dim_n) {
        s_matrix_c[row * u_gemm.dim_n + col] = sum;
    }
}
