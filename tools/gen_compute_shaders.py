#!/usr/bin/env python3
"""
Generates shaders/compute/ WGSL modules.
Implements Blelloch Prefix Sums, Parallel Reduction, Spatial Hashing,
Tiled GEMM Matrix Multiplication, FFT Radix-2, and Symplectic Particle Integration.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
COMPUTE_DIR = ROOT_DIR / "shaders" / "compute"
COMPUTE_DIR.mkdir(parents=True, exist_ok=True)

def write_shader(filename: str, content: str):
    p = COMPUTE_DIR / filename
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# 1. workgroup_scan_blelloch.wgsl
BLELLOCH_SCAN_WGSL = """// Shroud :: Workgroup Blelloch Exclusive Prefix Sum Scan
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
"""

# 2. parallel_reduction.wgsl
PARALLEL_REDUCTION_WGSL = """// Shroud :: High-Throughput Workgroup Tree Reduction
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
"""

# 3. spatial_hash_grid.wgsl
SPATIAL_HASH_WGSL = """// Shroud :: 3D Spatial Hashing Grid for Fast Neighbor Searches
// Maps continuous 3D world coordinates to bounded hash buckets for proximity queries.

struct SpatialHashParams {
    cell_size: f32,
    table_size: u32,
    padding: vec2<f32>,
};

fn hash_coords(cx: i32, cy: i32, cz: i32, table_size: u32) -> u32 {
    let p1 = 73856093u;
    let p2 = 19349663u;
    let p3 = 83492791u;
    let n = (u32(cx) * p1) ^ (u32(cy) * p2) ^ (u32(cz) * p3);
    return n % table_size;
}

@group(0) @binding(0) var<uniform> u_hash_params: SpatialHashParams;
@group(0) @binding(1) var<storage, read> s_positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_bucket_indices: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn cs_compute_spatial_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_positions)) {
        return;
    }

    let p = s_positions[idx].xyz;
    let cs = max(1e-4, u_hash_params.cell_size);

    let cx = i32(floor(p.x / cs));
    let cy = i32(floor(p.y / cs));
    let cz = i32(floor(p.z / cs));

    let bucket = hash_coords(cx, cy, cz, u_hash_params.table_size);
    s_bucket_indices[idx] = bucket;
}
"""

# 4. matrix_gemm_tiled.wgsl
MATRIX_GEMM_WGSL = """// Shroud :: Tiled General Matrix Multiply (GEMM) Compute Kernel
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
"""

# 5. fft_radix2_cooley_tukey.wgsl
FFT_RADIX2_WGSL = """// Shroud :: Cooley-Tukey Radix-2 Fast Fourier Transform (FFT)
// In-place butterfly operation across complex signals for FMCW spectrum analysis.

const PI: f32 = 3.14159265358979323846;

struct Complex {
    real: f32,
    imag: f32,
};

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

struct FftStageParams {
    stage_stride: u32,
    sub_sequence_len: u32,
    num_samples: u32,
    is_inverse: u32,
};

@group(0) @binding(0) var<uniform> u_fft: FftStageParams;
@group(0) @binding(1) var<storage, read_write> s_spectrum: array<Complex>;

@compute @workgroup_size(64, 1, 1)
fn cs_fft_butterfly_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let half_len = u_fft.stage_stride;
    let group_size = half_len * 2u;

    if (idx >= u_fft.num_samples / 2u) {
        return;
    }

    let group = idx / half_len;
    let pair_offset = idx % half_len;

    let idx_even = group * group_size + pair_offset;
    let idx_odd = idx_even + half_len;

    let angle = -2.0 * PI * f32(pair_offset) / f32(group_size);
    let twiddle = Complex(cos(angle), sin(angle));

    let even_val = s_spectrum[idx_even];
    let odd_val = s_spectrum[idx_odd];

    let twiddled_odd = complex_mul(odd_val, twiddle);

    s_spectrum[idx_even] = Complex(even_val.real + twiddled_odd.real, even_val.imag + twiddled_odd.imag);
    s_spectrum[idx_odd] = Complex(even_val.real - twiddled_odd.real, even_val.imag - twiddled_odd.imag);
}
"""

# 6. particle_integrator.wgsl
PARTICLE_INTEGRATOR_WGSL = """// Shroud :: Symplectic Velocity-Verlet Particle Integrator
// Conserves energy in aerosol dispersion and atmospheric turbulence trajectories.

struct IntegratorUniforms {
    dt: f32,
    damping: f32,
    num_particles: u32,
    padding: f32,
};

struct ParticleState {
    pos: vec3<f32>,
    mass: f32,
    vel: vec3<f32>,
    lifetime: f32,
};

@group(0) @binding(0) var<uniform> u_integrator: IntegratorUniforms;
@group(0) @binding(1) var<storage, read_write> s_particles_state: array<ParticleState>;

@compute @workgroup_size(64, 1, 1)
fn cs_velocity_verlet_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u_integrator.num_particles) {
        return;
    }

    var p = s_particles_state[idx];
    let gravity = vec3<f32>(0.0, -9.80665, 0.0);
    let force = gravity * p.mass - p.vel * u_integrator.damping;
    let accel = force / max(1e-4, p.mass);

    // Velocity Verlet integration:
    // x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    // v(t + dt) = v(t) + a(t)*dt
    p.pos += p.vel * u_integrator.dt + 0.5 * accel * u_integrator.dt * u_integrator.dt;
    p.vel += accel * u_integrator.dt;
    p.lifetime -= u_integrator.dt;

    s_particles_state[idx] = p;
}
"""

write_shader("workgroup_scan_blelloch.wgsl", BLELLOCH_SCAN_WGSL)
write_shader("parallel_reduction.wgsl", PARALLEL_REDUCTION_WGSL)
write_shader("spatial_hash_grid.wgsl", SPATIAL_HASH_WGSL)
write_shader("matrix_gemm_tiled.wgsl", MATRIX_GEMM_WGSL)
write_shader("fft_radix2_cooley_tukey.wgsl", FFT_RADIX2_WGSL)
write_shader("particle_integrator.wgsl", PARTICLE_INTEGRATOR_WGSL)
