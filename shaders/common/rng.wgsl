// Shroud :: Stateless & State-Based GPU Pseudorandom Generators (PCG32 / Hash)
// Module: shaders/common/rng.wgsl

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(seed: ptr<function, u32>) -> f32 {
    *seed = pcg_hash(*seed);
    return f32(*seed) * 2.3283064365386963e-10;
}

@group(0) @binding(0) var<storage, read_write> s_rng_samples: array<vec2<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&s_rng_samples)) {
        var state = idx * 1664525u + 1013904223u;
        let u0 = rand_f32(&state);
        let u1 = rand_f32(&state);
        s_rng_samples[idx] = vec2<f32>(u0, u1);
    }
}
