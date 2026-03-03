// Shroud :: Procedural 3D Fractal Brownian Motion (fBm) Fog Generator
// 3D Perlin noise synthesis with 4 octaves and analytical gradients.

fn hash33(p_in: vec3<f32>) -> vec3<f32> {
    var p = vec3<f32>(
        dot(p_in, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p_in, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p_in, vec3<f32>(113.5, 271.9, 124.6))
    );
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

fn gradient_noise(p: vec3<f32>) -> f32 {
    let pi = floor(p);
    let pf = fract(p);
    let w = pf * pf * (3.0 - 2.0 * pf);

    let d000 = dot(hash33(pi + vec3<f32>(0.0, 0.0, 0.0)), pf - vec3<f32>(0.0, 0.0, 0.0));
    let d100 = dot(hash33(pi + vec3<f32>(1.0, 0.0, 0.0)), pf - vec3<f32>(1.0, 0.0, 0.0));
    let d010 = dot(hash33(pi + vec3<f32>(0.0, 1.0, 0.0)), pf - vec3<f32>(0.0, 1.0, 0.0));
    let d110 = dot(hash33(pi + vec3<f32>(1.0, 1.0, 0.0)), pf - vec3<f32>(1.0, 1.0, 0.0));
    let d001 = dot(hash33(pi + vec3<f32>(0.0, 0.0, 1.0)), pf - vec3<f32>(0.0, 0.0, 1.0));
    let d101 = dot(hash33(pi + vec3<f32>(1.0, 0.0, 1.0)), pf - vec3<f32>(1.0, 0.0, 1.0));
    let d011 = dot(hash33(pi + vec3<f32>(0.0, 1.0, 1.0)), pf - vec3<f32>(0.0, 1.0, 1.0));
    let d111 = dot(hash33(pi + vec3<f32>(1.0, 1.0, 1.0)), pf - vec3<f32>(1.0, 1.0, 1.0));

    let dx00 = mix(d000, d100, w.x);
    let dx10 = mix(d010, d110, w.x);
    let dx01 = mix(d001, d101, w.x);
    let dx11 = mix(d011, d111, w.x);

    let dxy0 = mix(dx00, dx10, w.y);
    let dxy1 = mix(dx01, dx11, w.y);

    return mix(dxy0, dxy1, w.z);
}

fn fbm_3d(pos: vec3<f32>, octaves: u32) -> f32 {
    var p = pos;
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;

    for (var i = 0u; i < octaves; i = i + 1u) {
        value += amplitude * gradient_noise(p * frequency);
        p = p * 2.02;
        amplitude *= 0.5;
    }
    return value;
}

@group(0) @binding(0) var<storage, read> s_sample_coords: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> s_density_grid: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_generate_fbm_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_sample_coords)) {
        return;
    }

    let p = s_sample_coords[idx].xyz;
    let n = fbm_3d(p, 4u);
    let density = clamp(0.5 + 0.5 * n, 0.0, 1.0);
    s_density_grid[idx] = density;
}
