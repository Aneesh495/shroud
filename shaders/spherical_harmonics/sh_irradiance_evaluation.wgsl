// Shroud :: Real Spherical Harmonics L0-L3 Irradiance Evaluation
// Implements Ramamoorthi & Hanrahan 2001 analytical cosine convolution.

const PI: f32 = 3.14159265358979323846;

// Analytical cosine convolution coefficients A_l
const A0: f32 = 3.14159265;         // pi
const A1: f32 = 2.09439510;         // 2 * pi / 3
const A2: f32 = 0.78539816;         // pi / 4
const A3: f32 = 0.0;                // 0

struct ShCoefficientsL3 {
    c: array<vec3<f32>, 16>,
};

// Evaluates 16 real spherical harmonics basis functions Y_l^m(n) for bands l = 0, 1, 2, 3
fn evaluate_sh_basis_l3(n: vec3<f32>) -> array<f32, 16> {
    var y: array<f32, 16>;
    let x = n.x;
    let y_coord = n.y;
    let z = n.z;

    // Band 0 (l = 0, m = 0)
    y[0] = 0.28209479;

    // Band 1 (l = 1, m = -1, 0, 1)
    y[1] = 0.48860251 * y_coord;
    y[2] = 0.48860251 * z;
    y[3] = 0.48860251 * x;

    // Band 2 (l = 2, m = -2, -1, 0, 1, 2)
    y[4] = 1.09254843 * x * y_coord;
    y[5] = 1.09254843 * y_coord * z;
    y[6] = 0.31539156 * (3.0 * z * z - 1.0);
    y[7] = 1.09254843 * x * z;
    y[8] = 0.54627421 * (x * x - y_coord * y_coord);

    // Band 3 (l = 3, m = -3, -2, -1, 0, 1, 2, 3)
    y[9] = 0.59004358 * y_coord * (3.0 * x * x - y_coord * y_coord);
    y[10] = 2.89061144 * x * y_coord * z;
    y[11] = 0.45704579 * y_coord * (5.0 * z * z - 1.0);
    y[12] = 0.37317633 * z * (5.0 * z * z - 3.0);
    y[13] = 0.45704579 * x * (5.0 * z * z - 1.0);
    y[14] = 1.44530572 * z * (x * x - y_coord * y_coord);
    y[15] = 0.59004358 * x * (x * x - 3.0 * y_coord * y_coord);

    return y;
}

// Reconstructs incident diffuse irradiance E(n) = sum_{l,m} A_l * c_{l,m} * Y_l^m(n)
fn reconstruct_diffuse_irradiance(n: vec3<f32>, sh: ShCoefficientsL3) -> vec3<f32> {
    let y = evaluate_sh_basis_l3(n);
    var irradiance = vec3<f32>(0.0);

    // Band 0
    irradiance += A0 * sh.c[0] * y[0];

    // Band 1
    for (var i = 1u; i <= 3u; i = i + 1u) {
        irradiance += A1 * sh.c[i] * y[i];
    }

    // Band 2
    for (var i = 4u; i <= 8u; i = i + 1u) {
        irradiance += A2 * sh.c[i] * y[i];
    }

    // Band 3 (A3 is 0 for cosine lobe, but included for general radiance)
    return max(vec3<f32>(0.0), irradiance);
}

@group(0) @binding(0) var<uniform> u_sh_env: ShCoefficientsL3;
@group(0) @binding(1) var<storage, read> s_surface_normals: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_irradiance_output: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_evaluate_sh_irradiance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_surface_normals)) {
        return;
    }
    let n = normalize(s_surface_normals[idx].xyz);
    let e = reconstruct_diffuse_irradiance(n, u_sh_env);
    s_irradiance_output[idx] = vec4<f32>(e, 1.0);
}
