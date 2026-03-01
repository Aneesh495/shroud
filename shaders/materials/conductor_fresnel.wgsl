// Shroud :: Complex Conductor Fresnel Reflection
// Computes reflectance for metals with complex refractive index (eta, kappa).

struct ConductorOpticalConstants {
    eta: vec3<f32>,    // Real part of index of refraction
    padding0: f32,
    kappa: vec3<f32>,  // Imaginary extinction coefficient
    padding1: f32,
};

fn fresnel_conductor_channel(cos_theta_i: f32, eta: f32, k: f32) -> f32 {
    let cos2 = cos_theta_i * cos_theta_i;
    let sin2 = max(0.0, 1.0 - cos2);
    let eta2 = eta * eta;
    let k2 = k * k;

    let t0 = eta2 - k2 - sin2;
    let a2_plus_b2 = sqrt(max(0.0, t0 * t0 + 4.0 * eta2 * k2));
    let t1 = a2_plus_b2 + cos2;
    let a = sqrt(max(0.0, 0.5 * (a2_plus_b2 + t0)));
    let t2 = 2.0 * cos_theta_i * a;
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = cos2 * a2_plus_b2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    return 0.5 * (rp + rs);
}

fn evaluate_conductor_fresnel(cos_theta_i: f32, eta: vec3<f32>, kappa: vec3<f32>) -> vec3<f32> {
    let c = clamp(abs(cos_theta_i), 0.0, 1.0);
    return vec3<f32>(
        fresnel_conductor_channel(c, eta.r, kappa.r),
        fresnel_conductor_channel(c, eta.g, kappa.g),
        fresnel_conductor_channel(c, eta.b, kappa.b)
    );
}

@group(0) @binding(0) var<uniform> u_metal: ConductorOpticalConstants;
@group(0) @binding(1) var<storage, read> s_cos_angles: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_fresnel_spectrum: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_conductor_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_cos_angles)) {
        return;
    }
    let cos_theta = s_cos_angles[idx];
    let refl = evaluate_conductor_fresnel(cos_theta, u_metal.eta, u_metal.kappa);
    s_fresnel_spectrum[idx] = vec4<f32>(refl, 1.0);
}
