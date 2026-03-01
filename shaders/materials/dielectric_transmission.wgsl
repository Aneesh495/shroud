// Shroud :: Dielectric Microfacet Transmission & Refraction (Walter et al. 2007)
// Snell's law refraction, Total Internal Reflection (TIR), and rough transmission.

const PI: f32 = 3.14159265358979323846;

struct DielectricParams {
    ior_inside: f32,
    ior_outside: f32,
    roughness: f32,
    dispersion_cauchy: f32,
};

fn fresnel_dielectric(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    var c_i = clamp(cos_theta_i, -1.0, 1.0);
    var ei = eta_i;
    var et = eta_t;

    if (c_i < 0.0) {
        c_i = -c_i;
        let tmp = ei;
        ei = et;
        et = tmp;
    }

    let sin_t = (ei / et) * sqrt(max(0.0, 1.0 - c_i * c_i));
    if (sin_t >= 1.0) {
        return 1.0; // Total internal reflection
    }

    let c_t = sqrt(max(0.0, 1.0 - sin_t * sin_t));
    let r_parl = ((et * c_i) - (ei * c_t)) / ((et * c_i) + (ei * c_t));
    let r_perp = ((ei * c_i) - (et * c_t)) / ((ei * c_i) + (et * c_t));

    return 0.5 * (r_parl * r_parl + r_perp * r_perp);
}

fn refract_vector(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_theta = dot(-v, n);
    let sin2_theta = max(0.0, 1.0 - cos_theta * cos_theta);
    let sin2_t = eta * eta * sin2_theta;
    if (sin2_t >= 1.0) {
        return reflect(-v, n); // TIR fallback
    }
    let cos_t = sqrt(max(0.0, 1.0 - sin2_t));
    return eta * -v + (eta * cos_theta - cos_t) * n;
}

@group(0) @binding(0) var<uniform> u_dielectric: DielectricParams;
@group(0) @binding(1) var<storage, read> s_incident_rays: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_refracted_rays: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_dielectric_refract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_incident_rays)) {
        return;
    }

    let v = s_incident_rays[idx].xyz;
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let eta = u_dielectric.ior_outside / u_dielectric.ior_inside;

    let f = fresnel_dielectric(dot(v, n), u_dielectric.ior_outside, u_dielectric.ior_inside);
    let wt = refract_vector(v, n, eta);

    s_refracted_rays[idx] = vec4<f32>(wt, f);
}
