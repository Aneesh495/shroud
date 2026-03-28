// Shroud :: Material Model: Portland Cement Concrete Road Pavement
// Identifier: mat_concrete_pavement
// Physical constants, spectral response tables, and Cook-Torrance BRDF evaluation.

/*
 * ASTM / ISO Standard Physical Characteristics for Portland Cement Concrete Road Pavement:
 * - Base Hemispherical Reflectance (Visible): 0.3500
 * - Near-Infrared Reflectance (905nm LiDAR): 0.1000
 * - Surface Micro-Roughness parameter (alpha): 0.7200
 * - Real Index of Refraction (n): 1.6000
 * - Long-Wave Infrared Emissivity (8-14 um): 0.9000
 * - Thermal Inertia (J / (m^2 * K * s^0.5)): 1000.0
 * - Thermal Conductivity (W / (m * K)): 1.40
 *
 * Numerical Formulation:
 * The bidirectional reflectance distribution function (BRDF) f_r(omega_i, omega_o)
 * decomposes into a microfacet specular reflection term and a volumetric diffuse term:
 * f_r(omega_i, omega_o) = f_{spec} + f_{diff}
 * where f_{spec} = (D(h) * G(l, v) * F(v, h)) / (4 * (n . l) * (n . v))
 * and f_{diff} = (1 - F) * (rho / pi) * (n . l)
 */


const PI: f32 = 3.14159265358979323846;
const INV_PI: f32 = 0.31830988618379067154;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

const MAT_BASE_ALBEDO_VIS: f32 = 0.3500;
const MAT_ALBEDO_NIR_905: f32 = 0.1000;
const MAT_ROUGHNESS: f32 = 0.7200;
const MAT_INDEX_REFRACTION: f32 = 1.6000;
const MAT_LWIR_EMISSIVITY: f32 = 0.9000;
const MAT_THERMAL_INERTIA: f32 = 1000.0;
const MAT_THERMAL_CONDUCTIVITY: f32 = 1.40;

struct MaterialQuery {
    incident_dir: vec3<f32>,
    surface_temp_k: f32,
    view_dir: vec3<f32>,
    wavelength_nm: f32,
    normal: vec3<f32>,
    padding: f32,
};

struct MaterialEvaluationResult {
    brdf_value: vec3<f32>,
    pdf: f32,
    thermal_radiance: f32,
    fresnel_reflectance: f32,
    padding: vec2<f32>,
};

fn evaluate_spectral_reflectance(wavelength_nm: f32) -> f32 {
    if (wavelength_nm < 400.0) {
        return MAT_BASE_ALBEDO_VIS * 0.85;
    } else if (wavelength_nm <= 700.0) {
        let t = (wavelength_nm - 400.0) / 300.0;
        return mix(MAT_BASE_ALBEDO_VIS * 0.9, MAT_BASE_ALBEDO_VIS * 1.1, t);
    } else if (wavelength_nm <= 1000.0) {
        let t = (wavelength_nm - 700.0) / 300.0;
        return mix(MAT_BASE_ALBEDO_VIS * 1.1, MAT_ALBEDO_NIR_905, t);
    } else {
        return MAT_ALBEDO_NIR_905 * 0.9;
    }
}

fn fresnel_dielectric(cos_theta: f32, eta: f32) -> f32 {
    let c = clamp(abs(cos_theta), 0.0, 1.0);
    let g2 = eta * eta - 1.0 + c * c;
    if (g2 <= 0.0) {
        return 1.0;
    }
    let g = sqrt(g2);
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d + 1e-7);
}

fn smith_shadowing(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt((-n_dot_v * alpha2 + n_dot_v) * n_dot_v + alpha2);
    let lambda_l = n_dot_v * sqrt((-n_dot_l * alpha2 + n_dot_l) * n_dot_l + alpha2);
    return 2.0 * n_dot_l * n_dot_v / max(1e-7, lambda_v + lambda_l);
}

fn evaluate_thermal_emission(temp_k: f32) -> f32 {
    return MAT_LWIR_EMISSIVITY * (STEFAN_BOLTZMANN / PI) * pow(max(10.0, temp_k), 4.0);
}

@group(0) @binding(0) var<storage, read> s_queries: array<MaterialQuery>;
@group(0) @binding(1) var<storage, read_write> s_results: array<MaterialEvaluationResult>;

@compute @workgroup_size(64, 1, 1)
fn cs_eval_material_concrete_pavement(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_results) || idx >= arrayLength(&s_queries)) {
        return;
    }
    let q = s_queries[idx];
    let n = normalize(q.normal);
    let l = normalize(q.incident_dir);
    let v = normalize(q.view_dir);

    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));
    let h = normalize(v + l);
    let n_dot_h = max(0.0, dot(n, h));
    let v_dot_h = max(0.0, dot(v, h));

    let refl_spec = evaluate_spectral_reflectance(q.wavelength_nm);
    let f = fresnel_dielectric(v_dot_h, MAT_INDEX_REFRACTION);
    let d = ggx_distribution(n_dot_h, MAT_ROUGHNESS);
    let g = smith_shadowing(n_dot_l, n_dot_v, MAT_ROUGHNESS);

    let specular = (d * g * f) / max(1e-6, 4.0 * n_dot_l * n_dot_v);
    let diffuse = (refl_spec * (1.0 - f) * INV_PI) * n_dot_l;

    var res: MaterialEvaluationResult;
    res.brdf_value = vec3<f32>(diffuse + specular);
    res.pdf = d * n_dot_h / max(1e-6, 4.0 * v_dot_h);
    res.thermal_radiance = evaluate_thermal_emission(q.surface_temp_k);
    res.fresnel_reflectance = f;
    res.padding = vec2<f32>(0.0, 0.0);

    s_results[idx] = res;
}
