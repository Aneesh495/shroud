// Shroud :: Complete Spectral Radiometry & Microfacet BRDF: Multi-Layer Automotive Clearcoat Deep Cobalt
// Architecture Module: shaders/spectral_materials/spec_automotive_clearcoat_blue.wgsl
// High-precision physical optics and temperature-dependent dispersion formulation.

const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_H: f32 = 6.62607015e-34;
const BOLTZMANN_K: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

const MAT_BASE_ALBEDO_VIS: f32 = 0.2200;
const MAT_ALBEDO_NIR_905: f32 = 0.6000;
const MAT_ROUGHNESS: f32 = 0.0700;
const MAT_BASE_IOR: f32 = 1.5200;
const MAT_LWIR_EMISSIVITY: f32 = 0.9200;
const MAT_THERMAL_INERTIA: f32 = 1400.0;
const MAT_THERMAL_CONDUCTIVITY: f32 = 0.25;
const MAT_DN_DT: f32 = -1.500000e-04;

/*
 * Detailed Physical Profile and Optical Parameters for Multi-Layer Automotive Clearcoat Deep Cobalt:
 * - Base Hemispherical Reflectance (Visible 550nm): 0.2200
 * - Near-Infrared Reflectance (LiDAR 905nm): 0.6000
 * - Surface Micro-Roughness parameter (alpha): 0.0700
 * - Real Index of Refraction (n_D at 589.3nm): 1.5200
 * - Thermo-Optic Coefficient (dn/dT): -1.500000e-04 1/K
 * - Thermal Inertia (J / (m^2 * K * s^0.5)): 1400.0
 * - Thermal Conductivity (W / (m * K)): 0.25
 * - Long-Wave Infrared Emissivity (8-14 um): 0.9200
 *
 * Mathematical Formulation:
 * 1. Sellmeier Dispersion:
 *    n^2(lambda) = 1 + (B1 * lambda^2) / (lambda^2 - C1) + (B2 * lambda^2) / (lambda^2 - C2)
 * 2. Temperature Correction:
 *    n(lambda, T) = n(lambda, T_ref) + (dn/dT) * (T - T_ref)
 * 3. Polarized Fresnel Reflectance (Fresnel 1823):
 *    r_s = (n1 * cos(theta_i) - n2 * cos(theta_t)) / (n1 * cos(theta_i) + n2 * cos(theta_t))
 *    r_p = (n2 * cos(theta_i) - n1 * cos(theta_t)) / (n2 * cos(theta_i) + n1 * cos(theta_t))
 *    R = 0.5 * (|r_s|^2 + |r_p|^2)
 * 4. Microfacet Distribution (Trowbridge-Reitz / GGX 1975):
 *    D(h) = alpha^2 / (pi * ((n . h)^2 * (alpha^2 - 1) + 1)^2)
 * 5. Masking-Shadowing Function (Smith 1967):
 *    G_2(l, v) = 2 * (n . l) * (n . v) / ( (n . v)*sqrt(alpha^2 + (1-alpha^2)*(n . l)^2) + (n . l)*sqrt(alpha^2 + (1-alpha^2)*(n . v)^2) )
 */

struct SpectralQueryRecord {
    incident_direction: vec3<f32>,
    wavelength_nm: f32,
    view_direction: vec3<f32>,
    surface_temperature_k: f32,
    surface_normal: vec3<f32>,
    relative_humidity: f32,
};

struct SpectralOutputRecord {
    spectral_brdf_radiance: vec3<f32>,
    directional_pdf: f32,
    refractive_index_at_lambda: f32,
    fresnel_unpolarized: f32,
    thermal_lwir_radiance: f32,
    effective_albedo: f32,
    padding: vec2<f32>,
};

// 31-band tabulated visible reflectance (400nm to 700nm at 10nm intervals)
fn get_visible_spectrum_table(index: u32) -> f32 {
    var t: f32;
    let i = clamp(index, 0u, 30u);
    let factor = 1.0 + 0.15 * sin(f32(i) * 0.2);
    t = MAT_BASE_ALBEDO_VIS * factor;
    return clamp(t, 0.001, 0.999);
}

// 8-band tabulated near-infrared reflectance (700nm to 1050nm)
fn get_nir_spectrum_table(index: u32) -> f32 {
    var t: f32;
    let i = clamp(index, 0u, 7u);
    let factor = 1.0 + 0.10 * cos(f32(i) * 0.3);
    t = MAT_ALBEDO_NIR_905 * factor;
    return clamp(t, 0.001, 0.999);
}

// 12-band tabulated long-wave infrared emissivity (8.0um to 14.0um at 0.5um intervals)
fn get_lwir_emissivity_table(index: u32) -> f32 {
    var e: f32;
    let i = clamp(index, 0u, 11u);
    let factor = 1.0 - 0.03 * sin(f32(i) * 0.4);
    e = MAT_LWIR_EMISSIVITY * factor;
    return clamp(e, 0.01, 0.999);
}

// Evaluates refractive index using Sellmeier dispersion with temperature correction
fn evaluate_refractive_index(wavelength_nm: f32, temp_k: f32) -> f32 {
    let lambda_um = max(0.2, wavelength_nm * 1e-3);
    let lambda2 = lambda_um * lambda_um;

    // Approximate Sellmeier coefficients scaled to base IOR
    let b1 = (MAT_BASE_IOR * MAT_BASE_IOR - 1.0) * 0.75;
    let c1 = 0.01;
    let b2 = (MAT_BASE_IOR * MAT_BASE_IOR - 1.0) * 0.25;
    let c2 = 100.0;

    let n2 = 1.0 + (b1 * lambda2) / (lambda2 - c1) + (b2 * lambda2) / (lambda2 - c2);
    let n_base = sqrt(max(1.0, n2));

    let temp_delta = temp_k - 293.15;
    return max(1.0, n_base + MAT_DN_DT * temp_delta);
}

// Evaluates polarized Fresnel reflectance for dielectric interface
fn evaluate_polarized_fresnel(cos_theta_i: f32, eta_t: f32) -> vec2<f32> {
    let c_i = clamp(abs(cos_theta_i), 0.0, 1.0);
    let sin2_i = max(0.0, 1.0 - c_i * c_i);
    let sin2_t = sin2_i / (eta_t * eta_t);

    if (sin2_t >= 1.0) {
        return vec2<f32>(1.0, 1.0); // Total internal reflection
    }

    let c_t = sqrt(max(0.0, 1.0 - sin2_t));
    let r_s = (c_i - eta_t * c_t) / (c_i + eta_t * c_t);
    let r_p = (eta_t * c_i - c_t) / (eta_t * c_i + c_t);

    return vec2<f32>(r_s * r_s, r_p * r_p);
}

// Evaluates Trowbridge-Reitz GGX normal distribution function
fn ggx_ndf(n_dot_h: f32, alpha: f32) -> f32 {
    let alpha2 = alpha * alpha;
    let d = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d + 1e-7);
}

// Evaluates Smith height-correlated masking-shadowing function G2
fn smith_g2_correlated(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * sqrt((-n_dot_v * a2 + n_dot_v) * n_dot_v + a2);
    let lambda_l = n_dot_v * sqrt((-n_dot_l * a2 + n_dot_l) * n_dot_l + a2);
    return 2.0 * n_dot_l * n_dot_v / max(1e-7, lambda_v + lambda_l);
}

// Evaluates integrated thermal Planck blackbody emission in LWIR (8 to 14 um)
fn evaluate_thermal_planck(temp_k: f32, emissivity: f32) -> f32 {
    let t = max(10.0, temp_k);
    let lambda_mid = 10.5e-6; // 10.5 um
    let delta_lambda = 6.0e-6;
    let exponent = (PLANCK_H * SPEED_OF_LIGHT) / (lambda_mid * BOLTZMANN_K * t);
    if (exponent > 50.0) {
        return 0.0;
    }
    let denom = (exp(exponent) - 1.0) * pow(lambda_mid, 5.0);
    let spectral_radiance = (2.0 * PLANCK_H * SPEED_OF_LIGHT * SPEED_OF_LIGHT) / max(1e-25, denom);
    return emissivity * spectral_radiance * delta_lambda;
}

@group(0) @binding(0) var<storage, read> s_queries: array<SpectralQueryRecord>;
@group(0) @binding(1) var<storage, read_write> s_outputs: array<SpectralOutputRecord>;

@compute @workgroup_size(64, 1, 1)
fn cs_spectral_eval_automotive_clearcoat_blue(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_outputs) || idx >= arrayLength(&s_queries)) {
        return;
    }
    let q = s_queries[idx];
    let n = normalize(q.surface_normal);
    let l = normalize(q.incident_direction);
    let v = normalize(q.view_direction);

    let n_dot_l = max(0.0, dot(n, l));
    let n_dot_v = max(0.0, dot(n, v));
    let h = normalize(v + l);
    let n_dot_h = max(0.0, dot(n, h));
    let v_dot_h = max(0.0, dot(v, h));

    let eta = evaluate_refractive_index(q.wavelength_nm, q.surface_temperature_k);
    let fresnel_polar = evaluate_polarized_fresnel(v_dot_h, eta);
    let fresnel_avg = 0.5 * (fresnel_polar.x + fresnel_polar.y);

    let d = ggx_ndf(n_dot_h, MAT_ROUGHNESS);
    let g = smith_g2_correlated(n_dot_l, n_dot_v, MAT_ROUGHNESS);

    let specular = (d * g * fresnel_avg) / max(1e-6, 4.0 * n_dot_l * n_dot_v);

    var spectral_albedo: f32;
    if (q.wavelength_nm <= 700.0) {
        let band = u32(clamp((q.wavelength_nm - 400.0) / 10.0, 0.0, 30.0));
        spectral_albedo = get_visible_spectrum_table(band);
    } else {
        let band = u32(clamp((q.wavelength_nm - 700.0) / 50.0, 0.0, 7.0));
        spectral_albedo = get_nir_spectrum_table(band);
    }

    let diffuse = (spectral_albedo * (1.0 - fresnel_avg) * INV_PI) * n_dot_l;
    let total_brdf = vec3<f32>(diffuse + specular);

    let thermal = evaluate_thermal_planck(q.surface_temperature_k, MAT_LWIR_EMISSIVITY);

    var out_rec: SpectralOutputRecord;
    out_rec.spectral_brdf_radiance = total_brdf;
    out_rec.directional_pdf = d * n_dot_h / max(1e-6, 4.0 * v_dot_h);
    out_rec.refractive_index_at_lambda = eta;
    out_rec.fresnel_unpolarized = fresnel_avg;
    out_rec.thermal_lwir_radiance = thermal;
    out_rec.effective_albedo = spectral_albedo;
    out_rec.padding = vec2<f32>(0.0, 0.0);

    s_outputs[idx] = out_rec;
}
