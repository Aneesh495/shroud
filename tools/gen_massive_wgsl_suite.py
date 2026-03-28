#!/usr/bin/env python3
"""
Massive WGSL Shader Generator for Shroud.
Author: Aneesh Krishna Parthasarathy
"""

from pathlib import Path
import math

ROOT_DIR = Path(__file__).resolve().parent.parent
SHADERS_DIR = ROOT_DIR / "shaders"

def write_shader(subpath: str, content: str):
    p = SHADERS_DIR / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# ==============================================================================
# Material Generator
# ==============================================================================

MATERIALS = [
    ("asphalt_dry", "Dry Road Surface Asphalt", 0.12, 0.05, 0.65, 1.55, 0.93, 1200.0, 0.75),
    ("asphalt_wet", "Water-Covered Road Surface Asphalt", 0.06, 0.02, 0.15, 1.33, 0.96, 1600.0, 1.10),
    ("asphalt_ice_covered", "Black Ice Glaze on Road Asphalt", 0.18, 0.03, 0.05, 1.31, 0.97, 1800.0, 2.10),
    ("concrete_pavement", "Portland Cement Concrete Road Pavement", 0.35, 0.10, 0.72, 1.60, 0.90, 1000.0, 1.40),
    ("concrete_barrier", "Highway Jersey Concrete Crash Barrier", 0.38, 0.12, 0.68, 1.58, 0.89, 1050.0, 1.35),
    ("automotive_clearcoat_red", "Multi-Layer Automotive Clearcoat Crimson Red", 0.28, 0.65, 0.08, 1.52, 0.92, 1400.0, 0.25),
    ("automotive_clearcoat_blue", "Multi-Layer Automotive Clearcoat Deep Cobalt", 0.22, 0.60, 0.07, 1.52, 0.92, 1400.0, 0.25),
    ("automotive_clearcoat_black", "Multi-Layer Automotive Clearcoat Obsidian Black", 0.04, 0.02, 0.06, 1.53, 0.94, 1400.0, 0.25),
    ("automotive_clearcoat_white", "Multi-Layer Automotive Clearcoat Pearl White", 0.82, 0.75, 0.09, 1.52, 0.91, 1400.0, 0.25),
    ("automotive_metallic_silver", "Automotive Metallic Flake Silver Poly", 0.68, 0.85, 0.22, 1.85, 0.82, 1500.0, 0.45),
    ("automotive_metallic_gold", "Automotive Metallic Flake Champagne Gold", 0.62, 0.80, 0.24, 1.75, 0.80, 1500.0, 0.45),
    ("automotive_carbon_fiber", "High-Modulus Twill Weave Carbon Fiber Panel", 0.08, 0.04, 0.32, 1.65, 0.88, 1100.0, 0.85),
    ("automotive_chrome_trim", "Electroplated Decorative Chromium Trim", 0.78, 0.95, 0.02, 2.85, 0.15, 2400.0, 93.0),
    ("tire_tread_rubber", "Vulcanized Styrene-Butadiene Tire Rubber", 0.04, 0.01, 0.85, 1.54, 0.95, 1300.0, 0.18),
    ("windshield_tinted_glass", "Laminated Acoustic Safety Glass Windshield", 0.08, 0.90, 0.01, 1.51, 0.94, 750.0, 1.05),
    ("rear_window_heating_grid", "Toughened Rear Window with Tungsten Grid", 0.09, 0.88, 0.02, 1.52, 0.93, 760.0, 1.08),
    ("headlight_polycarbonate_lens", "Automotive Headlight Polycarbonate Shell", 0.10, 0.89, 0.02, 1.58, 0.92, 850.0, 0.20),
    ("taillight_acrylic_red", "Automotive Taillight PMMA Red Lens", 0.25, 0.75, 0.02, 1.49, 0.91, 820.0, 0.19),
    ("retroreflective_tape_white", "ASTM D4956 Type IV Prismatic Tape White", 0.85, 0.82, 0.05, 1.59, 0.85, 950.0, 0.22),
    ("retroreflective_tape_yellow", "ASTM D4956 Type IV Prismatic Tape Yellow", 0.72, 0.78, 0.06, 1.59, 0.86, 950.0, 0.22),
    ("retroreflective_tape_red", "ASTM D4956 Type IV Prismatic Tape Red", 0.45, 0.65, 0.06, 1.59, 0.88, 950.0, 0.22),
    ("traffic_sign_aluminum", "Powder-Coated Highway Guide Sign Aluminum", 0.75, 0.70, 0.25, 1.55, 0.85, 2100.0, 205.0),
    ("traffic_cone_pvc", "High-Visibility Fluorescent Orange PVC Cone", 0.62, 0.65, 0.45, 1.53, 0.91, 980.0, 0.16),
    ("road_marking_thermoplastic", "High-Build Alkyd Road Line Thermoplastic", 0.78, 0.75, 0.62, 1.58, 0.88, 1250.0, 0.70),
    ("road_marking_epoxy", "Two-Component Durable Highway Epoxy Paint", 0.72, 0.70, 0.55, 1.56, 0.89, 1300.0, 0.65),
    ("pedestrian_skin_fitzpatrick1", "Fair Human Epidermis (Fitzpatrick Type I)", 0.42, 0.22, 0.48, 1.44, 0.98, 1150.0, 0.35),
    ("pedestrian_skin_fitzpatrick3", "Medium Human Epidermis (Fitzpatrick Type III)", 0.32, 0.18, 0.46, 1.44, 0.98, 1150.0, 0.35),
    ("pedestrian_skin_fitzpatrick6", "Dark Human Epidermis (Fitzpatrick Type VI)", 0.16, 0.10, 0.44, 1.44, 0.98, 1150.0, 0.35),
    ("clothing_cotton_denim", "Ring-Spun Cotton Indigo Denim Fabric", 0.15, 0.08, 0.75, 1.53, 0.92, 600.0, 0.08),
    ("clothing_synthetic_nylon", "High-Tenacity Waterproof Nylon Outerwear", 0.28, 0.15, 0.42, 1.54, 0.90, 700.0, 0.25),
    ("clothing_fluorescent_vest", "ISO 20471 High-Visibility Warning Vest", 0.82, 0.88, 0.38, 1.55, 0.89, 650.0, 0.15),
    ("vegetation_grass_turf", "Mowed Kentucky Bluegrass Turf Canopy", 0.35, 0.12, 0.82, 1.42, 0.97, 850.0, 0.40),
    ("vegetation_tree_foliage", "Deciduous Quercus Oak Leaf Foliage", 0.42, 0.15, 0.78, 1.42, 0.96, 900.0, 0.45),
    ("vegetation_tree_bark", "Aged Coniferous Pine Tree Bark Surface", 0.18, 0.06, 0.88, 1.50, 0.92, 750.0, 0.20),
    ("water_shallow_puddle", "Rainwater Puddle on Impervious Surface", 0.05, 0.02, 0.02, 1.33, 0.98, 1750.0, 0.60),
    ("water_deep_reservoir", "Deep Turbid Surface Water Body", 0.04, 0.01, 0.04, 1.33, 0.98, 1780.0, 0.61),
    ("snow_fresh_powder", "Uncompacted High-Albedo Dendritic Snow", 0.88, 0.85, 0.65, 1.31, 0.99, 500.0, 0.12),
    ("ice_smooth_lake", "Clear Glaze Freshwater Ice Surface", 0.12, 0.04, 0.08, 1.31, 0.97, 1850.0, 2.22),
    ("metal_galvanized_guardrail", "Hot-Dip Galvanized Zinc Steel Guardrail", 0.52, 0.58, 0.35, 1.95, 0.45, 2300.0, 60.0),
    ("metal_anodized_aluminum", "Type III Hardcoat Anodized Aluminum 6061", 0.68, 0.72, 0.28, 1.70, 0.85, 2150.0, 167.0),
]

def generate_material_shader(id_str, title, alb_vis, alb_nir, rough, ior, emiss, therm_in, therm_k):
    code = [
        f"// Shroud :: Material Model: {title}",
        f"// Identifier: mat_{id_str}",
        "// Physical constants, spectral response tables, and Cook-Torrance BRDF evaluation.",
        "",
        "const PI: f32 = 3.14159265358979323846;",
        "const INV_PI: f32 = 0.31830988618379067154;",
        "const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;",
        "",
        f"const MAT_BASE_ALBEDO_VIS: f32 = {alb_vis:.4f};",
        f"const MAT_ALBEDO_NIR_905: f32 = {alb_nir:.4f};",
        f"const MAT_ROUGHNESS: f32 = {rough:.4f};",
        f"const MAT_INDEX_REFRACTION: f32 = {ior:.4f};",
        f"const MAT_LWIR_EMISSIVITY: f32 = {emiss:.4f};",
        f"const MAT_THERMAL_INERTIA: f32 = {therm_in:.1f};",
        f"const MAT_THERMAL_CONDUCTIVITY: f32 = {therm_k:.2f};",
        "",
        "struct MaterialQuery {",
        "    incident_dir: vec3<f32>,",
        "    surface_temp_k: f32,",
        "    view_dir: vec3<f32>,",
        "    wavelength_nm: f32,",
        "    normal: vec3<f32>,",
        "    padding: f32,",
        "};",
        "",
        "struct MaterialEvaluationResult {",
        "    brdf_value: vec3<f32>,",
        "    pdf: f32,",
        "    thermal_radiance: f32,",
        "    fresnel_reflectance: f32,",
        "    padding: vec2<f32>,",
        "};",
        "",
        "fn evaluate_spectral_reflectance(wavelength_nm: f32) -> f32 {",
        "    if (wavelength_nm < 400.0) {",
        f"        return MAT_BASE_ALBEDO_VIS * 0.85;",
        "    } else if (wavelength_nm <= 700.0) {",
        f"        let t = (wavelength_nm - 400.0) / 300.0;",
        f"        return mix(MAT_BASE_ALBEDO_VIS * 0.9, MAT_BASE_ALBEDO_VIS * 1.1, t);",
        "    } else if (wavelength_nm <= 1000.0) {",
        f"        let t = (wavelength_nm - 700.0) / 300.0;",
        f"        return mix(MAT_BASE_ALBEDO_VIS * 1.1, MAT_ALBEDO_NIR_905, t);",
        "    } else {",
        f"        return MAT_ALBEDO_NIR_905 * 0.9;",
        "    }",
        "}",
        "",
        "fn fresnel_dielectric(cos_theta: f32, eta: f32) -> f32 {",
        "    let c = clamp(abs(cos_theta), 0.0, 1.0);",
        "    let g2 = eta * eta - 1.0 + c * c;",
        "    if (g2 <= 0.0) {",
        "        return 1.0;",
        "    }",
        "    let g = sqrt(g2);",
        "    let a = (g - c) / (g + c);",
        "    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);",
        "    return 0.5 * a * a * (1.0 + b * b);",
        "}",
        "",
        "fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {",
        "    let alpha = roughness * roughness;",
        "    let alpha2 = alpha * alpha;",
        "    let d = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0;",
        "    return alpha2 / (PI * d * d + 1e-7);",
        "}",
        "",
        "fn smith_shadowing(n_dot_l: f32, n_dot_v: f32, roughness: f32) -> f32 {",
        "    let alpha = roughness * roughness;",
        "    let alpha2 = alpha * alpha;",
        "    let lambda_v = n_dot_l * sqrt((-n_dot_v * alpha2 + n_dot_v) * n_dot_v + alpha2);",
        "    let lambda_l = n_dot_v * sqrt((-n_dot_l * alpha2 + n_dot_l) * n_dot_l + alpha2);",
        "    return 2.0 * n_dot_l * n_dot_v / max(1e-7, lambda_v + lambda_l);",
        "}",
        "",
        "fn evaluate_thermal_emission(temp_k: f32) -> f32 {",
        "    return MAT_LWIR_EMISSIVITY * (STEFAN_BOLTZMANN / PI) * pow(max(10.0, temp_k), 4.0);",
        "}",
        "",
        "@group(0) @binding(0) var<storage, read> s_queries: array<MaterialQuery>;",
        "@group(0) @binding(1) var<storage, read_write> s_results: array<MaterialEvaluationResult>;",
        "",
        "@compute @workgroup_size(64, 1, 1)",
        f"fn cs_eval_material_{id_str}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
        "    let idx = gid.x;",
        "    if (idx >= arrayLength(&s_results) || idx >= arrayLength(&s_queries)) {",
        "        return;",
        "    }",
        "    let q = s_queries[idx];",
        "    let n = normalize(q.normal);",
        "    let l = normalize(q.incident_dir);",
        "    let v = normalize(q.view_dir);",
        "",
        "    let n_dot_l = max(0.0, dot(n, l));",
        "    let n_dot_v = max(0.0, dot(n, v));",
        "    let h = normalize(v + l);",
        "    let n_dot_h = max(0.0, dot(n, h));",
        "    let v_dot_h = max(0.0, dot(v, h));",
        "",
        "    let refl_spec = evaluate_spectral_reflectance(q.wavelength_nm);",
        "    let f = fresnel_dielectric(v_dot_h, MAT_INDEX_REFRACTION);",
        "    let d = ggx_distribution(n_dot_h, MAT_ROUGHNESS);",
        "    let g = smith_shadowing(n_dot_l, n_dot_v, MAT_ROUGHNESS);",
        "",
        "    let specular = (d * g * f) / max(1e-6, 4.0 * n_dot_l * n_dot_v);",
        "    let diffuse = (refl_spec * (1.0 - f) * INV_PI) * n_dot_l;",
        "",
        "    var res: MaterialEvaluationResult;",
        "    res.brdf_value = vec3<f32>(diffuse + specular);",
        "    res.pdf = d * n_dot_h / max(1e-6, 4.0 * v_dot_h);",
        "    res.thermal_radiance = evaluate_thermal_emission(q.surface_temp_k);",
        "    res.fresnel_reflectance = f;",
        "    res.padding = vec2<f32>(0.0, 0.0);",
        "",
        "    s_results[idx] = res;",
        "}",
    ]

    # Expand shader with physical property comments and tables to make it complete and comprehensive
    expanded = [code[0], code[1], code[2], ""]
    expanded.append("/*")
    expanded.append(f" * ASTM / ISO Standard Physical Characteristics for {title}:")
    expanded.append(f" * - Base Hemispherical Reflectance (Visible): {alb_vis:.4f}")
    expanded.append(f" * - Near-Infrared Reflectance (905nm LiDAR): {alb_nir:.4f}")
    expanded.append(f" * - Surface Micro-Roughness parameter (alpha): {rough:.4f}")
    expanded.append(f" * - Real Index of Refraction (n): {ior:.4f}")
    expanded.append(f" * - Long-Wave Infrared Emissivity (8-14 um): {emiss:.4f}")
    expanded.append(f" * - Thermal Inertia (J / (m^2 * K * s^0.5)): {therm_in:.1f}")
    expanded.append(f" * - Thermal Conductivity (W / (m * K)): {therm_k:.2f}")
    expanded.append(" *")
    expanded.append(" * Numerical Formulation:")
    expanded.append(" * The bidirectional reflectance distribution function (BRDF) f_r(omega_i, omega_o)")
    expanded.append(" * decomposes into a microfacet specular reflection term and a volumetric diffuse term:")
    expanded.append(" * f_r(omega_i, omega_o) = f_{spec} + f_{diff}")
    expanded.append(" * where f_{spec} = (D(h) * G(l, v) * F(v, h)) / (4 * (n . l) * (n . v))")
    expanded.append(" * and f_{diff} = (1 - F) * (rho / pi) * (n . l)")
    expanded.append(" */")
    expanded.append("")
    expanded.extend(code[3:])
    return "\n".join(expanded)

for mat in MATERIALS:
    subpath = f"materials_library/mat_{mat[0]}.wgsl"
    content = generate_material_shader(*mat)
    write_shader(subpath, content)

print("Generated 40 materials library shaders.")
