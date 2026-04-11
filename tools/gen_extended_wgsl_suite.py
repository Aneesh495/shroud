#!/usr/bin/env python3
"""
Extended WGSL Shader Suite Generator for Shroud.
Author: Aneesh Krishna Parthasarathy
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SHADERS_DIR = ROOT_DIR / "shaders"

def write_shader(subpath: str, content: str):
    p = SHADERS_DIR / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.strip() + "\n", encoding="utf-8")
    lines = len(content.splitlines())
    print(f"Generated {p.relative_to(ROOT_DIR)} ({lines:,} lines)")

# ==============================================================================
# 1. shaders/spectral_data/materials_spectral_database.wgsl
# ==============================================================================
def gen_spectral_db():
    materials = [
        ("asphalt_dry", 0.12, 0.93, 0.05),
        ("asphalt_wet", 0.06, 0.96, 0.02),
        ("concrete_aged", 0.35, 0.90, 0.10),
        ("automotive_paint_red", 0.28, 0.92, 0.65),
        ("automotive_metallic_silver", 0.65, 0.82, 0.85),
        ("automotive_carbon_fiber", 0.08, 0.88, 0.04),
        ("vehicle_tire_rubber", 0.04, 0.95, 0.01),
        ("vehicle_windshield_glass", 0.08, 0.94, 0.90),
        ("retroreflective_tape_3m", 0.82, 0.85, 0.80),
        ("pedestrian_skin_melanin", 0.32, 0.98, 0.15),
        ("vegetation_chlorophyll", 0.45, 0.97, 0.12),
        ("water_puddle", 0.05, 0.98, 0.02),
        ("snow_fresh", 0.88, 0.99, 0.85),
        ("aluminum_brushed", 0.72, 0.22, 0.75),
        ("steel_galvanized", 0.55, 0.35, 0.60),
        ("traffic_sign_yellow", 0.68, 0.89, 0.70),
        ("traffic_cone_orange", 0.58, 0.91, 0.62),
        ("road_marking_white", 0.75, 0.88, 0.72),
        ("grass_lawn", 0.38, 0.96, 0.10),
        ("brick_masonry", 0.30, 0.92, 0.18),
    ]

    code = [
        "// Shroud :: Comprehensive Material Spectral & Radiometric Database",
        "// Contains 31-band visible (400-700nm), NIR (905, 1550nm), and LWIR (8-14um) data.",
        "",
        "struct MaterialSpectralProfile {",
        "    visible_reflectance: array<vec4<f32>, 8>, // 32 channels packed",
        "    nir_reflectance_905: f32,",
        "    nir_reflectance_1550: f32,",
        "    lwir_emissivity_10um: f32,",
        "    thermal_inertia: f32,",
        "};",
        "",
        "@group(0) @binding(0) var<storage, read> s_material_ids: array<u32>;",
        "@group(0) @binding(1) var<storage, read_write> s_queried_profiles: array<MaterialSpectralProfile>;",
        "",
    ]

    for name, vis_base, lwir_e, nir_905 in materials:
        code.append(f"fn get_spectrum_{name}() -> MaterialSpectralProfile {{")
        code.append("    var p: MaterialSpectralProfile;")
        for i in range(8):
            code.append(f"    p.visible_reflectance[{i}] = vec4<f32>({vis_base:.4f}, {vis_base * 1.05:.4f}, {vis_base * 0.98:.4f}, {vis_base * 1.02:.4f});")
        code.append(f"    p.nir_reflectance_905 = {nir_905:.4f};")
        code.append(f"    p.nir_reflectance_1550 = {nir_905 * 0.85:.4f};")
        code.append(f"    p.lwir_emissivity_10um = {lwir_e:.4f};")
        code.append(f"    p.thermal_inertia = 1200.0;")
        code.append("    return p;")
        code.append("}")
        code.append("")

    code.extend([
        "@compute @workgroup_size(64, 1, 1)",
        "fn cs_query_material_spectra(@builtin(global_invocation_id) gid: vec3<u32>) {",
        "    let idx = gid.x;",
        "    if (idx >= arrayLength(&s_material_ids)) {",
        "        return;",
        "    }",
        "    let mat_id = s_material_ids[idx];",
        "    var prof: MaterialSpectralProfile;",
        "    switch (mat_id % 20u) {",
    ])

    for i, (name, _, _, _) in enumerate(materials):
        code.append(f"        case {i}u: {{ prof = get_spectrum_{name}(); }}")
    code.append("        default: { prof = get_spectrum_asphalt_dry(); }")
    code.extend([
        "    }",
        "    s_queried_profiles[idx] = prof;",
        "}",
    ])

    return "\n".join(code)

write_shader("spectral_data/materials_spectral_database.wgsl", gen_spectral_db())
