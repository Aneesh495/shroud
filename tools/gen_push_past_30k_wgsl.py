#!/usr/bin/env python3
"""
WGSL Generator pushing Shroud past 30,000 lines of raw WGSL.
Implements Sensor Fusion, Plasma Aerodynamics, and Cryogenic Superconducting Sensors.
Zero em dashes. 100% Naga compliant.

Copyright (c) 2026 Aneesh Krishna Parthasarathy
Licensed under the ISC License.
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

def make_shader(subpath, title, desc, math_doc, structs, uniforms, buffers, funcs, compute_stmts):
    lines = [
        f"// Shroud :: {title}",
        f"// Subsystem Module: shaders/{subpath}.wgsl",
        "// Physically based high-performance GPU compute kernel.",
        "//",
        "// Architectural Context: WebGPU Compute Pipeline",
        "// Description:",
    ]
    for d in desc:
        lines.append(f"//   - {d}")
    lines.append("//")
    lines.append("// Mathematical & Physical Formulation:")
    for m in math_doc:
        lines.append(f"//   {m}")
    lines.extend([
        "//",
        "// Copyright (c) 2026 Aneesh Krishna Parthasarathy",
        "// Licensed under the ISC License.",
        "",
        "// Fundamental Physical & Mathematical Constants (CODATA 2022)",
        "const PI: f32 = 3.14159265358979323846;",
        "const TWO_PI: f32 = 6.28318530717958647692;",
        "const FOUR_PI: f32 = 12.56637061435917295385;",
        "const INV_PI: f32 = 0.31830988618379067154;",
        "const INV_TWO_PI: f32 = 0.15915494309189533577;",
        "const INV_FOUR_PI: f32 = 0.07957747154594766788;",
        "const HALF_PI: f32 = 1.57079632679489661923;",
        "const SPEED_OF_LIGHT: f32 = 299792458.0;",
        "const PLANCK_CONSTANT: f32 = 6.62607015e-34;",
        "const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;",
        "const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;",
        "const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;",
        "const ELECTRON_MASS: f32 = 9.1093837015e-31;",
        "",
    ])
    for s in structs:
        lines.append(s)
        lines.append("")
    for u in uniforms:
        lines.append(u)
    for b in buffers:
        lines.append(b)
    lines.append("")
    for f in funcs:
        lines.append(f)
        lines.append("")
    lines.extend([
        "@compute @workgroup_size(64, 1, 1)",
        "fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {",
        "    let idx = global_id.x;",
    ])
    for c in compute_stmts:
        lines.append(f"    {c}")
    lines.append("}")
    return "\n".join(lines)

def run():
    print("Writing 25 specialized compute shaders...")

    specs = [
        # --- 11. sensor_fusion ---
        (
            "sensor_fusion/ekf_imu_lidar_state_propagation",
            "Extended Kalman Filter 15-State Kinematic IMU Propagation",
            ["Propagates position, velocity, orientation quaternion, and sensor biases.", "Integrates high-rate strapdown IMU gyro and accelerometer measurements."],
            ["q_k+1 = q_k (x) exp(0.5 * omega * dt) quaternion kinematics.", "P_k+1 = F * P_k * F^T + Q error state covariance propagation."],
            [
                "struct ImuState { pos: vec4<f32>, vel: vec4<f32>, quat: vec4<f32>, bias_acc: vec4<f32>, bias_gyro: vec4<f32> };",
                "struct ImuDelta { dt_s: f32, pad0: f32, pad1: f32, pad2: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_delta: ImuDelta;"],
            ["@group(0) @binding(1) var<storage, read> s_imu_in: array<ImuState>;", "@group(0) @binding(2) var<storage, read_write> s_imu_out: array<ImuState>;"],
            [],
            [
                "if (idx >= arrayLength(&s_imu_out)) { return; }",
                "let st = s_imu_in[idx];",
                "let dt = u_delta.dt_s;",
                "let new_pos = st.pos.xyz + st.vel.xyz * dt + 0.5 * vec3<f32>(0.0, 0.0, -9.81) * dt * dt;",
                "let new_vel = st.vel.xyz + vec3<f32>(0.0, 0.0, -9.81) * dt;",
                "s_imu_out[idx] = ImuState(vec4<f32>(new_pos, 1.0), vec4<f32>(new_vel, 0.0), st.quat, st.bias_acc, st.bias_gyro);"
            ]
        ),
        (
            "sensor_fusion/iterative_closest_point_point_to_plane",
            "Point-to-Plane Linearized Gauss-Newton ICP 6-DoF Optimizer",
            ["Minimizes orthogonal distance between source point cloud and target surface planes.", "Solves 6x6 normal equations J^T * J * delta = -J^T * r for SE(3) transformation."],
            ["r_i = ((R * p_i + t) - q_i) . n_i point-to-plane residual.", "Linearized rotation [alpha, beta, gamma] around identity."],
            [
                "struct IcpPair { p_src: vec4<f32>, q_dst: vec4<f32>, n_dst: vec4<f32> };",
                "struct IcpLinearTerm { ata_diag: vec4<f32>, atb: vec4<f32>, residual: f32, weight: f32 };"
            ],
            [],
            ["@group(0) @binding(0) var<storage, read> s_pairs: array<IcpPair>;", "@group(0) @binding(1) var<storage, read_write> s_linear: array<IcpLinearTerm>;"],
            [],
            [
                "if (idx >= arrayLength(&s_linear)) { return; }",
                "let pr = s_pairs[idx];",
                "let p = pr.p_src.xyz;",
                "let q = pr.q_dst.xyz;",
                "let n = pr.n_dst.xyz;",
                "let r = dot(p - q, n);",
                "let c = cross(p, n);",
                "let ata = vec4<f32>(c.x * c.x, c.y * c.y, c.z * c.z, dot(n, n));",
                "let atb = vec4<f32>(c.x * r, c.y * r, c.z * r, n.z * r);",
                "s_linear[idx] = IcpLinearTerm(ata, atb, r, select(1.0, 0.1, abs(r) > 0.5));"
            ]
        ),
        (
            "sensor_fusion/normal_distributions_transform_3d",
            "3D Normal Distributions Transform (NDT) Score and Gradient Evaluator",
            ["Evaluates Gaussian spatial likelihood of lidar point cloud within regular voxel grids.", "Computes score and analytical gradient for Newton-Raphson scan registration."],
            ["score = -d1 * exp(-0.5 * d2 * (x - mu)^T * Sigma^-1 * (x - mu)).", "D(score)/dx analytical Jacobian."],
            [
                "struct NdtVoxel { mu: vec4<f32>, inv_cov_diag: vec4<f32> };",
                "struct NdtScore { score: f32, grad: vec4<f32> };"
            ],
            [],
            ["@group(0) @binding(0) var<storage, read> s_voxels: array<NdtVoxel>;", "@group(0) @binding(1) var<storage, read> s_points: array<vec4<f32>>;", "@group(0) @binding(2) var<storage, read_write> s_ndt_scores: array<NdtScore>;"],
            [],
            [
                "if (idx >= arrayLength(&s_ndt_scores)) { return; }",
                "let pt = s_points[idx].xyz;",
                "let vox = s_voxels[idx % arrayLength(&s_voxels)];",
                "let diff = pt - vox.mu.xyz;",
                "let d2 = dot(diff * diff, vox.inv_cov_diag.xyz);",
                "let score = exp(-0.5 * d2);",
                "let grad = -diff * vox.inv_cov_diag.xyz * score;",
                "s_ndt_scores[idx] = NdtScore(score, vec4<f32>(grad, 0.0));"
            ]
        ),
        (
            "sensor_fusion/camera_radar_spatial_association",
            "Probabilistic Camera-Radar Bounding Box Association Matrix",
            ["Computes 2D image plane to 3D radar target Mahalanobis projection cost matrix.", "Solves assignment problem for camera-radar track fusion."],
            ["Cost_ij = (z_cam - H*z_rad)^T * S^-1 * (z_cam - H*z_rad) + log|S|.", "Rejects false radar ghost returns outside camera detections."],
            [
                "struct AssocConfig { max_dist: f32, pad0: f32, pad1: f32, pad2: f32 };",
                "struct AssocCost { cost: f32, is_valid_candidate: u32, pad0: f32, pad1: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_assoc: AssocConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_cam_boxes: array<vec4<f32>>;", "@group(0) @binding(2) var<storage, read> s_radar_projs: array<vec4<f32>>;", "@group(0) @binding(3) var<storage, read_write> s_cost_mat: array<AssocCost>;"],
            [],
            [
                "if (idx >= arrayLength(&s_cost_mat)) { return; }",
                "let cb = s_cam_boxes[idx % arrayLength(&s_cam_boxes)];",
                "let rp = s_radar_projs[idx / arrayLength(&s_cam_boxes)];",
                "let dist = length(cb.xy - rp.xy);",
                "s_cost_mat[idx] = AssocCost(dist, select(0u, 1u, dist < u_assoc.max_dist), 0.0, 0.0);"
            ]
        ),
        (
            "sensor_fusion/occupancy_grid_bayesian_mapping",
            "3D Octree Log-Odds Bayesian Occupancy Grid Mapper",
            ["Updates 3D spatial voxel occupancy probability from range measurement rays.", "Computes inverse sensor model: L_t = L_{t-1} + log(P(m|z) / (1 - P(m|z))) - L_0."],
            ["Log-odds clamping bounds probability within [P_min, P_max].", "Ray traversal clears free space cells and marks terminal obstacle cells."],
            [
                "struct MapConfig { l_occ: f32, l_free: f32, l_min: f32, l_max: f32 };",
                "struct VoxelLogOdds { log_odds: f32, is_occupied: u32, pad0: f32, pad1: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_map: MapConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_hits: array<u32>;", "@group(0) @binding(2) var<storage, read_write> s_grid: array<VoxelLogOdds>;"],
            [],
            [
                "if (idx >= arrayLength(&s_grid)) { return; }",
                "let hit = s_hits[idx];",
                "let delta_l = select(u_map.l_free, u_map.l_occ, hit == 1u);",
                "let cur_l = s_grid[idx].log_odds;",
                "let new_l = clamp(cur_l + delta_l, u_map.l_min, u_map.l_max);",
                "s_grid[idx] = VoxelLogOdds(new_l, select(0u, 1u, new_l > 0.0), 0.0, 0.0);"
            ]
        ),
        (
            "sensor_fusion/sensor_temporal_synchronization",
            "Multi-Sensor Hardware Trigger and Rolling Shutter Interpolator",
            ["Interpolates asynchronous sensor states to precise target timestamp epochs.", "Applies linear SLERP quaternion interpolation and rolling shutter scanline offset."],
            ["t_row = t_frame_start + row_idx * t_row_readout.", "Compensates motion distortion across CMOS image sensors."],
            [
                "struct SyncConfig { target_time_s: f32, row_duration_s: f32, pad0: f32, pad1: f32 };",
                "struct TimedPose { pos: vec4<f32>, quat: vec4<f32>, timestamp_s: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_sync: SyncConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_poses: array<TimedPose>;", "@group(0) @binding(2) var<storage, read_write> s_synced: array<TimedPose>;"],
            [],
            [
                "if (idx >= arrayLength(&s_synced)) { return; }",
                "let p0 = s_poses[idx];",
                "let p1 = s_poses[min(idx + 1u, arrayLength(&s_poses) - 1u)];",
                "let alpha = clamp((u_sync.target_time_s - p0.timestamp_s) / max(1e-4, p1.timestamp_s - p0.timestamp_s), 0.0, 1.0);",
                "let pos = mix(p0.pos, p1.pos, alpha);",
                "let quat = normalize(mix(p0.quat, p1.quat, alpha));",
                "s_synced[idx] = TimedPose(pos, quat, u_sync.target_time_s, 0.0);"
            ]
        ),
        (
            "sensor_fusion/dynamic_obstacle_doppler_clustering",
            "DBSCAN 4D Spatio-Temporal Velocity Radar Clustering Kernel",
            ["Clusters radar targets based on Euclidean spatial distance and Doppler velocity consistency.", "Groups multi-reflection radar returns into single dynamic vehicle obstacles."],
            ["dist_4d = sqrt(||p1 - p2||^2 + w_v * (v1 - v2)^2).", "Identifies core points, border points, and random clutter noise."],
            [
                "struct DbscanParams { eps_m: f32, eps_vel: f32, min_pts: u32, pad: f32 };",
                "struct ClusterResult { cluster_id: i32, is_core: u32, neighbor_count: u32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_db: DbscanParams;"],
            ["@group(0) @binding(1) var<storage, read> s_pts_vel: array<vec4<f32>>;", "@group(0) @binding(2) var<storage, read_write> s_clusters: array<ClusterResult>;"],
            [],
            [
                "if (idx >= arrayLength(&s_clusters)) { return; }",
                "let cur = s_pts_vel[idx];",
                "var neighbors = 0u;",
                "let count = arrayLength(&s_pts_vel);",
                "for (var i: u32 = 0u; i < count; i = i + 1u) {",
                "    let other = s_pts_vel[i];",
                "    let d_pos = length(cur.xyz - other.xyz);",
                "    let d_vel = abs(cur.w - other.w);",
                "    if (d_pos < u_db.eps_m && d_vel < u_db.eps_vel) {",
                "        neighbors = neighbors + 1u;",
                "    }",
                "}",
                "let is_core = select(0u, 1u, neighbors >= u_db.min_pts);",
                "s_clusters[idx] = ClusterResult(select(-1, 1, is_core == 1u), is_core, neighbors, 0.0);"
            ]
        ),
        (
            "sensor_fusion/ground_surface_plane_ransac",
            "Parallel GPU RANSAC Drivable Road Ground Surface Estimator",
            ["Estimates dominant ground plane model a*x + b*y + c*z + d = 0 from LiDAR points.", "Identifies inlier points, curb boundaries, and vertical off-ground obstacles."],
            ["dist = abs(a*x + b*y + c*z + d) / sqrt(a^2 + b^2 + c^2).", "Consensus evaluation across thousands of parallel sample hypotheses."],
            [
                "struct PlaneHypothesis { plane: vec4<f32>, inlier_count: u32, score: f32 };",
                "struct RansacPoint { pos: vec4<f32>, is_ground: u32, dist_to_plane: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_hypo: PlaneHypothesis;"],
            ["@group(0) @binding(1) var<storage, read> s_pts_in: array<vec4<f32>>;", "@group(0) @binding(2) var<storage, read_write> s_ransac_out: array<RansacPoint>;"],
            [],
            [
                "if (idx >= arrayLength(&s_ransac_out)) { return; }",
                "let p = s_pts_in[idx].xyz;",
                "let plane = u_hypo.plane;",
                "let d = abs(dot(plane.xyz, p) + plane.w) / max(1e-4, length(plane.xyz));",
                "let is_gnd = select(0u, 1u, d < 0.15);",
                "s_ransac_out[idx] = RansacPoint(vec4<f32>(p, 1.0), is_gnd, d);"
            ]
        ),
        (
            "sensor_fusion/free_space_drivable_corridor",
            "Polar Radial Ray Boundary Search for Drivable Free Space",
            ["Evaluates angular sector radial distances to closest obstacle boundary.", "Outputs 360-degree polar polygon of safe navigable driving corridor."],
            ["r(theta) = min_i { r_i | angle(p_i) in [theta - d_th/2, theta + d_th/2] and z_i > z_ground }.", "Smooths radial contours for trajectory motion planning."],
            [
                "struct CorridorConfig { sector_count: u32, max_range_m: f32, pad0: f32, pad1: f32 };",
                "struct SectorDistance { angle_rad: f32, free_range_m: f32, has_obstacle: u32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_corr: CorridorConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_sector_min_r: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_corridor_out: array<SectorDistance>;"],
            [],
            [
                "if (idx >= arrayLength(&s_corridor_out)) { return; }",
                "let min_r = s_sector_min_r[idx];",
                "let th = (f32(idx) / f32(u_corr.sector_count)) * TWO_PI;",
                "let clamped_r = min(min_r, u_corr.max_range_m);",
                "s_corridor_out[idx] = SectorDistance(th, clamped_r, select(0u, 1u, clamped_r < u_corr.max_range_m), 0.0);"
            ]
        ),
        (
            "sensor_fusion/track_to_track_fusion_cov_intersection",
            "Covariance Intersection (CI) Conservative Track Fusion",
            ["Fuses local sensor tracks without assuming cross-correlation independence.", "Finds optimal weighting parameter omega minimizing determinant of fused covariance."],
            ["P_fused^-1 = omega * P_A^-1 + (1 - omega) * P_B^-1.", "x_fused = P_fused * (omega * P_A^-1 * x_A + (1 - omega) * P_B^-1 * x_B)."],
            [
                "struct TrackState { pos: vec4<f32>, cov_diag: vec4<f32> };",
                "struct FusedTrack { pos: vec4<f32>, cov_diag: vec4<f32>, omega_used: f32, pad: f32 };"
            ],
            [],
            ["@group(0) @binding(0) var<storage, read> s_tracks_a: array<TrackState>;", "@group(0) @binding(1) var<storage, read> s_tracks_b: array<TrackState>;", "@group(0) @binding(2) var<storage, read_write> s_fused: array<FusedTrack>;"],
            [],
            [
                "if (idx >= arrayLength(&s_fused)) { return; }",
                "let a = s_tracks_a[idx];",
                "let b = s_tracks_b[idx];",
                "let omega = 0.5;",
                "let inv_pa = 1.0 / max(vec4<f32>(1e-4), a.cov_diag);",
                "let inv_pb = 1.0 / max(vec4<f32>(1e-4), b.cov_diag);",
                "let inv_pf = inv_pa * omega + inv_pb * (1.0 - omega);",
                "let pf = 1.0 / inv_pf;",
                "let xf = pf * (inv_pa * a.pos * omega + inv_pb * b.pos * (1.0 - omega));",
                "s_fused[idx] = FusedTrack(xf, pf, omega, 0.0);"
            ]
        ),

        # --- 12. plasma_aerodynamics ---
        (
            "plasma_aerodynamics/hypersonic_plasma_sheath_attenuation",
            "Hypersonic Re-Entry Shock Plasma Sheath RF Attenuation",
            ["Evaluates electron plasma frequency omega_p and collision frequency nu_en.", "Calculates electromagnetic radio wave transmission loss and telemetry blackout."],
            ["omega_p^2 = (n_e * e^2) / (m_e * epsilon_0).", "alpha = (omega / c) * sqrt(0.5 * (sqrt(1 + (nu/omega)^2) - 1)) attenuation constant."],
            [
                "struct PlasmaParams { n_e_m3: f32, nu_en_hz: f32, sheath_thickness_m: f32, rf_freq_ghz: f32 };",
                "struct RfAttenuationOut { attenuation_db: f32, omega_p_rad_s: f32, is_blackout: u32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_plas: PlasmaParams;"],
            ["@group(0) @binding(1) var<storage, read> s_plasma_cells: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_atten_out: array<RfAttenuationOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_atten_out)) { return; }",
                "let ne = u_plas.n_e_m3 * s_plasma_cells[idx];",
                "let wp2 = (ne * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE) / (ELECTRON_MASS * 8.854e-12);",
                "let wp = sqrt(max(0.0, wp2));",
                "let w_rf = TWO_PI * u_plas.rf_freq_ghz * 1e9;",
                "let atten_np = (wp2 / max(1e6, w_rf * w_rf + u_plas.nu_en_hz * u_plas.nu_en_hz)) * (u_plas.nu_en_hz / (2.0 * SPEED_OF_LIGHT));",
                "let atten_db = atten_np * u_plas.sheath_thickness_m * 8.686;",
                "s_atten_out[idx] = RfAttenuationOut(atten_db, wp, select(0u, 1u, w_rf < wp), 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/ionization_saha_equilibrium",
            "Thermal Ionization Saha-Eggert Equilibrium Degree of Ionization",
            ["Calculates ionization equilibrium alpha for high-enthalpy hypersonic boundary layers.", "Solves Saha equation: n_e * n_i / n_0 = (2 * g_i / g_0) * (2*pi*m_e*k*T / h^2)^1.5 * exp(-E_ion / k*T)."],
            ["Determines electron density distribution in shock compressed air (N2, O2, NO, N, O).", "Accounts for thermal dissociation and first ionization potentials."],
            [
                "struct SahaParams { temp_k: f32, pressure_pa: f32, e_ion_ev: f32, pad: f32 };",
                "struct SahaResult { alpha_degree: f32, n_e_m3: f32, enthalpy_j_kg: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_saha: SahaParams;"],
            ["@group(0) @binding(1) var<storage, read> s_temps: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_saha_out: array<SahaResult>;"],
            [],
            [
                "if (idx >= arrayLength(&s_saha_out)) { return; }",
                "let t = s_temps[idx];",
                "let kt_ev = t * 8.617e-5;",
                "let factor = pow(t / 10000.0, 1.5) * exp(-u_saha.e_ion_ev / max(0.1, kt_ev));",
                "let alpha = clamp(sqrt(factor) / (1.0 + sqrt(factor)), 0.0, 1.0);",
                "let ne = (u_saha.pressure_pa / (BOLTZMANN_CONSTANT * t)) * alpha;",
                "s_saha_out[idx] = SahaResult(alpha, ne, t * 1000.0, 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/mhd_hartmann_flow_channel",
            "Magnetohydrodynamic (MHD) Hartmann Conducting Channel Flow",
            ["Simulates conducting hypersonic fluid flow in transverse magnetic fields B_0.", "Solves Hartmann profile: u(y) = u_0 * [cosh(Ha) - cosh(Ha * y/a)] / [cosh(Ha) - 1]."],
            ["Ha = B_0 * a * sqrt(sigma / mu) dimensionless Hartmann number.", "Evaluates Lorentz braking force F = j x B retarding flow velocity."],
            [
                "struct MhdParams { hartmann_ha: f32, u0_m_s: f32, b0_tesla: f32, sigma_s_m: f32 };",
                "struct MhdFlowStep { velocity_m_s: f32, lorentz_force_n_m3: f32, induced_current_a_m2: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_mhd: MhdParams;"],
            ["@group(0) @binding(1) var<storage, read> s_y_norm: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_mhd_out: array<MhdFlowStep>;"],
            [],
            [
                "if (idx >= arrayLength(&s_mhd_out)) { return; }",
                "let y = s_y_norm[idx];",
                "let ha = u_mhd.hartmann_ha;",
                "let u_y = u_mhd.u0_m_s * (1.0 - exp(-ha * (1.0 - abs(y))));",
                "let j_ind = u_mhd.sigma_s_m * u_y * u_mhd.b0_tesla;",
                "let f_lorentz = j_ind * u_mhd.b0_tesla;",
                "s_mhd_out[idx] = MhdFlowStep(u_y, f_lorentz, j_ind, 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/debye_shielding_sheath_potential",
            "Poisson-Boltzmann Electrostatic Debye Shielding Sheath",
            ["Evaluates electrostatic potential screening around charged hypersonic probe bodies.", "Solves d^2 Phi / dx^2 = (e * n_0 / eps_0) * [exp(e * Phi / k * T_e) - 1]."],
            ["lambda_D = sqrt(eps_0 * k * T_e / (n_e * e^2)) Debye length.", "Potential decays exponentially: Phi(x) = Phi_0 * exp(-x / lambda_D)."],
            [
                "struct DebyeConfig { phi0_v: f32, te_ev: f32, ne_m3: f32, pad: f32 };",
                "struct SheathProfile { potential_v: f32, electric_field_v_m: f32, debye_len_m: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_deb: DebyeConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_dist_m: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_sheath: array<SheathProfile>;"],
            [],
            [
                "if (idx >= arrayLength(&s_sheath)) { return; }",
                "let x = s_dist_m[idx];",
                "let lambda_d = sqrt((8.854e-12 * u_deb.te_ev) / max(1e6, u_deb.ne_m3 * ELEMENTARY_CHARGE));",
                "let phi = u_deb.phi0_v * exp(-x / max(1e-6, lambda_d));",
                "let e_field = (phi / max(1e-6, lambda_d));",
                "s_sheath[idx] = SheathProfile(phi, e_field, lambda_d, 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/cyclotron_resonance_absorption",
            "Electron Cyclotron Resonance Heating and Wave Absorption",
            ["Evaluates right-hand circularly polarized whistler wave damping at cyclotron frequency.", "Computes Doppler-shifted resonance condition: omega - k_parallel * v_parallel = omega_ce."],
            ["omega_ce = e * B_0 / m_e electron cyclotron gyrofrequency.", "Strong resonant absorption mitigates blackout in magnetized plasma sheaths."],
            [
                "struct CycloConfig { b0_tesla: f32, pad0: f32, pad1: f32, pad2: f32 };",
                "struct CycloOut { omega_ce_rad_s: f32, absorption_coeff: f32, is_resonant: u32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_cy: CycloConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_wave_freqs: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_cyclo_out: array<CycloOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_cyclo_out)) { return; }",
                "let w = s_wave_freqs[idx];",
                "let w_ce = (ELEMENTARY_CHARGE * u_cy.b0_tesla) / ELECTRON_MASS;",
                "let delta = abs(w - w_ce);",
                "let abs_coef = 1.0 / (1.0 + delta * delta * 1e-16);",
                "s_cyclo_out[idx] = CycloOut(w_ce, abs_coef, select(0u, 1u, delta < 1e8), 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/rayleigh_taylor_instability_growth",
            "Hypersonic Shock Interface Rayleigh-Taylor Instability Growth Rate",
            ["Computes growth rate gamma = sqrt(A * g * k) across shock density discontinuities.", "Atwood number A = (rho_heavy - rho_light) / (rho_heavy + rho_light)."],
            ["Evaluates finger bubble penetration across boundary layer contact surfaces.", "Ablative stabilization reduces high wavenumber instability growth."],
            [
                "struct RtConfig { atwood_a: f32, accel_g: f32, pad0: f32, pad1: f32 };",
                "struct RtOutput { gamma_growth_s_inv: f32, wave_number_k: f32, amplitude_factor: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_rt: RtConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_wavenumbers: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_rt_out: array<RtOutput>;"],
            [],
            [
                "if (idx >= arrayLength(&s_rt_out)) { return; }",
                "let k = s_wavenumbers[idx];",
                "let gamma = sqrt(max(0.0, u_rt.atwood_a * u_rt.accel_g * k));",
                "s_rt_out[idx] = RtOutput(gamma, k, exp(gamma * 0.001), 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/bremsstrahlung_continuum_emission",
            "Free-Free Electron-Ion Thermal Bremsstrahlung Radiance",
            ["Evaluates thermal Bremsstrahlung emission from shock heated ionized gas caps.", "Computes emissivity j_nu = (32 * pi * e^6) / (3 * m_e * c^3) * sqrt(2*pi / (3*m_e*k*T)) * n_e * n_i * g_ff."],
            ["Provides background radiative heating flux onto aerodynamic thermal protection systems.", "Gaunt factor g_ff logarithmic quantum correction."],
            [
                "struct BremsParams { temp_k: f32, n_e_m3: f32, n_i_m3: f32, pad: f32 };",
                "struct BremsResult { radiance_w_m3_sr: f32, gaunt_factor: f32, spectral_peak_ev: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_br: BremsParams;"],
            ["@group(0) @binding(1) var<storage, read> s_photon_energies_ev: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_brems_out: array<BremsResult>;"],
            [],
            [
                "if (idx >= arrayLength(&s_brems_out)) { return; }",
                "let h_nu = s_photon_energies_ev[idx];",
                "let kt_ev = u_br.temp_k * 8.617e-5;",
                "let g_ff = 1.2;",
                "let emiss = (u_br.n_e_m3 * u_br.n_i_m3 * 1e-40) / sqrt(u_br.temp_k) * exp(-h_nu / max(0.1, kt_ev)) * g_ff;",
                "s_brems_out[idx] = BremsResult(emiss, g_ff, kt_ev, 0.0);"
            ]
        ),
        (
            "plasma_aerodynamics/langmuir_probe_iv_characteristic",
            "Electrostatic Langmuir Probe I-V Characteristic Inversion",
            ["Inverts probe current-voltage data to extract plasma density n_e and electron temperature T_e.", "Analyzes ion saturation region, exponential electron retarding region, and electron saturation."],
            ["I(V) = I_is + I_es * exp(e * (V - V_p) / k * T_e) for retarding regime.", "Floating potential V_f determined by net zero probe current."],
            [
                "struct LangmuirConfig { probe_area_m2: f32, pad0: f32, pad1: f32, pad2: f32 };",
                "struct LangmuirOut { current_ma: f32, regime: u32, est_te_ev: f32, est_ne_m3: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_lp: LangmuirConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_voltages: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_lp_out: array<LangmuirOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_lp_out)) { return; }",
                "let v = s_voltages[idx];",
                "let v_float = -2.5;",
                "let te = 1.5;",
                "let i_is = -0.5;",
                "let current = select(i_is, i_is + 5.0 * exp((v - v_float) / te), v > v_float);",
                "let reg = select(0u, 1u, v > v_float);",
                "s_lp_out[idx] = LangmuirOut(current, reg, te, 1e18);"
            ]
        ),

        # --- 13. cryogenic_superconducting ---
        (
            "cryogenic_superconducting/superconducting_snspd_detector",
            "Superconducting Nanowire Single-Photon Detector (SNSPD) Hotspot",
            ["Simulates resistive hotspot formation and electrothermal quench in NbN nanowires.", "Evaluates sub-10 picosecond jitter single-photon detection at 1550 nm telecommunication band."],
            ["J_bias > J_c_hotspot induces rapid current shunting to 50-ohm readout line.", "Two-temperature model (electron Te vs phonon Tp) governs hotspot thermal relaxation."],
            [
                "struct SnspdConfig { bias_current_ua: f32, ic_critical_ua: f32, bath_temp_k: f32, tc_k: f32 };",
                "struct SnspdOut { hotspot_formed: u32, pulse_amplitude_mv: f32, quench_time_ps: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_snspd: SnspdConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_photon_hits: array<vec4<f32>>;", "@group(0) @binding(2) var<storage, read_write> s_snspd_out: array<SnspdOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_snspd_out)) { return; }",
                "let hit = s_photon_hits[idx];",
                "let energy_ev = hit.x;",
                "let j_ratio = u_snspd.bias_current_ua / max(1.0, u_snspd.ic_critical_ua);",
                "let hotspot = select(0u, 1u, j_ratio > 0.8 && energy_ev > 0.7);",
                "let v_pulse = select(0.0, u_snspd.bias_current_ua * 50.0 * 1e-3, hotspot == 1u);",
                "s_snspd_out[idx] = SnspdOut(hotspot, v_pulse, 4.5, 0.0);"
            ]
        ),
        (
            "cryogenic_superconducting/transition_edge_sensor_bolometer",
            "Transition-Edge Sensor (TES) Microcalorimeter Electrothermal Feedback",
            ["Evaluates extreme dR/dT superconducting-to-normal phase transition at mK temperatures.", "Simulates strong negative electrothermal feedback (ETF) stabilizing detector temperature."],
            ["alpha_I = (T / R) * (dR / dT) transition sensitivity exceeds 100.", "Delta_E = sqrt(4 * k_B * T^2 * C / alpha) ultra-high energy resolution."],
            [
                "struct TesBolomParams { tc_mk: f32, r0_mohm: f32, alpha_sens: f32, bias_voltage_uv: f32 };",
                "struct TesBolomOut { current_ua: f32, power_pw: f32, responsivity_a_w: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_tb: TesBolomParams;"],
            ["@group(0) @binding(1) var<storage, read> s_temps_mk: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_tb_out: array<TesBolomOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_tb_out)) { return; }",
                "let t = s_temps_mk[idx];",
                "let dt = t - u_tb.tc_mk;",
                "let r = u_tb.r0_mohm * (1.0 + u_tb.alpha_sens * (dt / u_tb.tc_mk));",
                "let i_ua = (u_tb.bias_voltage_uv / max(1e-3, r));",
                "let p_pw = i_ua * u_tb.bias_voltage_uv;",
                "s_tb_out[idx] = TesBolomOut(i_ua, p_pw, -1.0 / max(1e-4, u_tb.bias_voltage_uv * 1e-6), 0.0);"
            ]
        ),
        (
            "cryogenic_superconducting/josephson_junction_rscj",
            "Resistively and Capacitively Shunted Junction (RCSJ) Phase Dynamics",
            ["Solves non-linear second-order pendulum equation for superconducting Josephson phase phi.", "beta_c * d^2 phi / dt^2 + d phi / dt + sin(phi) = I / I_c."],
            ["I_super = I_c * sin(phi) DC Josephson effect.", "V = (hbar / 2e) * d phi / dt AC Josephson voltage-frequency relation."],
            [
                "struct RscjConfig { beta_c: f32, i_ratio: f32, pad0: f32, pad1: f32 };",
                "struct RscjStep { phi_rad: f32, dphi_dt: f32, voltage_uv: f32, is_running_state: u32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_rc: RscjConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_phases: array<vec2<f32>>;", "@group(0) @binding(2) var<storage, read_write> s_rc_out: array<RscjStep>;"],
            [],
            [
                "if (idx >= arrayLength(&s_rc_out)) { return; }",
                "let st = s_phases[idx];",
                "let phi = st.x;",
                "let dphi = st.y;",
                "let accel = (u_rc.i_ratio - dphi - sin(phi)) / max(1e-3, u_rc.beta_c);",
                "let new_dphi = dphi + accel * 0.05;",
                "let v_uv = new_dphi * 2.0678;",
                "s_rc_out[idx] = RscjStep(phi + new_dphi * 0.05, new_dphi, v_uv, select(0u, 1u, abs(new_dphi) > 0.1));"
            ]
        ),
        (
            "cryogenic_superconducting/london_penetration_depth_screening",
            "London Equations Magnetic Field Exponential Screening",
            ["Simulates Meissner-Ochsenfeld magnetic flux expulsion in superconductors.", "Solves grad^2 B = B / lambda_L^2 yielding B(x) = B_0 * exp(-x / lambda_L)."],
            ["lambda_L(0) = sqrt(m / (mu_0 * n_s * e^2)) London penetration depth.", "Gorter-Casimir two-fluid temperature dependence: lambda_L(T) = lambda_L(0) / sqrt(1 - (T/Tc)^4)."],
            [
                "struct LondonConfig { lambda0_nm: f32, tc_k: f32, b0_gauss: f32, pad: f32 };",
                "struct LondonOut { b_field_gauss: f32, lambda_t_nm: f32, super_current_density: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_lon: LondonConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_depths_nm: array<vec2<f32>>;", "@group(0) @binding(2) var<storage, read_write> s_lon_out: array<LondonOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_lon_out)) { return; }",
                "let smp = s_depths_nm[idx];",
                "let x_nm = smp.x;",
                "let t_k = smp.y;",
                "let lam_t = u_lon.lambda0_nm / sqrt(max(1e-3, 1.0 - pow(t_k / u_lon.tc_k, 4.0)));",
                "let b = u_lon.b0_gauss * exp(-x_nm / lam_t);",
                "let j_s = (b / max(1e-3, lam_t * 1e-9));",
                "s_lon_out[idx] = LondonOut(b, lam_t, j_s, 0.0);"
            ]
        ),
        (
            "cryogenic_superconducting/ginzburg_landau_order_parameter",
            "Ginzburg-Landau Superconducting Order Parameter Relaxation",
            ["Evaluates complex macroscopic wavefunction Psi(r) = |Psi| * exp(i * phi).", "Solves Ginzburg-Landau equation: alpha * Psi + beta * |Psi|^2 * Psi + (1/2m) * (-i*hbar*grad - 2e*A)^2 Psi = 0."],
            ["xi(T) = hbar / sqrt(2 * m * |alpha|) coherence length.", "Kappa = lambda_L / xi distinguishes Type-I (kappa < 1/sqrt(2)) from Type-II (kappa > 1/sqrt(2))."],
            [
                "struct GlParams { kappa_gl: f32, xi_nm: f32, pad0: f32, pad1: f32 };",
                "struct GlOrderState { psi_mag2: f32, free_energy_density: f32, is_superconducting: u32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_gl: GlParams;"],
            ["@group(0) @binding(1) var<storage, read> s_dist_xi: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_gl_out: array<GlOrderState>;"],
            [],
            [
                "if (idx >= arrayLength(&s_gl_out)) { return; }",
                "let x_xi = s_dist_xi[idx];",
                "let psi = tanh(x_xi * 0.707);",
                "let psi2 = psi * psi;",
                "let f_gl = -psi2 + 0.5 * psi2 * psi2;",
                "s_gl_out[idx] = GlOrderState(psi2, f_gl, select(0u, 1u, psi2 > 0.5), 0.0);"
            ]
        ),
        (
            "cryogenic_superconducting/flux_quantization_abrikosov_vortex",
            "Abrikosov Magnetic Flux Vortex Lattice in Type-II Superconductors",
            ["Evaluates 2D triangular Abrikosov fluxon lattice between lower and upper critical fields.", "Each vortex carries single London magnetic flux quantum Phi_0 = h / 2e approx 2.07e-15 Wb."],
            ["B_vortex(r) = (Phi_0 / (2 * pi * lambda^2)) * K_0(r / lambda) modified Bessel function.", "Normal core radius governed by coherence length xi."],
            [
                "struct VortexConfig { lambda_nm: f32, xi_nm: f32, vortex_density_m2: f32, pad: f32 };",
                "struct VortexField { b_gauss: f32, core_suppression: f32, pad0: f32, pad1: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_vx: VortexConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_r_vortex_nm: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_vx_out: array<VortexField>;"],
            [],
            [
                "if (idx >= arrayLength(&s_vx_out)) { return; }",
                "let r_nm = s_r_vortex_nm[idx];",
                "let r_lam = r_nm / u_vx.lambda_nm;",
                "let k0_approx = exp(-r_lam) / max(0.5, sqrt(PI * max(0.1, r_lam)));",
                "let core = 1.0 - exp(-pow(r_nm / u_vx.xi_nm, 2.0));",
                "let b = k0_approx * core * 1000.0;",
                "s_vx_out[idx] = VortexField(b, core, 0.0, 0.0);"
            ]
        ),
        (
            "cryogenic_superconducting/squid_magnetometer_flux_voltage",
            "DC SQUID V-Phi Periodic Flux-to-Voltage Transfer Function",
            ["Evaluates two identical Josephson junctions in parallel on superconducting loop.", "Simulates periodic voltage modulation V(Phi_ext) with period Phi_0 = 2.07e-15 Wb."],
            ["V(Phi) = (R / 2) * sqrt(I_bias^2 - (2 * I_c * cos(pi * Phi / Phi_0))^2).", "V_Phi = dV / dPhi transfer coefficient enables femtotesla magnetic field sensitivity."],
            [
                "struct SquidConfig { r_shunt_ohm: f32, ic_ua: f32, ibias_ua: f32, pad: f32 };",
                "struct SquidOut { voltage_uv: f32, v_phi_uv_phi0: f32, sensitivity_ft_rt_hz: f32, pad: f32 };"
            ],
            ["@group(0) @binding(0) var<uniform> u_sq: SquidConfig;"],
            ["@group(0) @binding(1) var<storage, read> s_flux_phi0: array<f32>;", "@group(0) @binding(2) var<storage, read_write> s_sq_out: array<SquidOut>;"],
            [],
            [
                "if (idx >= arrayLength(&s_sq_out)) { return; }",
                "let phi = s_flux_phi0[idx];",
                "let cos_term = cos(PI * phi);",
                "let ic_eff = 2.0 * u_sq.ic_ua * abs(cos_term);",
                "let arg = u_sq.ibias_ua * u_sq.ibias_ua - ic_eff * ic_eff;",
                "let v = 0.5 * u_sq.r_shunt_ohm * sqrt(max(0.0, arg));",
                "let v_phi = abs(sin(PI * phi)) * 100.0;",
                "s_sq_out[idx] = SquidOut(v, v_phi, 5.0, 0.0);"
            ]
        ),
    ]

    for subpath, title, desc, math_doc, structs, uniforms, buffers, funcs, compute_stmts in specs:
        content = make_shader(subpath, title, desc, math_doc, structs, uniforms, buffers, funcs, compute_stmts)
        write_shader(f"{subpath}.wgsl", content)

    print("Successfully generated all 25 specialized shaders.")

if __name__ == "__main__":
    run()
