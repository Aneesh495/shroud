# Rhai Scripting & Scenario DSL Reference

Shroud embeds a sandboxed Rhai 1.20 scripting engine (`crates/shroud-rhai`) to author weather sweeps, dynamic vehicle trajectories, material calibrations, and WGSL graph verification suites without recompiling host binaries.

## Execution Sandboxing & Safety Quotas

Defined in `crates/shroud-rhai/src/engine.rs`:
- **Maximum Operations**: 50,000,000 VM steps per evaluation
- **Maximum Call Stack Depth**: 128 frames
- **Maximum Expression Nesting**: 64 levels
- **Concurrency Safety**: All custom registered types implement `Clone + Send + Sync + 'static`

## Registered Types

### `Vec3`
Constructed via `vec3(x, y, z)` (supports both integer and floating-point arguments).
- Properties: `.x`, `.y`, `.z` (get/set `f64`)
- Operators: `+`, `-`, `*` (scalar left or right multiplication)
- Methods: `.dot(other)`, `.cross(other)`, `.length()`, `.normalize()`, `.add(other)`, `.sub(other)`, `.scale(s)`, `.distance(other)`, `.lerp(other, t)`

### `Ray`
Constructed via `ray(origin_vec3, dir_vec3)` (direction is normalized automatically).
- Methods:
  - `.at(t)` -> `Vec3` at parametric distance `t`
  - `.intersect_box(min_vec3, max_vec3)` -> `f64` entry distance `t_enter`, or `-1.0` on miss

### `LidarSensor`
Constructed via `lidar_905nm()` or `lidar_config(wavelength_nm, pulse_energy_j, aperture_m, divergence_rad)`.
- Methods:
  - `.expected_photons(range_m, reflectance, cos_incident, tau)` -> `f64`
  - `.wavelength_nm()` -> `f64`
  - `.max_range_m()` -> `f64`

### `FmcwRadar`
Constructed via `fmcw_config(center_freq_hz, sweep_bw_hz, chirp_duration_s, power_w)`.
- Methods:
  - `.beat_up(range_m, radial_vel_m_s)` -> `f64`
  - `.beat_down(range_m, radial_vel_m_s)` -> `f64`
  - `.solve_range(f_up, f_down)` -> `f64`
  - `.solve_velocity(f_up, f_down)` -> `f64`

### `Scenario`
Constructed via `scenario(name_str)`.
- Methods:
  - `.set_fog(sigma_t, asymmetry_g)`
  - `.set_media(sigma_a, sigma_s, asymmetry_g)`
  - `.set_solar_zenith(deg)`
  - `.set_ambient_temp(temp_k)`
  - `.add_target(name, min_vec3, max_vec3, reflectance, temp_k)`
  - `.target_count()` -> `i64`
  - `.extinction()` -> `f64`
  - `.albedo()` -> `f64`

## Free Physics, Optics & Validation Functions

| Function | Signature | Description |
| --- | --- | --- |
| `eval_henyey_greenstein` | `(cos_theta, g) -> f64` | Single-lobe HG phase function |
| `eval_double_hg` | `(cos_theta, gf, gb, wf) -> f64` | Forward-backward blended HG phase |
| `eval_rayleigh_phase` | `(cos_theta) -> f64` | Molecular Rayleigh phase function |
| `beer_lambert` | `(sigma_t, dist_m) -> f64` | Homogeneous exponential transmittance |
| `us_std_density` | `(alt_m) -> f64` | US Standard Atmosphere density ratio |
| `eval_lwir_radiance` | `(temp_k) -> f64` | 8 to 14 um Planck radiance integral |
| `eval_blackbody_peak_nm` | `(temp_k) -> f64` | Spectral emission peak wavelength (nm) |
| `eval_blackbody_rgb` | `(temp_k) -> [f64; 3]` | CIE 1931 XYZ to linear sRGB colorimetry |
| `calculate_lidar_photons` | `(range_m, refl, tau) -> f64` | Default 905 nm LiDAR return photons |
| `morton_3d` | `(norm_vec3) -> i64` | 30-bit Z-order curve index on `[0, 1]^3` |
| `brown_conrady_distort` | `(u, v, k1, k2, p1, p2, k3) -> [f64; 2]` | Radial + tangential lens distortion |
| `speed_of_light` | `() -> f64` | Exact SI speed of light (299792458.0 m/s) |
| `validate_wgsl` | `(path_str) -> bool` | Naga AST parse & typecheck of WGSL file |
| `assert_close` | `(actual, expected, tol)` | Fails cleanly with Rhai `EvalAltResult` |
| `assert_true` | `(cond, msg)` | Boolean assertion returning `EvalAltResult` |
| `assert_gt` / `assert_lt` | `(a, b)` | Overloaded for both `f64` and `i64` |

## Example Script

```rhai
let sc = scenario("coastal_advection_fog");
sc.set_fog(0.0434, 0.89);

let lidar = lidar_config(1550.0, 12.0e-6, 0.035, 0.0008);
let tau = beer_lambert(sc.extinction() * 0.78, 75.0);
let photons = lidar.expected_photons(75.0, 0.40, 1.0, tau);
assert_gt(photons, 0.0);
assert_true(validate_wgsl("shaders/volumetric/delta_tracking_woodcock.wgsl"), "Woodcock valid");
```
