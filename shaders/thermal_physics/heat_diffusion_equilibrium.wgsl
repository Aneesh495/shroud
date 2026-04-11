// Shroud :: Finite Difference Surface Heat Diffusion & Radiation Solver
// Solves 2D/3D thermal diffusion equation with Stefan-Boltzmann radiative cooling.

const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;

struct ThermalSimParams {
    thermal_conductivity_k: f32,
    density_rho: f32,
    specific_heat_cp: f32,
    delta_time_s: f32,
    grid_spacing_m: f32,
    ambient_temp_k: f32,
    sky_temp_k: f32,
    emissivity: f32,
};

@group(0) @binding(0) var<uniform> u_therm: ThermalSimParams;
@group(0) @binding(1) var<storage, read> s_temp_prev: array<f32>;
@group(0) @binding(2) var<storage, read_write> s_temp_next: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_heat_diffusion_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = arrayLength(&s_temp_next);
    if (idx >= n) {
        return;
    }
    let t_c = s_temp_prev[idx];
    let t_l = s_temp_prev[select(idx - 1u, 0u, idx == 0u)];
    let t_r = s_temp_prev[select(idx + 1u, n - 1u, idx == n - 1u)];

    let dx2 = u_therm.grid_spacing_m * u_therm.grid_spacing_m;
    let laplacian = (t_l - 2.0 * t_c + t_r) / max(1e-6, dx2);

    let alpha = u_therm.thermal_conductivity_k / (u_therm.density_rho * u_therm.specific_heat_cp);
    let q_cond = alpha * laplacian;

    // Stefan-Boltzmann radiative cooling term
    let q_rad = u_therm.emissivity * STEFAN_BOLTZMANN * (pow(t_c, 4.0) - pow(u_therm.sky_temp_k, 4.0));
    let t_new = t_c + (q_cond - q_rad * 0.01) * u_therm.delta_time_s;

    s_temp_next[idx] = max(10.0, t_new);
}
