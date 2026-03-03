// Shroud :: Airborne Particulate & Soot Suspension Dynamics
// Simulates Stokes settling velocity and radiative thermal absorption.

struct ParticulatePhysicsUniforms {
    particle_density_kg_m3: f32,
    dynamic_viscosity_pa_s: f32,
    gravity_m_s2: f32,
    delta_time_s: f32,
};

struct AerosolParticle {
    position: vec3<f32>,
    radius_m: f32,
    velocity: vec3<f32>,
    mass_kg: f32,
};

// Computes Stokes terminal settling velocity v = (2/9) * r^2 * g * (rho_p - rho_f) / mu
fn stokes_settling_velocity(r: f32, rho_p: f32, mu: f32, g: f32) -> f32 {
    let rho_air = 1.225;
    return (2.0 / 9.0) * r * r * g * (rho_p - rho_air) / max(1e-6, mu);
}

@group(0) @binding(0) var<uniform> u_part_params: ParticulatePhysicsUniforms;
@group(0) @binding(1) var<storage, read_write> s_particles: array<AerosolParticle>;

@compute @workgroup_size(64, 1, 1)
fn cs_particulate_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_particles)) {
        return;
    }

    var p = s_particles[idx];
    let v_settle = stokes_settling_velocity(
        p.radius_m,
        u_part_params.particle_density_kg_m3,
        u_part_params.dynamic_viscosity_pa_s,
        u_part_params.gravity_m_s2
    );

    p.velocity.y = -v_settle;
    p.position += p.velocity * u_part_params.delta_time_s;

    s_particles[idx] = p;
}
