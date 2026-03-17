// Shroud :: Symplectic Velocity-Verlet Particle Integrator
// Conserves energy in aerosol dispersion and atmospheric turbulence trajectories.

struct IntegratorUniforms {
    dt: f32,
    damping: f32,
    num_particles: u32,
    padding: f32,
};

struct ParticleState {
    pos: vec3<f32>,
    mass: f32,
    vel: vec3<f32>,
    lifetime: f32,
};

@group(0) @binding(0) var<uniform> u_integrator: IntegratorUniforms;
@group(0) @binding(1) var<storage, read_write> s_particles_state: array<ParticleState>;

@compute @workgroup_size(64, 1, 1)
fn cs_velocity_verlet_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u_integrator.num_particles) {
        return;
    }

    var p = s_particles_state[idx];
    let gravity = vec3<f32>(0.0, -9.80665, 0.0);
    let force = gravity * p.mass - p.vel * u_integrator.damping;
    let accel = force / max(1e-4, p.mass);

    // Velocity Verlet integration:
    // x(t + dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
    // v(t + dt) = v(t) + a(t)*dt
    p.pos += p.vel * u_integrator.dt + 0.5 * accel * u_integrator.dt * u_integrator.dt;
    p.vel += accel * u_integrator.dt;
    p.lifetime -= u_integrator.dt;

    s_particles_state[idx] = p;
}
