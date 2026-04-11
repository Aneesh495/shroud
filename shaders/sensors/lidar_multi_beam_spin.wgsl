// Shroud :: Multi-Beam Spinning LiDAR Firing Pattern Scheduler
// Calculates 3D beam trajectories for 32/64/128-beam mechanical spinning LiDARs.

const PI: f32 = 3.14159265358979323846;

struct SpinningLidarConfig {
    rpm: f32,
    num_lasers: u32,
    elevation_min_deg: f32,
    elevation_max_deg: f32,
    current_timestamp_s: f32,
    pulse_repetition_freq_hz: f32,
    padding: vec2<f32>,
};

struct BeamFiringOutput {
    origin: vec3<f32>,
    direction: vec3<f32>,
    laser_channel: u32,
    firing_time_s: f32,
};

@group(0) @binding(0) var<uniform> u_spin: SpinningLidarConfig;
@group(0) @binding(1) var<storage, read_write> s_firing_beams: array<BeamFiringOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_generate_spinning_lidar_beams(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_firing_beams)) {
        return;
    }
    let channel = idx % u_spin.num_lasers;
    let fire_idx = idx / u_spin.num_lasers;

    // Elevation angle per channel
    let elev_range = u_spin.elevation_max_deg - u_spin.elevation_min_deg;
    let elev_deg = u_spin.elevation_min_deg + (f32(channel) / f32(max(1u, u_spin.num_lasers - 1u))) * elev_range;
    let elev_rad = radians(elev_deg);

    // Azimuth angle from rotational speed
    let rotations_per_sec = u_spin.rpm / 60.0;
    let delta_t = f32(fire_idx) / max(1.0, u_spin.pulse_repetition_freq_hz);
    let azimuth_rad = 2.0 * PI * fract(rotations_per_sec * (u_spin.current_timestamp_s + delta_t));

    let cos_elev = cos(elev_rad);
    let dir = vec3<f32>(cos_elev * sin(azimuth_rad), sin(elev_rad), cos_elev * cos(azimuth_rad));

    var beam: BeamFiringOutput;
    beam.origin = vec3<f32>(0.0, 1.8, 0.0); // Mounted roof height
    beam.direction = normalize(dir);
    beam.laser_channel = channel;
    beam.firing_time_s = u_spin.current_timestamp_s + delta_t;

    s_firing_beams[idx] = beam;
}
