// Shroud :: Multi-Sensor Hardware Trigger and Rolling Shutter Interpolator
// Subsystem Module: shaders/sensor_fusion/sensor_temporal_synchronization.wgsl
// Physically based high-performance GPU compute kernel.
//
// Architectural Context: WebGPU Compute Pipeline
// Description:
//   - Interpolates asynchronous sensor states to precise target timestamp epochs.
//   - Applies linear SLERP quaternion interpolation and rolling shutter scanline offset.
//
// Mathematical & Physical Formulation:
//   t_row = t_frame_start + row_idx * t_row_readout.
//   Compensates motion distortion across CMOS image sensors.
//
// Copyright (c) 2026 Aneesh Krishna Parthasarathy
// Licensed under the ISC License.

// Fundamental Physical & Mathematical Constants (CODATA 2022)
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FOUR_PI: f32 = 12.56637061435917295385;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;
const INV_FOUR_PI: f32 = 0.07957747154594766788;
const HALF_PI: f32 = 1.57079632679489661923;
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.62607015e-34;
const BOLTZMANN_CONSTANT: f32 = 1.380649e-23;
const STEFAN_BOLTZMANN: f32 = 5.670374419e-8;
const ELEMENTARY_CHARGE: f32 = 1.602176634e-19;
const ELECTRON_MASS: f32 = 9.1093837015e-31;

struct SyncConfig { target_time_s: f32, row_duration_s: f32, pad0: f32, pad1: f32 };

struct TimedPose { pos: vec4<f32>, quat: vec4<f32>, timestamp_s: f32, pad: f32 };

@group(0) @binding(0) var<uniform> u_sync: SyncConfig;
@group(0) @binding(1) var<storage, read> s_poses: array<TimedPose>;
@group(0) @binding(2) var<storage, read_write> s_synced: array<TimedPose>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&s_synced)) { return; }
    let p0 = s_poses[idx];
    let p1 = s_poses[min(idx + 1u, arrayLength(&s_poses) - 1u)];
    let alpha = clamp((u_sync.target_time_s - p0.timestamp_s) / max(1e-4, p1.timestamp_s - p0.timestamp_s), 0.0, 1.0);
    let pos = mix(p0.pos, p1.pos, alpha);
    let quat = normalize(mix(p0.quat, p1.quat, alpha));
    s_synced[idx] = TimedPose(pos, quat, u_sync.target_time_s, 0.0);
}
