//! Scenario description and parameter containers managed by Rhai scripts.

use shroud_core::atmospheric::MediaProperties;
use shroud_core::math::{Aabb, Vec3};
use shroud_core::sensors::LidarTransceiverConfig;

#[derive(Clone, Debug)]
pub struct SimulationTarget {
    pub name: String,
    pub bounds: Aabb,
    pub reflectance: f32,
    pub temperature_k: f32,
    pub velocity: Vec3,
}

#[derive(Clone, Debug)]
pub struct SimulationEnvironment {
    pub scenario_name: String,
    pub media: MediaProperties,
    pub solar_zenith_deg: f32,
    pub ambient_temp_k: f32,
    pub targets: Vec<SimulationTarget>,
    pub lidar_config: LidarTransceiverConfig,
}

impl Default for SimulationEnvironment {
    fn default() -> Self {
        Self {
            scenario_name: "default_clear_sky".to_string(),
            media: MediaProperties::VACUUM,
            solar_zenith_deg: 45.0,
            ambient_temp_k: 293.15,
            targets: Vec::new(),
            lidar_config: LidarTransceiverConfig::default_automotive_905nm(),
        }
    }
}

impl SimulationEnvironment {
    pub fn add_target(&mut self, name: &str, min: Vec3, max: Vec3, reflectance: f32, temp_k: f32) {
        self.targets.push(SimulationTarget {
            name: name.to_string(),
            bounds: Aabb::new(min, max),
            reflectance,
            temperature_k: temp_k,
            velocity: Vec3::ZERO,
        });
    }

    pub fn set_fog(&mut self, extinction_sigma_t: f32, asymmetry_g: f32) {
        self.media = MediaProperties::new(extinction_sigma_t * 0.1, extinction_sigma_t * 0.9, asymmetry_g);
    }
}
