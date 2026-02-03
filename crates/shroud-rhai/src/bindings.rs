//! Rhai engine bindings for shroud-core primitives and math.

use rhai::{CustomType, Engine, TypeBuilder};
use shroud_core::atmospheric::{henyey_greenstein, rayleigh_phase};
use shroud_core::math::Vec3;
use shroud_core::sensors::LidarTransceiverConfig;
use shroud_core::spectral::evaluate_lwir_radiance;

#[derive(Clone, Debug, PartialEq)]
pub struct ScriptVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl CustomType for ScriptVec3 {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("Vec3")
            .with_fn("vec3", Self::new)
            .with_fn("dot", Self::dot)
            .with_fn("cross", Self::cross)
            .with_fn("length", Self::length)
            .with_fn("normalize", Self::normalize)
            .with_get_set("x", Self::get_x, Self::set_x)
            .with_get_set("y", Self::get_y, Self::set_y)
            .with_get_set("z", Self::get_z, Self::set_z);
    }
}

impl ScriptVec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn dot(&mut self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(&mut self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn length(&mut self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&mut self) -> Self {
        let len = self.length();
        if len > 1e-8 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            Self { x: 0.0, y: 0.0, z: 0.0 }
        }
    }

    pub fn get_x(&mut self) -> f64 { self.x }
    pub fn set_x(&mut self, val: f64) { self.x = val; }
    pub fn get_y(&mut self) -> f64 { self.y }
    pub fn set_y(&mut self, val: f64) { self.y = val; }
    pub fn get_z(&mut self) -> f64 { self.z }
    pub fn set_z(&mut self, val: f64) { self.z = val; }

    pub fn to_core(&self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
}

/// Registers all Shroud domain types and functions to the Rhai engine.
pub fn register_shroud_types(engine: &mut Engine) {
    engine.build_type::<ScriptVec3>();

    // Free helper functions
    engine.register_fn("vec3", |x: f64, y: f64, z: f64| ScriptVec3::new(x, y, z));
    engine.register_fn("vec3", |x: i64, y: i64, z: i64| {
        ScriptVec3::new(x as f64, y as f64, z as f64)
    });

    // Optical and radiative transfer evaluations
    engine.register_fn("eval_henyey_greenstein", |cos_theta: f64, g: f64| {
        henyey_greenstein(cos_theta as f32, g as f32) as f64
    });

    engine.register_fn("eval_rayleigh_phase", |cos_theta: f64| {
        rayleigh_phase(cos_theta as f32) as f64
    });

    engine.register_fn("eval_lwir_radiance", |temp_k: f64| {
        evaluate_lwir_radiance(temp_k as f32, 64) as f64
    });

    engine.register_fn("calculate_lidar_photons", |range_m: f64, reflectance: f64, atmospheric_tau: f64| {
        let cfg = LidarTransceiverConfig::default_automotive_905nm();
        cfg.expected_return_photons(range_m as f32, reflectance as f32, 1.0, atmospheric_tau as f32) as f64
    });

    engine.register_fn("assert_close", |actual: f64, expected: f64, tol: f64| {
        if (actual - expected).abs() > tol {
            panic!("Assertion failed: actual {actual} != expected {expected} (tol {tol})");
        }
    });
}
