//! Rhai engine bindings for shroud-core primitives, optical physics, sensor models, and WGSL validation.

use rhai::{Array, CustomType, Dynamic, Engine, EvalAltResult, TypeBuilder};
use shroud_core::atmospheric::{
    beer_lambert_transmittance, double_henyey_greenstein, henyey_greenstein, rayleigh_phase,
    us_standard_atmosphere_density_ratio, MediaProperties,
};
use shroud_core::bvh_types::morton_3d;
use shroud_core::constants::SPEED_OF_LIGHT;
use shroud_core::math::{Aabb, Ray, Vec2, Vec3};
use shroud_core::sensors::{DistortionCoefficients, FmcwConfig, LidarTransceiverConfig};
use shroud_core::spectral::{evaluate_lwir_radiance, SampledSpectrum};
use shroud_validator::validate_wgsl_source;
use std::path::Path;

use crate::scenario::SimulationEnvironment;

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
            .with_fn("add", Self::add)
            .with_fn("sub", Self::sub)
            .with_fn("scale", Self::scale)
            .with_fn("distance", Self::distance)
            .with_fn("lerp", Self::lerp)
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

    pub fn add(&mut self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    pub fn sub(&mut self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    pub fn scale(&mut self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn distance(&mut self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn lerp(&mut self, other: Self, t: f64) -> Self {
        let a = t.clamp(0.0, 1.0);
        Self {
            x: self.x + (other.x - self.x) * a,
            y: self.y + (other.y - self.y) * a,
            z: self.z + (other.z - self.z) * a,
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

#[derive(Clone, Debug)]
pub struct ScriptRay {
    pub origin: ScriptVec3,
    pub direction: ScriptVec3,
}

impl CustomType for ScriptRay {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("Ray")
            .with_fn("ray", Self::new)
            .with_fn("at", Self::at)
            .with_fn("intersect_box", Self::intersect_box);
    }
}

impl ScriptRay {
    pub fn new(origin: ScriptVec3, mut direction: ScriptVec3) -> Self {
        let dir = direction.normalize();
        Self { origin, direction: dir }
    }

    pub fn at(&mut self, t: f64) -> ScriptVec3 {
        ScriptVec3 {
            x: self.origin.x + self.direction.x * t,
            y: self.origin.y + self.direction.y * t,
            z: self.origin.z + self.direction.z * t,
        }
    }

    pub fn intersect_box(&mut self, box_min: ScriptVec3, box_max: ScriptVec3) -> f64 {
        let aabb = Aabb::new(box_min.to_core(), box_max.to_core());
        let r = Ray::new(self.origin.to_core(), self.direction.to_core(), 0.0, 1e6);
        match aabb.intersect_ray(&r) {
            Some((t_enter, _)) => t_enter as f64,
            None => -1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScriptLidar {
    pub config: LidarTransceiverConfig,
}

impl CustomType for ScriptLidar {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("LidarSensor")
            .with_fn("expected_photons", Self::expected_photons)
            .with_fn("wavelength_nm", Self::wavelength_nm)
            .with_fn("max_range_m", Self::max_range_m);
    }
}

impl ScriptLidar {
    pub fn expected_photons(&mut self, range_m: f64, reflectance: f64, cos_inc: f64, tau: f64) -> f64 {
        self.config.expected_return_photons(
            range_m as f32,
            reflectance as f32,
            cos_inc as f32,
            tau as f32,
        ) as f64
    }

    pub fn wavelength_nm(&mut self) -> f64 {
        self.config.wavelength_nm as f64
    }

    pub fn max_range_m(&mut self) -> f64 {
        self.config.max_range_m as f64
    }
}

#[derive(Clone, Debug)]
pub struct ScriptFmcw {
    pub config: FmcwConfig,
}

impl CustomType for ScriptFmcw {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("FmcwRadar")
            .with_fn("beat_up", Self::beat_up)
            .with_fn("beat_down", Self::beat_down)
            .with_fn("solve_range", Self::solve_range)
            .with_fn("solve_velocity", Self::solve_velocity);
    }
}

impl ScriptFmcw {
    pub fn beat_up(&mut self, range_m: f64, rel_vel: f64) -> f64 {
        self.config.calculate_beat_frequencies(range_m as f32, rel_vel as f32).0 as f64
    }

    pub fn beat_down(&mut self, range_m: f64, rel_vel: f64) -> f64 {
        self.config.calculate_beat_frequencies(range_m as f32, rel_vel as f32).1 as f64
    }

    pub fn solve_range(&mut self, f_up: f64, f_down: f64) -> f64 {
        self.config.solve_range_and_velocity(f_up as f32, f_down as f32).0 as f64
    }

    pub fn solve_velocity(&mut self, f_up: f64, f_down: f64) -> f64 {
        self.config.solve_range_and_velocity(f_up as f32, f_down as f32).1 as f64
    }
}

#[derive(Clone, Debug)]
pub struct ScriptScenario {
    pub env: SimulationEnvironment,
}

impl CustomType for ScriptScenario {
    fn build(mut builder: TypeBuilder<Self>) {
        builder
            .with_name("Scenario")
            .with_fn("set_fog", Self::set_fog)
            .with_fn("set_media", Self::set_media)
            .with_fn("set_solar_zenith", Self::set_solar_zenith)
            .with_fn("set_ambient_temp", Self::set_ambient_temp)
            .with_fn("add_target", Self::add_target)
            .with_fn("target_count", Self::target_count)
            .with_fn("extinction", Self::extinction)
            .with_fn("albedo", Self::albedo);
    }
}

impl ScriptScenario {
    pub fn set_fog(&mut self, sigma_t: f64, g: f64) {
        self.env.set_fog(sigma_t as f32, g as f32);
    }

    pub fn set_media(&mut self, sigma_a: f64, sigma_s: f64, g: f64) {
        self.env.media = MediaProperties::new(sigma_a as f32, sigma_s as f32, g as f32);
    }

    pub fn set_solar_zenith(&mut self, deg: f64) {
        self.env.solar_zenith_deg = deg as f32;
    }

    pub fn set_ambient_temp(&mut self, temp_k: f64) {
        self.env.ambient_temp_k = temp_k as f32;
    }

    pub fn add_target(
        &mut self,
        name: &str,
        min_v: ScriptVec3,
        max_v: ScriptVec3,
        reflectance: f64,
        temp_k: f64,
    ) {
        self.env.add_target(
            name,
            min_v.to_core(),
            max_v.to_core(),
            reflectance as f32,
            temp_k as f32,
        );
    }

    pub fn target_count(&mut self) -> i64 {
        self.env.targets.len() as i64
    }

    pub fn extinction(&mut self) -> f64 {
        self.env.media.sigma_t() as f64
    }

    pub fn albedo(&mut self) -> f64 {
        self.env.media.albedo() as f64
    }
}

/// Registers all Shroud domain types and functions to the Rhai engine.
pub fn register_shroud_types(engine: &mut Engine) {
    engine.build_type::<ScriptVec3>();
    engine.build_type::<ScriptRay>();
    engine.build_type::<ScriptLidar>();
    engine.build_type::<ScriptFmcw>();
    engine.build_type::<ScriptScenario>();

    // Vector constructors and arithmetic operators
    engine.register_fn("vec3", |x: f64, y: f64, z: f64| ScriptVec3::new(x, y, z));
    engine.register_fn("vec3", |x: i64, y: i64, z: i64| {
        ScriptVec3::new(x as f64, y as f64, z as f64)
    });
    engine.register_fn("+", |mut a: ScriptVec3, b: ScriptVec3| a.add(b));
    engine.register_fn("-", |mut a: ScriptVec3, b: ScriptVec3| a.sub(b));
    engine.register_fn("*", |mut a: ScriptVec3, s: f64| a.scale(s));
    engine.register_fn("*", |s: f64, mut a: ScriptVec3| a.scale(s));

    // Ray & Scenario constructors
    engine.register_fn("ray", |o: ScriptVec3, d: ScriptVec3| ScriptRay::new(o, d));
    engine.register_fn("scenario", |name: &str| {
        let mut env = SimulationEnvironment::default();
        env.scenario_name = name.to_string();
        ScriptScenario { env }
    });

    // Sensor constructors
    engine.register_fn("lidar_905nm", || ScriptLidar {
        config: LidarTransceiverConfig::default_automotive_905nm(),
    });
    engine.register_fn(
        "lidar_config",
        |wavelength_nm: f64, energy_j: f64, aperture_m: f64, div_rad: f64| {
            let mut cfg = LidarTransceiverConfig::default_automotive_905nm();
            cfg.wavelength_nm = wavelength_nm as f32;
            cfg.pulse_energy_joules = energy_j as f32;
            cfg.aperture_diameter_m = aperture_m as f32;
            cfg.beam_divergence_rad = div_rad as f32;
            ScriptLidar { config: cfg }
        },
    );
    engine.register_fn(
        "fmcw_config",
        |center_hz: f64, bw_hz: f64, duration_s: f64, power_w: f64| ScriptFmcw {
            config: FmcwConfig {
                center_frequency_hz: center_hz as f32,
                sweep_bandwidth_hz: bw_hz as f32,
                chirp_duration_s: duration_s as f32,
                power_watts: power_w as f32,
                _pad: [0.0; 4],
            },
        },
    );

    // Optical and radiative transfer evaluations
    engine.register_fn("eval_henyey_greenstein", |cos_theta: f64, g: f64| {
        henyey_greenstein(cos_theta as f32, g as f32) as f64
    });

    engine.register_fn(
        "eval_double_hg",
        |cos_theta: f64, g_fwd: f64, g_back: f64, w_fwd: f64| {
            double_henyey_greenstein(
                cos_theta as f32,
                g_fwd as f32,
                g_back as f32,
                w_fwd as f32,
            ) as f64
        },
    );

    engine.register_fn("eval_rayleigh_phase", |cos_theta: f64| {
        rayleigh_phase(cos_theta as f32) as f64
    });

    engine.register_fn("beer_lambert", |sigma_t: f64, dist_m: f64| {
        beer_lambert_transmittance(sigma_t as f32, dist_m as f32) as f64
    });

    engine.register_fn("us_std_density", |alt_m: f64| {
        us_standard_atmosphere_density_ratio(alt_m as f32) as f64
    });

    engine.register_fn("eval_lwir_radiance", |temp_k: f64| {
        evaluate_lwir_radiance(temp_k as f32, 64) as f64
    });

    engine.register_fn("eval_blackbody_peak_nm", |temp_k: f64| {
        let s = SampledSpectrum::blackbody(temp_k as f32);
        let mut max_i = 0usize;
        let mut max_v = -1.0f32;
        for (i, &val) in s.values.iter().enumerate() {
            if val > max_v {
                max_v = val;
                max_i = i;
            }
        }
        400.0 + (max_i as f64) * 10.0
    });

    engine.register_fn("eval_blackbody_rgb", |temp_k: f64| -> Array {
        let rgb = SampledSpectrum::blackbody(temp_k as f32).to_rgb();
        vec![
            Dynamic::from(rgb.x as f64),
            Dynamic::from(rgb.y as f64),
            Dynamic::from(rgb.z as f64),
        ]
    });

    engine.register_fn(
        "calculate_lidar_photons",
        |range_m: f64, reflectance: f64, atmospheric_tau: f64| {
            let cfg = LidarTransceiverConfig::default_automotive_905nm();
            cfg.expected_return_photons(
                range_m as f32,
                reflectance as f32,
                1.0,
                atmospheric_tau as f32,
            ) as f64
        },
    );

    engine.register_fn("morton_3d", |v: ScriptVec3| -> i64 {
        morton_3d(v.to_core()) as i64
    });

    engine.register_fn(
        "brown_conrady_distort",
        |u: f64, v: f64, k1: f64, k2: f64, p1: f64, p2: f64, k3: f64| -> Array {
            let d = DistortionCoefficients {
                k1: k1 as f32,
                k2: k2 as f32,
                p1: p1 as f32,
                p2: p2 as f32,
                k3: k3 as f32,
                _pad: [0.0; 3],
            };
            let res = d.apply(Vec2::new(u as f32, v as f32));
            vec![Dynamic::from(res.x as f64), Dynamic::from(res.y as f64)]
        },
    );

    engine.register_fn("speed_of_light", || SPEED_OF_LIGHT);

    // Shader validation helper for scenario scripts checking referenced WGSL kernels
    engine.register_fn("validate_wgsl", |path_str: &str| -> bool {
        let p = Path::new(path_str);
        match std::fs::read_to_string(p) {
            Ok(src) => validate_wgsl_source(path_str, &src).is_valid,
            Err(_) => false,
        }
    });

    // Clean assertion functions returning Result rather than panicking
    engine.register_fn(
        "assert_close",
        |actual: f64, expected: f64, tol: f64| -> Result<(), Box<EvalAltResult>> {
            if (actual - expected).abs() > tol {
                Err(format!(
                    "Assertion failed: actual {actual} != expected {expected} (tol {tol})"
                )
                .into())
            } else {
                Ok(())
            }
        },
    );

    engine.register_fn(
        "assert_true",
        |cond: bool, msg: &str| -> Result<(), Box<EvalAltResult>> {
            if !cond {
                Err(format!("Assertion failed: {msg}").into())
            } else {
                Ok(())
            }
        },
    );

    engine.register_fn(
        "assert_gt",
        |a: f64, b: f64| -> Result<(), Box<EvalAltResult>> {
            if !(a > b) {
                Err(format!("Assertion failed: expected {a} > {b}").into())
            } else {
                Ok(())
            }
        },
    );

    engine.register_fn(
        "assert_gt",
        |a: i64, b: i64| -> Result<(), Box<EvalAltResult>> {
            if !(a > b) {
                Err(format!("Assertion failed: expected {a} > {b}").into())
            } else {
                Ok(())
            }
        },
    );

    engine.register_fn(
        "assert_lt",
        |a: f64, b: f64| -> Result<(), Box<EvalAltResult>> {
            if !(a < b) {
                Err(format!("Assertion failed: expected {a} < {b}").into())
            } else {
                Ok(())
            }
        },
    );

    engine.register_fn(
        "assert_lt",
        |a: i64, b: i64| -> Result<(), Box<EvalAltResult>> {
            if !(a < b) {
                Err(format!("Assertion failed: expected {a} < {b}").into())
            } else {
                Ok(())
            }
        },
    );
}
