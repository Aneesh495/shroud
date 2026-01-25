//! Sensor models for Time-of-Flight LiDAR, FMCW LiDAR, and Thermal/Optical cameras.

use crate::constants::SPEED_OF_LIGHT;
use crate::math::Vec2;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Brown-Conrady camera lens distortion model.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DistortionCoefficients {
    pub k1: f32, // Radial 2nd order
    pub k2: f32, // Radial 4th order
    pub p1: f32, // Tangential x
    pub p2: f32, // Tangential y
    pub k3: f32, // Radial 6th order
    pub _pad: [f32; 3],
}

impl DistortionCoefficients {
    pub const ZERO: Self = Self {
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
        k3: 0.0,
        _pad: [0.0; 3],
    };

    pub fn apply(&self, uv_norm: Vec2) -> Vec2 {
        let r2 = uv_norm.length_squared();
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let x = uv_norm.x;
        let y = uv_norm.y;

        let dx_tangential = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy_tangential = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;

        Vec2::new(x * radial + dx_tangential, y * radial + dy_tangential)
    }
}

/// Time-of-Flight (ToF) pulsed LiDAR transceiver specification.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LidarTransceiverConfig {
    pub wavelength_nm: f32,         // Typically 905 nm or 1550 nm
    pub pulse_energy_joules: f32,   // Energy per laser pulse (e.g. 1.0e-6 J)
    pub pulse_duration_fwhm_s: f32, // Pulse duration FWHM (e.g. 5.0e-9 s = 5 ns)
    pub beam_divergence_rad: f32,   // Beam divergence half-angle (e.g. 0.0015 rad)
    pub aperture_diameter_m: f32,   // Receiver aperture diameter (e.g. 0.03 m)
    pub optical_efficiency: f32,    // Combined lens transmission and filter efficiency
    pub apd_gain: f32,              // Avalanche photodiode multiplication gain (e.g. 30.0)
    pub apd_excess_noise_factor: f32, // APD excess noise factor F
    pub dark_count_rate_hz: f32,    // Background thermal dark count rate
    pub max_range_m: f32,           // Maximum operational range
    pub range_bin_resolution_m: f32,// Range bin resolution (e.g. 0.03 m)
    pub _pad: f32,
}

impl LidarTransceiverConfig {
    pub fn default_automotive_905nm() -> Self {
        Self {
            wavelength_nm: 905.0,
            pulse_energy_joules: 5.0e-6,
            pulse_duration_fwhm_s: 4.0e-9,
            beam_divergence_rad: 0.001,
            aperture_diameter_m: 0.025,
            optical_efficiency: 0.85,
            apd_gain: 25.0,
            apd_excess_noise_factor: 2.8,
            dark_count_rate_hz: 50_000.0,
            max_range_m: 250.0,
            range_bin_resolution_m: 0.03,
            _pad: 0.0,
        }
    }

    /// Computes returned photon count for a diffuse Lambertian target at range R.
    pub fn expected_return_photons(
        &self,
        range_m: f32,
        target_reflectance: f32,
        cos_incident: f32,
        atmospheric_transmittance: f32,
    ) -> f32 {
        if range_m <= 0.1 {
            return 0.0;
        }

        let photon_energy = (crate::constants::PLANCK_CONSTANT * SPEED_OF_LIGHT)
            / (self.wavelength_nm as f64 * 1e-9);
        let tx_photons = (self.pulse_energy_joules as f64) / photon_energy;

        let aperture_area = std::f64::consts::PI * 0.25 * (self.aperture_diameter_m as f64).powi(2);
        let geometric_factor = aperture_area / (std::f64::consts::PI * (range_m as f64).powi(2));

        let total_photons = tx_photons
            * (target_reflectance as f64)
            * (cos_incident.max(0.0) as f64)
            * (self.optical_efficiency as f64)
            * (atmospheric_transmittance as f64).powi(2)
            * geometric_factor;

        total_photons as f32
    }
}

/// FMCW (Frequency Modulated Continuous Wave) LiDAR chirp parameters.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FmcwConfig {
    pub center_frequency_hz: f32,   // Center optical carrier (e.g. 193.4 THz for 1550 nm)
    pub sweep_bandwidth_hz: f32,    // Chirp sweep bandwidth B (e.g. 500 MHz)
    pub chirp_duration_s: f32,      // Sweep duration T_c (e.g. 20.0e-6 s = 20 us)
    pub power_watts: f32,           // Continuous optical transmit power
    pub _pad: [f32; 4],
}

impl FmcwConfig {
    /// Computes beat frequencies for up-chirp and down-chirp given target range and radial velocity.
    pub fn calculate_beat_frequencies(&self, range_m: f32, radial_velocity_m_s: f32) -> (f32, f32) {
        let c = SPEED_OF_LIGHT as f32;
        let b = self.sweep_bandwidth_hz;
        let t_c = self.chirp_duration_s;
        let lambda = c / self.center_frequency_hz;

        let f_range = (2.0 * b * range_m) / (c * t_c);
        let f_doppler = (2.0 * radial_velocity_m_s) / lambda;

        let f_up = (f_range - f_doppler).abs();
        let f_down = (f_range + f_doppler).abs();

        (f_up, f_down)
    }

    /// Inverts measured beat frequencies to recover true range and radial velocity.
    pub fn solve_range_and_velocity(&self, f_up: f32, f_down: f32) -> (f32, f32) {
        let c = SPEED_OF_LIGHT as f32;
        let b = self.sweep_bandwidth_hz;
        let t_c = self.chirp_duration_s;
        let lambda = c / self.center_frequency_hz;

        let f_range = (f_down + f_up) * 0.5;
        let f_doppler = (f_down - f_up) * 0.5;

        let range = (c * t_c * f_range) / (2.0 * b);
        let velocity = (lambda * f_doppler) / 2.0;

        (range, velocity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lidar_inverse_square_falloff() {
        let cfg = LidarTransceiverConfig::default_automotive_905nm();
        let p_10m = cfg.expected_return_photons(10.0, 0.8, 1.0, 1.0);
        let p_20m = cfg.expected_return_photons(20.0, 0.8, 1.0, 1.0);

        // In pure vacuum (transmittance 1.0), return photon count drops by exactly factor of 4
        let ratio = p_10m / p_20m;
        assert!((ratio - 4.0).abs() < 1e-4, "Ratio was {ratio}");
    }

    #[test]
    fn test_fmcw_range_velocity_recovery() {
        let cfg = FmcwConfig {
            center_frequency_hz: 77.0e9,    // 77 GHz automotive millimeter wave radar
            sweep_bandwidth_hz: 1.0e9,      // 1 GHz sweep bandwidth
            chirp_duration_s: 40.0e-6,      // 40 us sweep duration
            power_watts: 0.1,
            _pad: [0.0; 4],
        };

        let true_range = 85.5; // meters
        let true_velocity = -18.2; // meters/sec relative radial velocity

        let (f_up, f_down) = cfg.calculate_beat_frequencies(true_range, true_velocity);
        let (est_range, est_velocity) = cfg.solve_range_and_velocity(f_up, f_down);

        assert!((est_range - true_range).abs() < 1e-2);
        assert!((est_velocity - true_velocity).abs() < 1e-2);
    }
}
