//! Atmospheric radiative transfer, participating media extinction, and phase functions.

use crate::constants::{STANDARD_GRAVITY, STANDARD_SEA_LEVEL_TEMPERATURE};
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Atmospheric phase function evaluation parameterization.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MediaProperties {
    pub sigma_a: f32, // Absorption coefficient (1 / m)
    pub sigma_s: f32, // Scattering coefficient (1 / m)
    pub g: f32,       // Henyey-Greenstein asymmetry parameter (-1.0 to 1.0)
    pub _pad: f32,
}

impl MediaProperties {
    pub const VACUUM: Self = Self {
        sigma_a: 0.0,
        sigma_s: 0.0,
        g: 0.0,
        _pad: 0.0,
    };

    pub fn new(sigma_a: f32, sigma_s: f32, g: f32) -> Self {
        Self { sigma_a, sigma_s, g, _pad: 0.0 }
    }

    #[inline]
    pub fn sigma_t(&self) -> f32 {
        self.sigma_a + self.sigma_s
    }

    #[inline]
    pub fn albedo(&self) -> f32 {
        let st = self.sigma_t();
        if st > 1e-8 {
            self.sigma_s / st
        } else {
            0.0
        }
    }
}

/// Evaluates Henyey-Greenstein phase function for scattering angle cosine.
#[inline]
pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    if denom <= 0.0 {
        return 0.0;
    }
    (1.0 - g * g) / (4.0 * std::f32::consts::PI * denom.powf(1.5))
}

/// Evaluates Rayleigh phase function for molecular scattering.
#[inline]
pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    (3.0 / (16.0 * std::f32::consts::PI)) * (1.0 + cos_theta * cos_theta)
}

/// Double Henyey-Greenstein phase function for forward-backward marine fog and aerosol droplets.
#[inline]
pub fn double_henyey_greenstein(cos_theta: f32, g_forward: f32, g_backward: f32, weight_forward: f32) -> f32 {
    let pf = henyey_greenstein(cos_theta, g_forward);
    let pb = henyey_greenstein(cos_theta, g_backward);
    weight_forward * pf + (1.0 - weight_forward) * pb
}

/// Computes air density ratio relative to sea level at a given altitude in meters (US Standard Atmosphere).
pub fn us_standard_atmosphere_density_ratio(altitude_m: f32) -> f32 {
    if altitude_m <= 11_000.0 {
        // Troposphere with standard lapse rate -6.5 K / km
        let t0 = STANDARD_SEA_LEVEL_TEMPERATURE as f32;
        let lapse_rate = 0.0065;
        let t = t0 - lapse_rate * altitude_m;
        let exponent = (STANDARD_GRAVITY as f32) / (287.05 * lapse_rate) - 1.0;
        (t / t0).powf(exponent)
    } else {
        // Stratosphere isothermal approximation up to 20 km
        let base_density = 0.3639; // density ratio at 11 km
        let h_diff = altitude_m - 11_000.0;
        let scale_height = 6340.0;
        base_density * (-h_diff / scale_height).exp()
    }
}

/// Computes Beer-Lambert transmittance for a uniform medium over distance in meters.
#[inline]
pub fn beer_lambert_transmittance(sigma_t: f32, distance_m: f32) -> f32 {
    (-sigma_t * distance_m).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_function_normalization() {
        // Numerical integration of Henyey-Greenstein over the unit sphere must equal 1.0
        let g = 0.65;
        let num_steps = 1000;
        let d_cos = 2.0 / num_steps as f32;
        let mut integral = 0.0;

        for i in 0..num_steps {
            let cos_theta = -1.0 + (i as f32 + 0.5) * d_cos;
            let p = henyey_greenstein(cos_theta, g);
            // Integral over solid angle: 2 * pi * \int_{-1}^1 p(cos_theta) d(cos_theta)
            integral += 2.0 * std::f32::consts::PI * p * d_cos;
        }

        assert!((integral - 1.0).abs() < 1e-3, "Integral was {integral}");
    }

    #[test]
    fn test_rayleigh_phase_normalization() {
        let num_steps = 1000;
        let d_cos = 2.0 / num_steps as f32;
        let mut integral = 0.0;

        for i in 0..num_steps {
            let cos_theta = -1.0 + (i as f32 + 0.5) * d_cos;
            let p = rayleigh_phase(cos_theta);
            integral += 2.0 * std::f32::consts::PI * p * d_cos;
        }

        assert!((integral - 1.0).abs() < 1e-4, "Rayleigh integral was {integral}");
    }

    #[test]
    fn test_atmospheric_density_decrease() {
        let rho_0 = us_standard_atmosphere_density_ratio(0.0);
        let rho_5k = us_standard_atmosphere_density_ratio(5000.0);
        let rho_10k = us_standard_atmosphere_density_ratio(10000.0);
        assert!((rho_0 - 1.0).abs() < 1e-4);
        assert!(rho_5k < rho_0 && rho_5k > 0.5);
        assert!(rho_10k < rho_5k && rho_10k > 0.3);
    }
}
