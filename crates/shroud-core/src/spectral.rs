//! Spectral radiometric representation, Planck blackbody equations, and CIE colorimetry.

use crate::constants::{BOLTZMANN_CONSTANT, PLANCK_CONSTANT, SPEED_OF_LIGHT};
use crate::math::Vec3;
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 31-band sampled spectrum from 400 nm to 700 nm in 10 nm increments.
pub const SPECTRAL_BANDS: usize = 31;
pub const LAMBDA_MIN_NM: f32 = 400.0;
pub const LAMBDA_MAX_NM: f32 = 700.0;
pub const LAMBDA_STEP_NM: f32 = 10.0;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SampledSpectrum {
    pub values: [f32; SPECTRAL_BANDS],
    pub _pad: f32,
}

impl SampledSpectrum {
    pub const ZERO: Self = Self {
        values: [0.0; SPECTRAL_BANDS],
        _pad: 0.0,
    };

    pub fn uniform(v: f32) -> Self {
        Self {
            values: [v; SPECTRAL_BANDS],
            _pad: 0.0,
        }
    }

    /// Evaluates Planck blackbody spectral radiance (W / (m^2 * sr * m)) at a given temperature in Kelvin.
    pub fn blackbody(temperature_k: f32) -> Self {
        let mut s = Self::ZERO;
        let t = temperature_k as f64;
        let c = SPEED_OF_LIGHT;
        let h = PLANCK_CONSTANT;
        let k = BOLTZMANN_CONSTANT;

        for i in 0..SPECTRAL_BANDS {
            let lambda_m = (LAMBDA_MIN_NM + i as f32 * LAMBDA_STEP_NM) as f64 * 1e-9;
            let exponent = (h * c) / (lambda_m * k * t);
            if exponent > 700.0 {
                s.values[i] = 0.0;
            } else {
                let denom = (exponent.exp() - 1.0) * lambda_m.powi(5);
                let radiance = (2.0 * h * c * c) / denom;
                // Scale to W / (m^2 * sr * nm)
                s.values[i] = (radiance * 1e-9) as f32;
            }
        }
        s
    }

    /// Converts sampled spectrum to CIE 1931 XYZ tristimulus values.
    pub fn to_xyz(&self) -> Vec3 {
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        let mut z = 0.0f32;

        for i in 0..SPECTRAL_BANDS {
            let val = self.values[i];
            x += val * CIE_X_FIT[i];
            y += val * CIE_Y_FIT[i];
            z += val * CIE_Z_FIT[i];
        }

        let delta_lambda = LAMBDA_STEP_NM;
        Vec3::new(x * delta_lambda, y * delta_lambda, z * delta_lambda)
    }

    /// Converts sampled spectrum to linear sRGB.
    pub fn to_rgb(&self) -> Vec3 {
        let xyz = self.to_xyz();
        // Standard XYZ to sRGB D65 matrix
        let r = 3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z;
        let g = -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z;
        let b = 0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z;
        Vec3::new(r.max(0.0), g.max(0.0), b.max(0.0))
    }
}

/// Evaluates thermal Planck radiance in long-wave infrared (LWIR, 8 to 14 micrometers) in W / (m^2 * sr).
pub fn evaluate_lwir_radiance(temperature_k: f32, num_steps: usize) -> f32 {
    let t = temperature_k as f64;
    let c = SPEED_OF_LIGHT;
    let h = PLANCK_CONSTANT;
    let k = BOLTZMANN_CONSTANT;

    let lambda_start = 8.0e-6; // 8 um
    let lambda_end = 14.0e-6;  // 14 um
    let step = (lambda_end - lambda_start) / (num_steps as f64);

    let mut total_radiance = 0.0;
    for i in 0..num_steps {
        let lambda_m = lambda_start + (i as f64 + 0.5) * step;
        let exponent = (h * c) / (lambda_m * k * t);
        if exponent < 700.0 {
            let denom = (exponent.exp() - 1.0) * lambda_m.powi(5);
            let radiance = (2.0 * h * c * c) / denom;
            total_radiance += radiance * step;
        }
    }
    total_radiance as f32
}

// Pre-tabulated CIE 1931 standard observer color matching functions (400-700nm at 10nm intervals)
static CIE_X_FIT: [f32; 31] = [
    0.014310, 0.043510, 0.134380, 0.283900, 0.348280, 0.336200, 0.290800, 0.195360,
    0.095640, 0.032010, 0.004900, 0.009300, 0.063270, 0.165500, 0.290400, 0.433450,
    0.594500, 0.762100, 0.916300, 1.026300, 1.062200, 1.002600, 0.854450, 0.642400,
    0.447900, 0.283500, 0.164900, 0.087400, 0.046770, 0.022700, 0.011359,
];

static CIE_Y_FIT: [f32; 31] = [
    0.000396, 0.001210, 0.004000, 0.011600, 0.023000, 0.038000, 0.060000, 0.090980,
    0.139020, 0.208020, 0.323000, 0.503000, 0.710000, 0.862000, 0.954000, 0.995000,
    0.995000, 0.952000, 0.870000, 0.757000, 0.631000, 0.503000, 0.381000, 0.265000,
    0.175000, 0.107000, 0.061000, 0.032000, 0.017000, 0.008210, 0.004102,
];

static CIE_Z_FIT: [f32; 31] = [
    0.067850, 0.207400, 0.645600, 1.385600, 1.747060, 1.772110, 1.669200, 1.287640,
    0.813000, 0.465180, 0.272000, 0.158200, 0.078250, 0.042160, 0.020300, 0.008700,
    0.003900, 0.002100, 0.001650, 0.001100, 0.000800, 0.000340, 0.000190, 0.000050,
    0.000020, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blackbody_solar_peak() {
        // Sun surface approx 5778 K
        let solar = SampledSpectrum::blackbody(5778.0);
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for i in 0..SPECTRAL_BANDS {
            if solar.values[i] > max_val {
                max_val = solar.values[i];
                max_idx = i;
            }
        }
        let peak_wavelength = LAMBDA_MIN_NM + max_idx as f32 * LAMBDA_STEP_NM;
        // Wien's displacement law predicts 2.89777e-3 / 5778 ~= 501 nm
        assert!((peak_wavelength - 500.0).abs() <= 20.0);
    }

    #[test]
    fn test_lwir_ambient_range() {
        // At 300 K (ambient Earth temperature), LWIR radiance (8-14um) should be approx 50-60 W / (m^2 * sr)
        let lwir = evaluate_lwir_radiance(300.0, 100);
        assert!(lwir > 40.0 && lwir < 70.0, "LWIR radiance at 300K was {lwir}");
    }
}
