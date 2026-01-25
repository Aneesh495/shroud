//! Fundamental physical constants and standard environmental reference values.
//!
//! All values are based on the 2019 SI redefinition and CODATA recommended values.

/// Speed of light in vacuum (m/s).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Planck constant (J * s).
pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;

/// Boltzmann constant (J / K).
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;

/// Stefan-Boltzmann constant (W / (m^2 * K^4)).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Standard acceleration due to gravity at sea level (m / s^2).
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// Universal molar gas constant (J / (mol * K)).
pub const MOLAR_GAS_CONSTANT: f64 = 8.314_462_618;

/// Avogadro constant (1 / mol).
pub const AVOGADRO_CONSTANT: f64 = 6.022_140_76e23;

/// Standard sea-level atmospheric pressure (Pa).
pub const STANDARD_SEA_LEVEL_PRESSURE: f64 = 101_325.0;

/// Standard sea-level atmospheric temperature (K).
pub const STANDARD_SEA_LEVEL_TEMPERATURE: f64 = 288.15;

/// Mean molar mass of dry air at Earth sea level (kg / mol).
pub const DRY_AIR_MOLAR_MASS: f64 = 0.028_964_4;

/// Mean volumetric radius of the Earth (m).
pub const EARTH_MEAN_RADIUS: f64 = 6_371_000.0;

/// First radiation constant c_1L = 2 * h * c^2 (W * m^2 / sr).
pub const FIRST_RADIATION_CONSTANT: f64 = 2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT * SPEED_OF_LIGHT;

/// Second radiation constant c_2 = h * c / k_B (m * K).
pub const SECOND_RADIATION_CONSTANT: f64 = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / BOLTZMANN_CONSTANT;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_radiation_constants() {
        let c1 = FIRST_RADIATION_CONSTANT;
        let c2 = SECOND_RADIATION_CONSTANT;
        assert!(c1 > 1.19e-16 && c1 < 1.20e-16);
        assert!(c2 > 1.43e-2 && c2 < 1.44e-2);
    }

    #[test]
    fn test_ideal_gas_ratio() {
        // Specific gas constant for dry air: R_spec = R / M_0
        let r_spec = MOLAR_GAS_CONSTANT / DRY_AIR_MOLAR_MASS;
        assert!((r_spec - 287.05).abs() < 0.1);
    }
}
