//! C ABI bindings for the Shroud engine.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint};

#[no_mangle]
pub extern "C" fn shroud_version_major() -> c_int {
    0
}

#[no_mangle]
pub extern "C" fn shroud_version_minor() -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn shroud_version_patch() -> c_int {
    0
}

#[no_mangle]
pub extern "C" fn shroud_eval_henyey_greenstein(cos_theta: c_float, g: c_float) -> c_float {
    shroud_core::atmospheric::henyey_greenstein(cos_theta, g)
}

#[no_mangle]
pub extern "C" fn shroud_eval_rayleigh_phase(cos_theta: c_float) -> c_float {
    shroud_core::atmospheric::rayleigh_phase(cos_theta)
}

#[no_mangle]
pub extern "C" fn shroud_eval_lwir_radiance(temp_k: c_float) -> c_float {
    shroud_core::spectral::evaluate_lwir_radiance(temp_k, 64)
}

#[no_mangle]
pub extern "C" fn shroud_calculate_lidar_photons(
    range_m: c_float,
    reflectance: c_float,
    atmospheric_tau: c_float,
) -> c_float {
    let cfg = shroud_core::sensors::LidarTransceiverConfig::default_automotive_905nm();
    cfg.expected_return_photons(range_m, reflectance, 1.0, atmospheric_tau)
}

#[no_mangle]
pub extern "C" fn shroud_morton_3d(x: c_float, y: c_float, z: c_float) -> c_uint {
    shroud_core::bvh_types::morton_3d(shroud_core::math::Vec3::new(x, y, z))
}

#[no_mangle]
pub extern "C" fn shroud_validate_wgsl(
    name: *const c_char,
    source: *const c_char,
    out_error: *mut c_char,
    max_error_len: usize,
) -> c_int {
    if source.is_null() {
        return -1;
    }

    let c_name = if name.is_null() {
        "unnamed"
    } else {
        unsafe { CStr::from_ptr(name).to_str().unwrap_or("invalid_name") }
    };

    let c_src = unsafe {
        match CStr::from_ptr(source).to_str() {
            Ok(s) => s,
            Err(_) => return -2,
        }
    };

    let res = shroud_validator::validator::validate_wgsl_source(c_name, c_src);
    if res.is_valid {
        0
    } else {
        if !out_error.is_null() && max_error_len > 0 {
            let err_msg = res.errors.join("; ");
            let c_str = CString::new(err_msg).unwrap_or_default();
            let bytes = c_str.as_bytes_with_nul();
            let copy_len = bytes.len().min(max_error_len);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out_error, copy_len);
                *out_error.add(copy_len - 1) = 0;
            }
        }
        1
    }
}

#[no_mangle]
pub extern "C" fn shroud_run_rhai_script(
    script: *const c_char,
    out_error: *mut c_char,
    max_error_len: usize,
) -> c_int {
    if script.is_null() {
        return -1;
    }

    let c_script = unsafe {
        match CStr::from_ptr(script).to_str() {
            Ok(s) => s,
            Err(_) => return -2,
        }
    };

    let engine = shroud_rhai::engine::ShroudRhaiEngine::new();
    match engine.eval_script::<rhai::Dynamic>(c_script) {
        Ok(_) => 0,
        Err(e) => {
            if !out_error.is_null() && max_error_len > 0 {
                let err_msg = format!("{e}");
                let c_str = CString::new(err_msg).unwrap_or_default();
                let bytes = c_str.as_bytes_with_nul();
                let copy_len = bytes.len().min(max_error_len);
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out_error, copy_len);
                    *out_error.add(copy_len - 1) = 0;
                }
            }
            1
        }
    }
}
