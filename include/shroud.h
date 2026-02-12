/*
 * Shroud :: High-Performance WebGPU Spectral Radiative Transfer & Sensor Engine
 * C ABI Header Interface
 */

#ifndef SHROUD_H
#define SHROUD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* Version inquiry */
int shroud_version_major(void);
int shroud_version_minor(void);
int shroud_version_patch(void);

/* Atmospheric & optical phase evaluations */
float shroud_eval_henyey_greenstein(float cos_theta, float g);
float shroud_eval_rayleigh_phase(float cos_theta);
float shroud_eval_lwir_radiance(float temp_k);
float shroud_calculate_lidar_photons(float range_m, float reflectance, float atmospheric_tau);

/* Spatial indexing */
uint32_t shroud_morton_3d(float x, float y, float z);

/* WGSL static validation through Naga */
int shroud_validate_wgsl(const char* name, const char* source, char* out_error, size_t max_error_len);

/* Rhai scenario script execution */
int shroud_run_rhai_script(const char* script, char* out_error, size_t max_error_len);

#ifdef __cplusplus
}
#endif

#endif /* SHROUD_H */
