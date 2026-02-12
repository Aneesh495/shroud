#include "../include/shroud.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    printf("Testing Shroud C ABI...\n");

    int maj = shroud_version_major();
    int min = shroud_version_minor();
    int pat = shroud_version_patch();
    printf("Shroud Version: %d.%d.%d\n", maj, min, pat);
    assert(maj == 0 && min == 1 && pat == 0);

    /* Test Henyey-Greenstein */
    float hg = shroud_eval_henyey_greenstein(1.0f, 0.5f);
    assert(hg > 0.0f);

    /* Test Rayleigh phase */
    float r = shroud_eval_rayleigh_phase(0.0f);
    assert(r > 0.0f);

    /* Test LWIR radiance */
    float lwir = shroud_eval_lwir_radiance(300.0f);
    assert(lwir > 40.0f && lwir < 70.0f);

    /* Test Morton 3D encoding */
    uint32_t m0 = shroud_morton_3d(0.0f, 0.0f, 0.0f);
    uint32_t m1 = shroud_morton_3d(0.5f, 0.5f, 0.5f);
    assert(m0 < m1);

    /* Test WGSL validation via C ABI */
    const char* valid_wgsl =
        "@compute @workgroup_size(64, 1, 1)\n"
        "fn cs_test(@builtin(global_invocation_id) gid: vec3<u32>) {}\n";
    char err_buf[512] = {0};
    int res = shroud_validate_wgsl("test_valid", valid_wgsl, err_buf, sizeof(err_buf));
    assert(res == 0);

    const char* bad_wgsl = "invalid syntax @#$%%";
    res = shroud_validate_wgsl("test_bad", bad_wgsl, err_buf, sizeof(err_buf));
    assert(res == 1);
    assert(err_buf[0] != '\0');

    /* Test Rhai evaluation via C ABI */
    const char* rhai_code = "let x = 10; let y = 20; x + y;";
    res = shroud_run_rhai_script(rhai_code, err_buf, sizeof(err_buf));
    assert(res == 0);

    printf("All C ABI checks PASSED successfully.\n");
    return 0;
}
