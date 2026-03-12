// Shroud :: Thin-Lens Camera with Depth of Field & Circle of Confusion (CoC)
// Samples circular lens pupil apertures to simulate optical defocus and bokeh.

struct ThinLensUniforms {
    focal_length_m: f32,
    f_number: f32,
    focus_distance_m: f32,
    sensor_width_m: f32,
};

struct CameraRayOutput {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    coc_diameter_m: f32,
};

fn sample_concentric_disk(u: vec2<f32>) -> vec2<f32> {
    let u_offset = 2.0 * u - vec2<f32>(1.0, 1.0);
    if (u_offset.x == 0.0 && u_offset.y == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }

    var theta: f32;
    var r: f32;
    if (abs(u_offset.x) > abs(u_offset.y)) {
        r = u_offset.x;
        theta = 0.78539816339 * (u_offset.y / u_offset.x);
    } else {
        r = u_offset.y;
        theta = 1.57079632679 - 0.78539816339 * (u_offset.x / u_offset.y);
    }
    return r * vec2<f32>(cos(theta), sin(theta));
}

@group(0) @binding(0) var<uniform> u_lens: ThinLensUniforms;
@group(0) @binding(1) var<storage, read> s_film_coords: array<vec4<f32>>; // film_x, film_y, u_lens.x, u_lens.y
@group(0) @binding(2) var<storage, read_write> s_camera_rays: array<CameraRayOutput>;

@compute @workgroup_size(64, 1, 1)
fn cs_generate_thin_lens_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_camera_rays)) {
        return;
    }

    let input_coord = s_film_coords[idx];
    let film_uv = input_coord.xy;
    let lens_uv = input_coord.zw;

    let aperture_radius = u_lens.focal_length_m / (2.0 * u_lens.f_number);
    let lens_point = sample_concentric_disk(lens_uv) * aperture_radius;

    // Focus point on the plane of focus
    let focus_plane_z = u_lens.focus_distance_m;
    let focus_point = vec3<f32>(film_uv.x * focus_plane_z, film_uv.y * focus_plane_z, focus_plane_z);

    let ray_orig = vec3<f32>(lens_point.x, lens_point.y, 0.0);
    let ray_dir = normalize(focus_point - ray_orig);

    var out_r: CameraRayOutput;
    out_r.origin = ray_orig;
    out_r.t_min = 0.01;
    out_r.direction = ray_dir;
    out_r.coc_diameter_m = aperture_radius * 2.0;

    s_camera_rays[idx] = out_r;
}
