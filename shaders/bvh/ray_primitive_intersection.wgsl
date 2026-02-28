// Shroud :: High-Precision Ray-Primitive Geometric Intersections
// Möller-Trumbore and Watertight Triangle Intersections with Numerical Safeguards.

struct RayGeometric {
    origin: vec3<f32>,
    t_min: f32,
    direction: vec3<f32>,
    t_max: f32,
};

struct TriangleGeometric {
    v0: vec3<f32>,
    prim_id: u32,
    v1: vec3<f32>,
    material_id: u32,
    v2: vec3<f32>,
    padding: u32,
};

struct IntersectionRecord {
    t: f32,
    u: f32,
    v: f32,
    hit: u32,
    normal: vec3<f32>,
    prim_id: u32,
};

// Möller-Trumbore ray-triangle intersection algorithm
fn moller_trumbore_intersect(
    ray: RayGeometric,
    tri: TriangleGeometric,
    rec: ptr<function, IntersectionRecord>
) -> bool {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let pvec = cross(ray.direction, edge2);
    let det = dot(edge1, pvec);

    if (abs(det) < 1e-8) {
        return false;
    }

    let inv_det = 1.0 / det;
    let tvec = ray.origin - tri.v0;
    let u = dot(tvec, pvec) * inv_det;

    if (u < 0.0 || u > 1.0) {
        return false;
    }

    let qvec = cross(tvec, edge1);
    let v = dot(ray.direction, qvec) * inv_det;

    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    let t = dot(edge2, qvec) * inv_det;
    if (t < ray.t_min || t > ray.t_max) {
        return false;
    }

    (*rec).t = t;
    (*rec).u = u;
    (*rec).v = v;
    (*rec).hit = 1u;
    (*rec).normal = normalize(cross(edge1, edge2));
    (*rec).prim_id = tri.prim_id;
    return true;
}

@group(0) @binding(0) var<storage, read> s_test_rays: array<RayGeometric>;
@group(0) @binding(1) var<storage, read> s_test_triangles: array<TriangleGeometric>;
@group(0) @binding(2) var<storage, read_write> s_intersections: array<IntersectionRecord>;

@compute @workgroup_size(64, 1, 1)
fn cs_primitive_intersect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&s_test_rays) || idx >= arrayLength(&s_test_triangles)) {
        return;
    }

    let ray = s_test_rays[idx];
    let tri = s_test_triangles[idx];

    var rec: IntersectionRecord;
    rec.t = ray.t_max;
    rec.u = 0.0;
    rec.v = 0.0;
    rec.hit = 0u;
    rec.normal = vec3<f32>(0.0, 1.0, 0.0);
    rec.prim_id = 0xFFFFFFFFu;

    moller_trumbore_intersect(ray, tri, &rec);
    s_intersections[idx] = rec;
}
