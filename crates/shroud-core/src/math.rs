//! High-precision geometric and spatial mathematical primitives.
//!
//! Includes zero-copy GPU memory layouts compatible with WGSL std140 / std430 packing.

use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 2D floating-point vector.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-8 {
            Self::new(self.x / len, self.y / len)
        } else {
            Self::ZERO
        }
    }
}

/// 3D floating-point vector with explicit 16-byte alignment for WGSL compatibility.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _pad: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, _pad: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0, _pad: 0.0 };
    pub const X: Self = Self { x: 1.0, y: 0.0, z: 0.0, _pad: 0.0 };
    pub const Y: Self = Self { x: 0.0, y: 1.0, z: 0.0, _pad: 0.0 };
    pub const Z: Self = Self { x: 0.0, y: 0.0, z: 1.0, _pad: 0.0 };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _pad: 0.0 }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-8 {
            Self::new(self.x / len, self.y / len, self.z / len)
        } else {
            Self::ZERO
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// 4D floating-point vector.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub const fn from_vec3(v: Vec3, w: f32) -> Self {
        Self { x: v.x, y: v.y, z: v.z, w }
    }

    #[inline]
    pub fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

/// 4x4 column-major matrix matching WGSL `mat4x4<f32>`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mat4 {
    pub cols: [Vec4; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        cols: [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        ],
    };

    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            cols: [
                Vec4::new(s.x, u.x, -f.x, 0.0),
                Vec4::new(s.y, u.y, -f.y, 0.0),
                Vec4::new(s.z, u.z, -f.z, 0.0),
                Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
            ],
        }
    }

    pub fn perspective(fov_y_rad: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let range_inv = 1.0 / (z_near - z_far);

        Self {
            cols: [
                Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, z_far * range_inv, -1.0),
                Vec4::new(0.0, 0.0, z_near * z_far * range_inv, 0.0),
            ],
        }
    }

    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.cols[0].x * v.x + self.cols[1].x * v.y + self.cols[2].x * v.z + self.cols[3].x * v.w,
            self.cols[0].y * v.x + self.cols[1].y * v.y + self.cols[2].y * v.z + self.cols[3].y * v.w,
            self.cols[0].z * v.x + self.cols[1].z * v.y + self.cols[2].z * v.z + self.cols[3].z * v.w,
            self.cols[0].w * v.x + self.cols[1].w * v.y + self.cols[2].w * v.z + self.cols[3].w * v.w,
        )
    }

    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        Self {
            cols: [
                self.mul_vec4(rhs.cols[0]),
                self.mul_vec4(rhs.cols[1]),
                self.mul_vec4(rhs.cols[2]),
                self.mul_vec4(rhs.cols[3]),
            ],
        }
    }

    pub fn inverse(&self) -> Option<Self> {
        let m = [
            self.cols[0].x, self.cols[0].y, self.cols[0].z, self.cols[0].w,
            self.cols[1].x, self.cols[1].y, self.cols[1].z, self.cols[1].w,
            self.cols[2].x, self.cols[2].y, self.cols[2].z, self.cols[2].w,
            self.cols[3].x, self.cols[3].y, self.cols[3].z, self.cols[3].w,
        ];

        let mut inv = [0.0f32; 16];

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

        let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if det.abs() < 1e-12 {
            return None;
        }

        let inv_det = 1.0 / det;

        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        Some(Self {
            cols: [
                Vec4::new(inv[0] * inv_det, inv[1] * inv_det, inv[2] * inv_det, inv[3] * inv_det),
                Vec4::new(inv[4] * inv_det, inv[5] * inv_det, inv[6] * inv_det, inv[7] * inv_det),
                Vec4::new(inv[8] * inv_det, inv[9] * inv_det, inv[10] * inv_det, inv[11] * inv_det),
                Vec4::new(inv[12] * inv_det, inv[13] * inv_det, inv[14] * inv_det, inv[15] * inv_det),
            ],
        })
    }
}

/// 3D Ray primitive for path tracing and sensor casting.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    pub t_min: f32,
    pub t_max: f32,
    pub _pad: [f32; 2],
}

impl Ray {
    pub fn new(origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> Self {
        Self {
            origin,
            dir: dir.normalize(),
            t_min,
            t_max,
            _pad: [0.0; 2],
        }
    }

    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// Axis-Aligned Bounding Box (AABB) for BVH acceleration structures.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        max: Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
    };

    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn union_point(&self, p: Vec3) -> Self {
        Self {
            min: self.min.min(p),
            max: self.max.max(p),
        }
    }

    pub fn union_aabb(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.extents();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Slab method for ray-box intersection with division-by-zero protection.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<(f32, f32)> {
        let inv_dx = if ray.dir.x.abs() > 1e-8 { 1.0 / ray.dir.x } else { 1e8 * ray.dir.x.signum() };
        let inv_dy = if ray.dir.y.abs() > 1e-8 { 1.0 / ray.dir.y } else { 1e8 * ray.dir.y.signum() };
        let inv_dz = if ray.dir.z.abs() > 1e-8 { 1.0 / ray.dir.z } else { 1e8 * ray.dir.z.signum() };

        let t1 = (self.min.x - ray.origin.x) * inv_dx;
        let t2 = (self.max.x - ray.origin.x) * inv_dx;
        let t3 = (self.min.y - ray.origin.y) * inv_dy;
        let t4 = (self.max.y - ray.origin.y) * inv_dy;
        let t5 = (self.min.z - ray.origin.z) * inv_dz;
        let t6 = (self.max.z - ray.origin.z) * inv_dz;

        let t_enter = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let t_exit = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if t_exit < 0.0 || t_enter > t_exit || t_enter > ray.t_max || t_exit < ray.t_min {
            None
        } else {
            Some((t_enter.max(ray.t_min), t_exit.min(ray.t_max)))
        }
    }
}

/// Orthonormal basis construction from a single unit normal vector (Duff et al. 2017).
#[inline]
pub fn build_orthonormal_basis(n: Vec3) -> (Vec3, Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let b1 = Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let b2 = Vec3::new(b, sign + n.y * n.y * a, -n.y);
    (b1, b2, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_operations() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
        let c = a.cross(b);
        assert_eq!(c.dot(a), 0.0);
        assert_eq!(c.dot(b), 0.0);
    }

    #[test]
    fn test_ray_aabb_intersection() {
        let box_ = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0), 0.0, 100.0);
        let hit = box_.intersect_ray(&ray);
        assert!(hit.is_some());
        let (t_in, t_out) = hit.unwrap();
        assert!((t_in - 4.0).abs() < 1e-4);
        assert!((t_out - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_orthonormal_basis() {
        let normals = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.577, 0.577, 0.577).normalize(),
        ];
        for n in normals {
            let (b1, b2, b3) = build_orthonormal_basis(n);
            assert!(b1.dot(b2).abs() < 1e-5);
            assert!(b1.dot(b3).abs() < 1e-5);
            assert!(b2.dot(b3).abs() < 1e-5);
            assert!((b1.length() - 1.0).abs() < 1e-5);
            assert!((b2.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_mat4_inverse() {
        let m = Mat4::look_at(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
        let inv = m.inverse().expect("Inverse failed");
        let ident = m.mul_mat4(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let val = match j {
                    0 => ident.cols[i].x,
                    1 => ident.cols[i].y,
                    2 => ident.cols[i].z,
                    _ => ident.cols[i].w,
                };
                assert!((val - expected).abs() < 1e-4, "Mismatch at col {i} row {j}");
            }
        }
    }
}
