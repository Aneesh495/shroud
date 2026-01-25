//! GPU memory layouts for Linear Bounding Volume Hierarchy (LBVH) and geometry.

use crate::math::{Aabb, Vec3};
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// GPU LBVH node layout (32 bytes, 16-byte aligned).
///
/// If `count > 0`, the node is a leaf pointing to primitives starting at `left_or_first`.
/// If `count == 0`, the node is an interior node with child indices `left_or_first` and `right_or_second`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuBvhNode {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub left_or_first: u32,
    pub right_or_second: u32,
    pub count: u32,
    pub _pad: u32,
}

impl GpuBvhNode {
    pub fn leaf(aabb: Aabb, first_prim: u32, prim_count: u32) -> Self {
        Self {
            aabb_min: aabb.min,
            aabb_max: aabb.max,
            left_or_first: first_prim,
            right_or_second: 0,
            count: prim_count,
            _pad: 0,
        }
    }

    pub fn interior(aabb: Aabb, left_child: u32, right_child: u32) -> Self {
        Self {
            aabb_min: aabb.min,
            aabb_max: aabb.max,
            left_or_first: left_child,
            right_or_second: right_child,
            count: 0,
            _pad: 0,
        }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

/// GPU Triangle representation (96 bytes).
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuTriangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub n0: Vec3,
    pub n1: Vec3,
    pub n2: Vec3,
    pub material_id: u32,
    pub _pad: [u32; 3],
}

impl GpuTriangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material_id: u32) -> Self {
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2).normalize();
        Self {
            v0,
            v1,
            v2,
            n0: normal,
            n1: normal,
            n2: normal,
            material_id,
            _pad: [0; 3],
        }
    }

    pub fn compute_aabb(&self) -> Aabb {
        let min = self.v0.min(self.v1).min(self.v2);
        let max = self.v0.max(self.v1).max(self.v2);
        Aabb::new(min, max)
    }

    pub fn centroid(&self) -> Vec3 {
        (self.v0 + self.v1 + self.v2) * (1.0 / 3.0)
    }
}

/// Computes 30-bit Morton code for a 3D coordinate normalized to [0, 1]^3.
#[inline]
pub fn morton_3d(norm_coord: Vec3) -> u32 {
    let x = (norm_coord.x.clamp(0.0, 1.0) * 1023.0) as u32;
    let y = (norm_coord.y.clamp(0.0, 1.0) * 1023.0) as u32;
    let z = (norm_coord.z.clamp(0.0, 1.0) * 1023.0) as u32;

    expand_bits_10(x) | (expand_bits_10(y) << 1) | (expand_bits_10(z) << 2)
}

#[inline]
fn expand_bits_10(mut v: u32) -> u32 {
    v &= 0x000003FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8))  & 0x0300F00F;
    v = (v | (v << 4))  & 0x030C30C3;
    v = (v | (v << 2))  & 0x09249249;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_encoding_order() {
        let p0 = Vec3::new(0.0, 0.0, 0.0);
        let p1 = Vec3::new(0.1, 0.1, 0.1);
        let p2 = Vec3::new(0.9, 0.9, 0.9);

        let m0 = morton_3d(p0);
        let m1 = morton_3d(p1);
        let m2 = morton_3d(p2);

        assert!(m0 < m1);
        assert!(m1 < m2);
    }

    #[test]
    fn test_triangle_aabb() {
        let tri = GpuTriangle::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
            0,
        );
        let aabb = tri.compute_aabb();
        assert_eq!(aabb.min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(aabb.max, Vec3::new(2.0, 3.0, 0.0));
    }
}
