use crate::geometry::*;
use cgmath::prelude::*;
use rand::prelude::*;

pub trait Light: Sync {
    fn sample(&self, point: &Vec4f, normal: &Vec4f) -> (Ray, Vec3f);
}

pub struct SunLight {
    direction: Vec4f,
    color: Vec3f,
}

impl SunLight {
    pub fn new(direction: Vec4f, color: Vec3f) -> Self {
        Self {
            direction: direction.normalize(),
            color,
        }
    }
}

impl Light for SunLight {
    fn sample(&self, point: &Vec4f, _normal: &Vec4f) -> (Ray, Vec3f) {
        (Ray::new(point.clone(), self.direction.clone()), self.color)
    }
}

pub struct PointLight {
    pub position: Vec4f,
    pub color: Vec3f,
}

impl Light for PointLight {
    fn sample(&self, point: &Vec4f, _normal: &Vec4f) -> (Ray, Vec3f) {
        let r = self.position - point;
        (
            Ray::new(point.clone(), r),
            self.color / (1e-4 + r.magnitude()),
        )
    }
}

pub struct AreaLight {
    // Transforms the rectangle [-0.5, 0.5] x [-0.5, 0.5] x {0}
    // to the area light
    pub transform: Mat4f,
    pub color: Vec3f,
}

impl Light for AreaLight {
    fn sample(&self, point: &Vec4f, _normal: &Vec4f) -> (Ray, Vec3f) {
        let mut rng = rand::thread_rng();
        let x = rng.gen::<f32>() - 0.5;
        let y = rng.gen::<f32>() - 0.5;
        let position = self.transform * Vec4f::new(x, y, 0.0, 1.0);
        let normal = (self.transform * Vec4f::unit_z()).normalize();
        let r = position - point;
        let mag_r = r.magnitude();

        let dot = normal.dot(-r / mag_r).clamp(0.0, 1.0);

        (
            Ray::new(point.clone(), r),
            dot * self.color / (1e-4 + mag_r),
        )
    }
}
