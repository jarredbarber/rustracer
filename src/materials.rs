use cgmath::prelude::*;

use crate::geometry::*;

pub trait BRDF {
    fn brdf(&self, point: &Vec4f, light: &Vec4f, normal: &Vec4f, uv: &Vec2f, look: &Vec4f)
        -> Vec3f;

    fn reflection_vector(&self, normal: &Vec4f, look: &Vec4f) -> Vec4f {
        2.0 * look.dot(*normal) * normal - look
    }

    fn refractive_index(&self) -> f32 {
        1.0
    }

    fn energy_distribution(&self) -> Vec3f {
        Vec3f::new(1.0, 0.0, 0.0)
    }
}

#[derive(Clone, Copy)]
pub struct Phong {
    pub diffuse: Vec3f,
    pub specular: Vec3f,
    pub spec_exp: f32,
    pub transmission: f32,
    pub reflection: f32,
    pub index: f32,
}

impl BRDF for Phong {
    fn brdf(
        &self,
        _point: &Vec4f,
        light: &Vec4f,
        normal: &Vec4f,
        _uv: &Vec2f,
        view: &Vec4f,
    ) -> Vec3f {
        let refl = self.reflection_vector(normal, light);

        let ln = normal.dot(*light);
        let rv = refl.dot(*view);

        self.diffuse * ln.clamp(0.0, 1.0) + self.specular * rv.clamp(0.0, 1.0).powf(self.spec_exp)
    }

    fn refractive_index(&self) -> f32 {
        self.index
    }

    fn energy_distribution(&self) -> Vec3f {
        let diffuse = 1.0; // - self.transmission - self.reflection;
        Vec3f::new(diffuse, self.reflection, self.transmission)
    }
}

pub struct Checkerboard(pub f32, pub f32);
impl BRDF for Checkerboard {
    fn brdf(
        &self,
        _point: &Vec4f,
        _light: &Vec4f,
        _normal: &Vec4f,
        uv: &Vec2f,
        _view: &Vec4f,
    ) -> Vec3f {
        let Checkerboard(nx, ny) = self;

        let tx: i32 = ((uv[0] * 0.5 + 0.5) * nx) as i32;
        let ty: i32 = ((uv[1] * 0.5 + 0.5) * ny) as i32;

        let tile = ((tx + ty) % 2) as f32;
        tile * Vec3f::new(1.0, 1.0, 1.0)
    }
}
