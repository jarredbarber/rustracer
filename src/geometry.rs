use cgmath::prelude::*;
use cgmath::{Deg, Matrix4, Rad, Vector2, Vector3, Vector4};

pub type Vec4f = Vector4<f32>;
pub type Vec3f = Vector3<f32>;
pub type Vec2f = Vector2<f32>;
pub type Mat4f = Matrix4<f32>;

pub fn proj_orth(x: &Vec4f, z: &Vec4f) -> Vec4f {
    let a = x.dot(*z);
    let b = z.dot(*z);
    x - (a / b) * z
}

pub struct Ray {
    pub origin: Vec4f,
    pub direction: Vec4f,
}

impl Ray {
    pub fn new(origin: Vec4f, dir: Vec4f) -> Ray {
        let direction = dir.normalize();
        assert!(direction.w == 0.0);
        assert!(origin.w == 1.0);
        Ray { origin, direction }
    }

    pub fn to(&self, t: f32) -> Vec4f {
        self.origin + t * self.direction
    }

    pub fn from_target(origin: Vec4f, target: &Vec4f) -> Ray {
        Ray::new(origin, target - origin)
    }

    pub fn transform(&self, transform: &Mat4f) -> Ray {
        Ray {
            origin: transform * self.origin,
            direction: (transform * self.direction).normalize(),
        }
    }

    pub fn bump(&self) -> Ray {
        Ray {
            origin: self.origin + 1e-3 * self.direction,
            direction: self.direction,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Intersection {
    pub point: Vec4f,
    pub normal: Vec4f,
    pub uv: Vec2f,
    pub t: f32,
}

pub trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<Intersection>;
}

pub struct CSGInt<O1: Intersectable, O2: Intersectable>(O1, O2);
impl<O1: Intersectable, O2: Intersectable> Intersectable for CSGInt<O1, O2> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let CSGInt(o1, o2) = self;
        let i1 = o1.intersect(&ray);
        if i1.is_none() {
            return None;
        }
        let i2 = o2.intersect(&ray);
        if i2.is_none() {
            return None;
        }

        // Both interesct, return closest
        let t1 = i1.unwrap().t;
        let t2 = i2.unwrap().t;

        if t1 < t2 {
            i1
        } else {
            i2
        }
    }
}

pub struct CSGUnion<O1: Intersectable, O2: Intersectable>(pub O1, pub O2);
impl<O1: Intersectable, O2: Intersectable> Intersectable for CSGUnion<O1, O2> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let CSGUnion(o1, o2) = self;
        let i1 = o1.intersect(&ray);
        let i2 = o2.intersect(&ray);
        if i1.is_none() {
            return i2;
        }
        if i2.is_none() {
            return i1;
        }

        // Both interesct, return closest
        let t1 = i1.unwrap().t;
        let t2 = i2.unwrap().t;

        if t1 < t2 {
            i1
        } else {
            i2
        }
    }
}

pub struct FlipNormal<T: Intersectable>(pub T);
impl<T: Intersectable> Intersectable for FlipNormal<T> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let FlipNormal(obj) = self;
        match obj.intersect(&ray) {
            None => None,
            Some(int) => Some(Intersection {
                point: int.point,
                normal: -int.normal,
                uv: int.uv,
                t: int.t,
            }),
        }
    }
}

pub struct Transformed<T: Intersectable>(Mat4f, Mat4f, T);
impl<T: Intersectable> Transformed<T> {
    pub fn new(obj: T, transform: Mat4f) -> Self {
        Transformed(transform, transform.invert().unwrap(), obj)
    }
}

impl<T: Intersectable> Intersectable for Transformed<T> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let Transformed(tx, itx, obj) = self;
        let ray_prime = ray.transform(itx);

        match obj.intersect(&ray_prime) {
            None => None,
            Some(Intersection {
                point,
                normal,
                uv,
                t: _,
            }) => Some(Intersection {
                point: tx * point,
                normal: (tx * normal).normalize(),
                uv,
                t: (tx * point - ray.origin).magnitude(),
            }),
        }
    }
}
