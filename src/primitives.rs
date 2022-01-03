use crate::geometry::*;
use cgmath::prelude::*;

pub struct Sphere();

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let x0 = ray.origin - Vec4f::unit_w();

        // solve for ||origin + t*dir - position||^2 = r^2
        // or
        // ||x0 + t*dir||^2 = r^2
        // or
        // (x0^2 - r^2) + 2*t*(dir . x0) + t^2 dir^2 = c + b*t + a*t^2 = 0
        let a = ray.direction.dot(ray.direction); // should be 1
        let b = 2.0 * x0.dot(ray.direction);
        let c = x0.dot(x0) - 1.0;

        // quadratic discriminant
        let d = b * b - 4.0 * a * c;

        let t: f32 = if d > 0.0 {
            let u = -0.5 * b / a;
            let v = 0.5 * d.sqrt() / a;

            // possible roots
            let mut t0 = u - v;
            let mut t1 = u + v;

            // Smallest non-negative root
            if t1 < t0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            if t0 >= 0.0 {
                t0
            } else {
                t1
            }
        } else {
            -1.0
        };

        if t > 0.0 {
            let point = ray.to(t);

            let normal = (point - Vec4f::unit_w()).normalize();

            Some(Intersection {
                point,
                normal,
                uv: Vec2f::zero(),
                t,
            })
        } else {
            None
        }
    }
}

pub struct Plane();

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        // Solve for (ray.origin + t*ray.direction).zhat() = 0
        // t = -ray.origin.zhat() / ray.direction.zhat()

        let a = ray.direction.z;
        let b = -ray.origin.z;

        // Ray is orthogonal to plane
        if a == 0.0 {
            return None;
        }

        let t = b / a;

        if t <= 0.0 {
            return None;
        }

        // Compute intersect point
        let point = ray.to(t);
        // Check x and y
        if point.x.abs() > 1.0 || point.y.abs() > 1.0 {
            return None;
        }

        // Ok now we have an intersection
        Some(Intersection {
            point,
            normal: Vec4f::unit_z(),
            uv: Vec2f::new(point.x, point.y),
            t,
        })
    }
}
