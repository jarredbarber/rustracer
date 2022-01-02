// #[macro_use]
// extern crate cgmath;
use cgmath::prelude::*;
use cgmath::{Deg, Matrix4, Rad, Vector2, Vector3, Vector4};
use rand::prelude::*;
use rayon::prelude::*;

type Vec4f = Vector4<f32>;
type Vec3f = Vector3<f32>;
type Vec2f = Vector2<f32>;
type Mat4f = Matrix4<f32>;

fn proj_orth(x: &Vec4f, z: &Vec4f) -> Vec4f {
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
}

#[derive(Clone, Copy)]
pub struct Intersection {
    pub point: Vec4f,
    pub normal: Vec4f,
    pub uv: Vec2f,
    pub t: f32,
}

trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<Intersection>;
}

trait BRDF {
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
struct Phong {
    diffuse: Vec3f,
    specular: Vec3f,
    spec_exp: f32,
    transmission: f32,
    reflection: f32,
    index: f32,
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

struct Checkerboard(f32, f32);
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
struct Plane;

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

struct CSGInt<O1: Intersectable, O2: Intersectable>(O1, O2);
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

struct CSGUnion<O1: Intersectable, O2: Intersectable>(O1, O2);
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

struct FlipNormal<T: Intersectable>(T);
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
struct Transformed<T: Intersectable>(Mat4f, Mat4f, T);
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

struct Sphere {}
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

trait Camera {
    fn sample_ray(&self, x: f32, y: f32) -> Ray;
}
struct PinholeCamera {
    origin: Vec4f,
    delta: Vec4f,
    xhat: Vec4f,
    yhat: Vec4f,
}

impl PinholeCamera {
    pub fn new(look: &Ray, up: &Vec4f, fov: Rad<f32>, aspect: f32) -> PinholeCamera {
        assert!(up.w == 0.0);
        let zhat = look.direction.normalize();
        let yhat_norm = proj_orth(&up, &zhat).normalize();
        let xhat_norm = (yhat_norm.truncate().cross(zhat.truncate())).extend(0.0);

        let tx = (fov * 0.5).tan();

        let xhat = tx * xhat_norm;
        let yhat = (tx / aspect) * yhat_norm;
        let delta = zhat; // - 0.5 * xhat - 0.5 * yhat;
        PinholeCamera {
            origin: look.origin,
            delta,
            xhat,
            yhat,
        }
    }
}

impl Camera for PinholeCamera {
    fn sample_ray(&self, x: f32, y: f32) -> Ray {
        Ray::new(self.origin, self.delta + x * self.xhat + y * self.yhat)
    }
}

trait Traceable: Intersectable + BRDF + Sync {}
impl<T: BRDF + Intersectable + Sync> Traceable for T {}
struct Object<O, M> {
    obj: O,
    mat: M,
}

impl<O, M> Object<O, M> {
    fn boxed(obj: O, mat: M) -> Box<Self> {
        Box::new(Self { obj, mat })
    }
}

impl<O: Intersectable, M> Intersectable for Object<O, M> {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        self.obj.intersect(&ray)
    }
}
impl<O, M: BRDF> BRDF for Object<O, M> {
    fn brdf(
        &self,
        point: &Vec4f,
        light: &Vec4f,
        normal: &Vec4f,
        uv: &Vec2f,
        view: &Vec4f,
    ) -> Vec3f {
        self.mat.brdf(point, light, normal, uv, view)
    }

    fn energy_distribution(&self) -> Vec3f {
        return self.mat.energy_distribution();
    }

    fn refractive_index(&self) -> f32 {
        return self.mat.refractive_index();
    }
}

struct Light {
    direction: Vec4f,
    color: Vec4f,
}

impl Light {
    fn new(direction: Vec4f, color: Vec4f) -> Self {
        Self {
            direction: direction.normalize(),
            color,
        }
    }
}

fn trace(scene: &Vec<Box<dyn Traceable>>, ray: &Ray, depth: u32) -> Option<Vec4f> {
    if depth > 10 {
        return None;
    }

    let light_color = 0.5 * Vec4f::new(1.0, 1.0, 1.0, 0.0);
    let lights = vec![
        Light::new(Vec4f::new(0.0, 5.0, 10.0, 0.0), light_color),
        Light::new(Vec4f::new(-1.0, 0.0, 0.2, 0.0), light_color),
    ];

    let mut result = None;
    for (id, obj) in scene.iter().enumerate() {
        match obj.intersect(&ray) {
            None => (),
            Some(int) => match result {
                None => result = Some((id, int)),
                Some((_, int2)) => {
                    if int.t < int2.t {
                        result = Some((id, int))
                    }
                }
            },
        }
    }

    match result {
        None => None,
        Some((obj, int)) => {
            let mut color = Vec4f::unit_w();
            let energy = scene[obj].energy_distribution();
            if energy[1] > 0.0 {
                // Generate reflected rays
                let refl_vec =
                    scene[obj].reflection_vector(&int.normal, &-ray.direction.normalize());
                let refl_ray = Ray::new(int.point + 1e-3 * refl_vec, refl_vec);
                match trace(&scene, &refl_ray, depth + 1) {
                    None => (),
                    Some(refl_color) => {
                        color += energy[1] * refl_color;
                    }
                }
            }
            if energy[2] > 0.0 {
                // Generate transmitted rays
                // point from intersection point to incoming ray
                let mut normal = int.normal.normalize();
                let light = ray.direction.normalize();
                let n = scene[obj].refractive_index();
                let r = if light.dot(normal) > 0.0 { 1.0 / n } else { n };
                if light.dot(normal) > 0.0 {
                    normal *= -1.0;
                }

                assert!(light.w == 0.0);
                assert!(normal.w == 0.0);
                let c = light.dot(normal);

                // let d = c * c + r * r - 1.0;
                let d = 1.0 - r * r * (1.0 - c * c);

                if d >= 0.0 {
                    // let tr_vec = r * (light + (d.sqrt() + c) * normal);
                    let tr_vec = r * (light - c * normal) - d.sqrt() * normal;

                    assert!(tr_vec.w == 0.0);

                    let tr_ray = Ray::new(int.point + 1e-3 * tr_vec, tr_vec);

                    match trace(&scene, &tr_ray, depth + 1) {
                        None => (),
                        Some(tr_color) => {
                            color += energy[2] * tr_color;
                        }
                    }
                }
            }
            if energy[0] > 0.0 && int.normal.dot(ray.direction) < 0.0 {
                // Factor in lighting
                for light in lights.iter() {
                    if light.direction.dot(int.normal) > 0.0 {
                        match trace(
                            &scene,
                            &Ray::new(int.point + 1e-3 * int.normal, light.direction),
                            depth + 1,
                        ) {
                            None => {
                                let look = -ray.direction;
                                let _color = scene[obj]
                                    .brdf(&int.point, &light.direction, &int.normal, &int.uv, &look)
                                    .extend(1.0);
                                color += energy[0] * light.color.mul_element_wise(_color);
                            }
                            _ => (),
                        }
                    }
                }
            }
            Some(color)
        }
    }
}

struct ImageBuffer {
    res_x: usize,
    res_y: usize,
    color: Vec<Vec4f>,
}

impl ImageBuffer {
    pub fn new(res_x: usize, res_y: usize) -> Self {
        ImageBuffer {
            res_x,
            res_y,
            color: vec![Vec4f::zero(); res_x * res_y * 4],
        }
    }

    pub fn accum(&mut self, x: usize, y: usize, data: &Vec4f) -> &mut Self {
        let ix = y * self.res_x + x;
        self.color[ix] += *data;
        self
    }

    pub fn write(&self, file: &str) -> () {
        use std::fs::File;
        use std::io::BufWriter;
        use std::path::Path;

        let path = Path::new(file);
        let file = File::create(path).unwrap();
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, self.res_x as u32, self.res_y as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_trns(vec![0xFFu8, 0xFFu8, 0xFFu8, 0xFFu8]);
        encoder.set_source_gamma(png::ScaledFloat::from_scaled(45455)); // 1.0 / 2.2, scaled by 100000
        encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2)); // 1.0 / 2.2, unscaled, but rounded
        let source_chromaticities = png::SourceChromaticities::new(
            // Using unscaled instantiation here
            (0.31270, 0.32900),
            (0.64000, 0.33000),
            (0.30000, 0.60000),
            (0.15000, 0.06000),
        );
        encoder.set_source_chromaticities(source_chromaticities);
        let mut writer = encoder.write_header().unwrap();

        let mut data = vec![0; (self.res_x * self.res_y * 4) as usize];

        for i in 0..(self.res_x * self.res_y) {
            for k in 0..4 {
                data[4 * i + k] = (self.color[i][k] * 255.0).clamp(0.0, 255.0) as u8;
            }
        }

        writer.write_image_data(&data).unwrap(); // Save
    }
}

fn render<C: Camera>(
    scene: &Vec<Box<dyn Traceable>>,
    camera: &C,
    output: &mut ImageBuffer,
    n_samples: usize,
) {
    let mut rng = rand::thread_rng();

    let mut rays = Vec::new();

    // bundle up all of the rays that we want to process
    for py in 0..output.res_y {
        for px in 0..output.res_x {
            for _ in 0..n_samples {
                let x: f32 = ((px as f32) + rng.gen::<f32>() - 0.5) / (output.res_x as f32) - 0.5;
                let y: f32 = ((py as f32) + rng.gen::<f32>() - 0.5) / (output.res_y as f32) - 0.5;
                let ray = camera.sample_ray(2.0 * x, -2.0 * y);

                rays.push((ray, px, py));
            }
        }
    }

    // trace each ray
    let pixels = rays
        .par_iter()
        .map(|(ray, px, py)| -> (usize, usize, Vec4f) {
            match trace(&scene, &ray, 0) {
                None => (*px, *py, Vec4f::zero()),
                Some(color) => (*px, *py, color),
            }
        })
        .collect::<Vec<(usize, usize, Vec4f)>>();

    // accumulate the output
    for &(px, py, color) in pixels.iter() {
        output.accum(px, py, &(color / (n_samples as f32)));
    }
}

fn main() {
    let res_x: usize = 1600;
    let res_y: usize = 1200;

    let aspect = (res_x as f32) / (res_y as f32);

    let camera = PinholeCamera::new(
        &Ray::from_target(
            Vec4f::new(-8.0, -4.5, 0.8, 1.0),
            &Vec4f::new(0.0, 0.0, 0.0, 1.0),
        ),
        &Vec4f::unit_z(),
        Deg(75.0).into(),
        aspect,
    );

    let mut scene: Vec<Box<dyn Traceable>> = vec![
        Object::boxed(
            Transformed::new(
                Sphere {},
                Mat4f::from_translation(Vec3f::new(0.0, 0.0, 0.5)),
            ),
            Phong {
                diffuse: 1.0 * Vec3f::new(0.5, 52.0 / 255.0, 235.0 / 255.0),
                specular: 1.0 * Vec3f::new(1.0, 1.0, 1.0),
                spec_exp: 12.0,
                transmission: 0.0,
                reflection: 0.8,
                index: 1.52,
            },
        ),
        Object::boxed(
            Transformed::new(
                Plane {},
                Mat4f::from_translation(Vec3f::new(0.0, 0.0, -1.0)) * Mat4f::from_scale(4.0),
            ),
            Checkerboard(8.0, 8.0),
        ),
    ];
    for kx in [-1.5, 1.5] {
        for ky in [-1.5, 1.5] {
            scene.push(Object::boxed(
                Transformed::new(
                    CSGUnion(
                        Sphere {},
                        FlipNormal(Transformed::new(Sphere {}, Mat4f::from_scale(0.95))),
                    ),
                    Matrix4::from_translation(Vec3f::new(kx, ky, -0.25)) * Matrix4::from_scale(0.5),
                ),
                Phong {
                    diffuse: 0.8 * Vec3f::new(0.5, (kx + 1.5) / 3.0, (ky + 1.5) / 3.0),
                    specular: 0.5 * Vec3f::new(1.0, 1.0, 1.0),
                    spec_exp: 8.0,
                    transmission: 1.0,
                    reflection: 0.0,
                    index: 1.1,
                },
            ));
        }
    }

    let n_samples: usize = 32;

    let mut output = ImageBuffer::new(res_x, res_y);

    render(&scene, &camera, &mut output, n_samples);

    output.write(r"image.png");
}
