// #[macro_use]
use cgmath::prelude::*;
use cgmath::{Deg, Point3, Rad};
use rand::prelude::*;
use rayon::prelude::*;

mod geometry;
mod lights;
mod materials;
mod primitives;

use crate::geometry::*;
use crate::lights::*;
use crate::materials::*;
use crate::primitives::*;

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

pub struct Scene {
    objects: Vec<Box<dyn Traceable>>,
    lights: Vec<Box<dyn Light>>,
}
fn trace(scene: &Scene, ray: &Ray, depth: u32) -> Option<Vec4f> {
    if depth > 10 {
        return None;
    }

    let mut result = None;
    for (id, obj) in scene.objects.iter().enumerate() {
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
            let energy = scene.objects[obj].energy_distribution();
            if energy[1] > 0.0 {
                // Generate reflected rays
                let refl_vec =
                    scene.objects[obj].reflection_vector(&int.normal, &-ray.direction.normalize());
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
                let n = scene.objects[obj].refractive_index();
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
                for light in scene.lights.iter() {
                    let (light_ray, light_color) = light.sample(&int.point, &int.normal);

                    if light_ray.direction.dot(int.normal) > 0.0 {
                        match trace(&scene, &light_ray.bump(), depth + 1) {
                            None => {
                                let look = -ray.direction;
                                let _color = scene.objects[obj]
                                    .brdf(
                                        &int.point,
                                        &light_ray.direction,
                                        &int.normal,
                                        &int.uv,
                                        &look,
                                    )
                                    .extend(1.0);
                                color +=
                                    energy[0] * light_color.extend(1.0).mul_element_wise(_color);
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

fn render<C: Camera>(scene: &Scene, camera: &C, output: &mut ImageBuffer, n_samples: usize) {
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

    let light_color = 0.5 * Vec3f::new(1.0, 1.0, 1.0);
    let lights: Vec<Box<dyn Light>> = vec![
        // Box::new(SunLight::new(Vec4f::new(0.0, 5.0, 10.0, 0.0), light_color)),
        // Box::new(SunLight::new(Vec4f::new(-1.0, 0.0, 0.2, 0.0), light_color)),
        Box::new(AreaLight {
            transform: Mat4f::from_translation(Vec3f::new(0.0, -5.0, 0.0))
                * Mat4f::from_angle_x::<Deg<f32>>(Deg(-90.0)),
            color: 10.0 * light_color,
        }),
        Box::new(AreaLight {
            transform: Mat4f::from_translation(Vec3f::new(0.0, 3.0, 2.5))
                * Mat4f::from_angle_y::<Deg<f32>>(Deg(180.0f32)),
            color: 2.0 * light_color,
        }),
    ];

    let mut objects: Vec<Box<dyn Traceable>> = vec![
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
            objects.push(Object::boxed(
                Transformed::new(
                    CSGUnion(
                        Sphere {},
                        FlipNormal(Transformed::new(Sphere {}, Mat4f::from_scale(0.95))),
                    ),
                    Mat4f::from_translation(Vec3f::new(kx, ky, -0.25)) * Mat4f::from_scale(0.5),
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

    let scene = Scene { objects, lights };

    let n_samples: usize = 256;

    let mut output = ImageBuffer::new(res_x, res_y);

    render(&scene, &camera, &mut output, n_samples);

    output.write(r"image.png");
}
