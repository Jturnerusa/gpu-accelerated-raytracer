use std::collections::HashMap;

use nalgebra::{Matrix4, Vector3, Vector4};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
pub enum Light {
    Area {
        size: f32,
        power: f32,
        transform: Matrix4<f32>,
        color: Vector4<f32>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Material {
    pub metalic: f32,
    pub roughness: f32,
    pub color: Vector4<f32>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Camera {
    pub fov: f32,
    pub zfar: f32,
    pub znear: f32,
    pub aspect_ratio: f32,
    pub transform: Matrix4<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Mesh {
    pub verts: Vec<Vector3<f32>>,
    pub normals: Vec<Vector3<f32>>,
    pub faces: Vec<[usize; 3]>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Object {
    pub mesh: String,
    pub material: String,
    pub transform: Matrix4<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Scene {
    pub camera: Camera,
    pub lights: Vec<Light>,
    pub meshes: HashMap<String, Mesh>,
    pub materials: HashMap<String, Material>,
    pub objects: Vec<Object>,
}
