pub mod gltf;

use nalgebra::{Matrix4, Vector3, Vector4};

#[derive(Clone, Copy, Debug, encase::ShaderType)]
pub struct Vertex {
    pub position: Vector3<f32>,
    pub normal: Vector3<f32>,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
pub struct Material {
    pub metalic: f32,
    pub roughness: f32,
    pub emission: f32,
    pub color: Vector4<f32>,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
pub struct Object {
    pub transform: Matrix4<f32>,
    pub mesh: u32,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
pub struct Mesh {
    pub primitive_start: u32,
    pub primitive_count: u32,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
pub struct Primitive {
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub index_start: u32,
    pub index_count: u32,
    pub material: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub projection: Matrix4<f32>,
    pub world: Matrix4<f32>,
}

pub trait Scene {
    fn load(
        &self,
        objects: &mut Vec<Object>,
        meshes: &mut Vec<Mesh>,
        primitives: &mut Vec<Primitive>,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
        materials: &mut Vec<Material>,
    ) -> Result<Camera, Box<dyn std::error::Error>>;
}
