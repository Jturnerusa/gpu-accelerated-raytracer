use std::io::Write;

pub mod gltf;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pos: [f32; 3],
    _p0: [u32; 1],
    normal: [f32; 3],
    _p1: [u32; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub metallic: f32,
    pub roughness: f32,
    pub emission: f32,
    pub ior: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Object {
    pub transform: [[f32; 4]; 4],
    pub mesh: u32,
    pub p0: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Mesh {
    pub primitive_start: u32,
    pub primitive_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Primitive {
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub index_start: u32,
    pub index_count: u32,
    pub material: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub projection: [[f32; 4]; 4],
    pub world: [[f32; 4]; 4],
}

#[derive(Clone, Debug)]
pub struct SceneDesc {
    pub world: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub objects: u32,
    pub meshes: u32,
    pub primitives: u32,
    pub vertices: u32,
    pub indices: u32,
    pub materials: u32,
    pub blas_entries: Vec<BlasEntry>,
}

#[derive(Clone, Debug)]
pub struct BlasEntry {
    pub transform: [[f32; 4]; 4],
    pub geometries: Vec<BlasGeometry>,
}

#[derive(Clone, Copy, Debug)]
pub struct BlasGeometry {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

pub trait Scene {
    fn load(
        &self,
        objects: &mut [u8],
        meshes: &mut [u8],
        primitives: &mut [u8],
        vertices: &mut [u8],
        indices: &mut [u8],
        materials: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>>;

    fn desc(&self) -> Result<SceneDesc, Box<dyn std::error::Error>>;
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3]) -> Self {
        Self {
            pos: position,
            _p0: [0],
            normal,
            _p1: [0],
        }
    }
}

impl Object {
    pub fn new(transform: [[f32; 4]; 4], mesh: u32) -> Self {
        Self {
            transform,
            mesh,
            p0: [0, 0, 0],
        }
    }
}

impl Material {
    pub fn new(metallic: f32, roughness: f32, emission: f32, ior: f32, color: [f32; 4]) -> Self {
        Self {
            metallic,
            roughness,
            emission,
            ior,
            color,
        }
    }
}
