pub mod gltf;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub metalic: f32,
    pub roughness: f32,
    pub emission: f32,
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Object {
    pub transform: [[f32; 4]; 4],
    pub mesh: u32,
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

#[derive(Clone, Copy, Debug)]
pub struct SceneDesc {
    pub world: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
    pub objects: u32,
    pub meshes: u32,
    pub primitives: u32,
    pub vertices: u32,
    pub indices: u32,
    pub materials: u32,
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

    fn configure_acceleration_structures(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tlas: wgpu::Tlas,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
    ) -> Result<wgpu::TlasPackage, Box<dyn std::error::Error>>;
}
