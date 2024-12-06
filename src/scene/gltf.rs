use std::borrow::Cow;

use nalgebra::{Matrix4, Perspective3, Vector3, Vector4};

use crate::scene::{Material, Object};

use super::{Camera, Mesh, Primitive, Vertex};

#[derive(Clone, Debug)]
pub struct Scene<'a> {
    gltf: gltf::Gltf,
    bin: Cow<'a, [u8]>,
}

impl<'a> Scene<'a> {
    pub fn from_slice(data: &'a [u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let glb = gltf::Glb::from_slice(data)?;
        let gltf = gltf::Gltf::from_slice(&glb.json)?;
        let bin = glb.bin.ok_or("failed to load binary data".to_string())?;

        Ok(Self { gltf, bin })
    }

    fn load_meshes(
        &self,
        meshes: &mut Vec<Mesh>,
        primitives: &mut Vec<Primitive>,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for mesh in self.gltf.document.meshes() {
            self.load_mesh(&mesh, meshes, primitives, vertices, indices)?;
        }

        Ok(())
    }

    fn load_mesh(
        &self,
        mesh: &gltf::Mesh,
        meshes: &mut Vec<Mesh>,
        primitives: &mut Vec<Primitive>,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        meshes.push(Mesh {
            primitive_start: primitives.len() as u32,
            primitive_count: mesh.primitives().len() as u32,
        });

        for primitive in mesh.primitives() {
            self.load_primitive(primitive, primitives, vertices, indices)?;
        }

        Ok(())
    }

    fn load_primitive(
        &self,
        primitive: gltf::Primitive,
        primitives: &mut Vec<Primitive>,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = primitive.reader(|_| Some(&self.bin));

        let vertex_start = primitives.iter().map(|i| i.vertex_count).sum();
        let vertex_count = reader.read_positions().unwrap().len() as u32;
        let index_start = primitives.iter().map(|i| i.index_count).sum();
        let index_count = reader.read_indices().unwrap().into_u32().len() as u32;
        let material = primitive.material().index().unwrap() as u32;

        primitives.push(Primitive {
            vertex_start,
            vertex_count,
            index_start,
            index_count,
            material,
        });

        for (position, normal) in reader
            .read_positions()
            .unwrap()
            .zip(reader.read_normals().unwrap())
        {
            vertices.push(Vertex {
                position: Vector3::new(position[0], position[1], position[2]),
                normal: Vector3::new(normal[0], normal[1], normal[2]),
            })
        }

        for index in reader.read_indices().unwrap().into_u32() {
            indices.push(index);
        }

        Ok(())
    }

    fn load_materials(
        &self,
        materials: &mut Vec<Material>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for material in self.gltf.document.materials() {
            let color = material.pbr_metallic_roughness().base_color_factor();

            materials.push(Material {
                metalic: material.pbr_metallic_roughness().metallic_factor(),
                roughness: material.pbr_metallic_roughness().roughness_factor(),
                emission: material.emissive_strength().unwrap_or(0.0),
                color: Vector4::new(color[0], color[1], color[2], color[3]),
            });
        }

        Ok(())
    }

    fn load_objects(&self, objects: &mut Vec<Object>) -> Result<(), Box<dyn std::error::Error>> {
        for node in self.gltf.document.nodes() {
            match node.mesh() {
                Some(mesh) => {
                    let transform = Matrix4::new(
                        node.transform().matrix()[0][0],
                        node.transform().matrix()[1][0],
                        node.transform().matrix()[2][0],
                        node.transform().matrix()[3][0],
                        node.transform().matrix()[0][1],
                        node.transform().matrix()[1][1],
                        node.transform().matrix()[2][1],
                        node.transform().matrix()[3][1],
                        node.transform().matrix()[0][2],
                        node.transform().matrix()[1][2],
                        node.transform().matrix()[2][2],
                        node.transform().matrix()[3][2],
                        node.transform().matrix()[0][3],
                        node.transform().matrix()[1][3],
                        node.transform().matrix()[2][3],
                        node.transform().matrix()[3][3],
                    );

                    objects.push(Object {
                        transform,
                        mesh: mesh.index() as u32,
                    });
                }
                None => continue,
            }
        }

        Ok(())
    }

    fn load_camera(&self) -> Result<Option<Camera>, Box<dyn std::error::Error>> {
        let node = match self
            .gltf
            .document
            .nodes()
            .find(|node| node.camera().is_some())
        {
            Some(node) => node,
            None => return Ok(None),
        };

        let world = Matrix4::new(
            node.transform().matrix()[0][0],
            node.transform().matrix()[1][0],
            node.transform().matrix()[2][0],
            node.transform().matrix()[3][0],
            node.transform().matrix()[0][1],
            node.transform().matrix()[1][1],
            node.transform().matrix()[2][1],
            node.transform().matrix()[3][1],
            node.transform().matrix()[0][2],
            node.transform().matrix()[1][2],
            node.transform().matrix()[2][2],
            node.transform().matrix()[3][2],
            node.transform().matrix()[0][3],
            node.transform().matrix()[1][3],
            node.transform().matrix()[2][3],
            node.transform().matrix()[3][3],
        );

        let projection = match node.camera().unwrap().projection() {
            gltf::camera::Projection::Orthographic(_) => {
                Err("does not support orthographic projections".to_string())?
            }
            gltf::camera::Projection::Perspective(perspective) => Perspective3::new(
                perspective
                    .aspect_ratio()
                    .ok_or("missing aspect_ratio field".to_string())?,
                perspective.yfov(),
                perspective.znear(),
                perspective.zfar().ok_or("missing zfar field".to_string())?,
            ),
        }
        .into_inner()
        .try_inverse()
        .ok_or("failed to invert projection matrix".to_string())?;

        Ok(Some(Camera { projection, world }))
    }
}

impl<'a> super::Scene for Scene<'a> {
    fn load(
        &self,
        objects: &mut Vec<Object>,
        meshes: &mut Vec<Mesh>,
        primitives: &mut Vec<Primitive>,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
        materials: &mut Vec<Material>,
    ) -> Result<Camera, Box<dyn std::error::Error>> {
        self.load_objects(objects)?;
        self.load_meshes(meshes, primitives, vertices, indices)?;
        self.load_materials(materials)?;

        let camera = self
            .load_camera()?
            .ok_or("no camera in scene".to_string())?;

        Ok(camera)
    }
}
