use std::{borrow::Cow, io::Write, iter};

use gltf::Gltf;
use nalgebra::{Matrix4, Perspective3};
use sdl2::sys::AnyModifier;

use crate::scene::{Material, Object};

use super::{Camera, Mesh, Primitive, SceneDesc, Vertex};

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
        meshes: &mut [u8],
        primitives: &mut [u8],
        vertices: &mut [u8],
        indices: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut vertex_counter = 0;
        let mut index_counter = 0;

        for mesh in self.gltf.document.meshes() {
            self.load_mesh(
                &mesh,
                meshes,
                primitives,
                vertices,
                indices,
                &mut vertex_counter,
                &mut index_counter,
            )?;
        }

        Ok(())
    }

    fn load_mesh(
        &self,
        mesh: &gltf::Mesh,
        mut meshes: &mut [u8],
        primitives: &mut [u8],
        vertices: &mut [u8],
        indices: &mut [u8],
        vertex_counter: &mut u32,
        index_counter: &mut u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        meshes.write_all(bytemuck::bytes_of(&Mesh {
            primitive_start: primitives.len() as u32,
            primitive_count: mesh.primitives().len() as u32,
        }))?;

        for primitive in mesh.primitives() {
            self.load_primitive(
                primitive,
                primitives,
                vertices,
                indices,
                vertex_counter,
                index_counter,
            )?;
        }

        Ok(())
    }

    fn load_primitive(
        &self,
        primitive: gltf::Primitive,
        mut primitives: &mut [u8],
        mut vertices: &mut [u8],
        mut indices: &mut [u8],
        vertex_counter: &mut u32,
        index_counter: &mut u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let reader = primitive.reader(|_| Some(&self.bin));

        let vertex_start = *vertex_counter;
        let vertex_count = reader.read_positions().unwrap().len() as u32;
        *vertex_counter += vertex_count;

        let index_start = *index_counter;
        let index_count = reader.read_indices().unwrap().into_u32().len() as u32;
        *index_counter += index_count;

        let material = primitive.material().index().unwrap() as u32;

        primitives.write_all(bytemuck::bytes_of(&Primitive {
            vertex_start,
            vertex_count,
            index_start,
            index_count,
            material,
        }))?;

        for (position, normal) in reader
            .read_positions()
            .unwrap()
            .zip(reader.read_normals().unwrap())
        {
            vertices.write_all(bytemuck::bytes_of(&Vertex::new(position, normal)))?;
        }

        for index in reader.read_indices().unwrap().into_u32() {
            indices.write_all(bytemuck::bytes_of(&index))?;
        }

        Ok(())
    }

    fn load_materials(&self, mut materials: &mut [u8]) -> Result<(), Box<dyn std::error::Error>> {
        for material in self.gltf.document.materials() {
            let color = material.pbr_metallic_roughness().base_color_factor();

            materials.write_all(bytemuck::bytes_of(&Material {
                metalic: material.pbr_metallic_roughness().metallic_factor(),
                roughness: material.pbr_metallic_roughness().roughness_factor(),
                emission: material.emissive_strength().unwrap_or(0.0),
                color,
            }))?;
        }

        Ok(())
    }

    fn load_objects(&self, mut objects: &mut [u8]) -> Result<(), Box<dyn std::error::Error>> {
        for node in self.gltf.document.nodes() {
            match node.mesh() {
                Some(mesh) => {
                    let transform = bytemuck::cast_slice(
                        Matrix4::new(
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
                        )
                        .as_slice(),
                    )
                    .try_into()
                    .unwrap();

                    objects.write_all(bytemuck::bytes_of(&Object {
                        transform,
                        mesh: mesh.index() as u32,
                    }))?;
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

        let world = bytemuck::cast_slice(
            Matrix4::new(
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
            )
            .as_slice(),
        )
        .try_into()
        .unwrap();

        let projection = match node.camera().unwrap().projection() {
            gltf::camera::Projection::Orthographic(_) => {
                Err("does not support orthographic projections".to_string())?
            }
            gltf::camera::Projection::Perspective(perspective) => bytemuck::cast_slice(
                Perspective3::new(
                    perspective
                        .aspect_ratio()
                        .ok_or("missing aspect_ratio field".to_string())?,
                    perspective.yfov(),
                    perspective.znear(),
                    perspective.zfar().ok_or("missing zfar field".to_string())?,
                )
                .into_inner()
                .try_inverse()
                .unwrap()
                .as_slice(),
            )
            .try_into()
            .unwrap(),
        };

        Ok(Some(Camera { projection, world }))
    }

    fn object_count(&self) -> u32 {
        self.gltf
            .document
            .nodes()
            .filter(|object| object.mesh().is_some())
            .count() as u32
    }

    fn mesh_count(&self) -> u32 {
        self.gltf.document.meshes().count() as u32
    }

    fn primitive_count(&self) -> u32 {
        self.gltf
            .document
            .meshes()
            .map(|mesh| mesh.primitives().len())
            .sum::<usize>() as u32
    }

    fn vertex_count(&self) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(self
            .gltf
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .try_fold(0, |acc, primitive| {
                Result::<usize, Box<dyn std::error::Error>>::Ok(
                    acc + primitive
                        .reader(|_| Some(&self.bin))
                        .read_positions()
                        .ok_or("failed to read positions".to_string())?
                        .len(),
                )
            })? as u32)
    }

    fn index_count(&self) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(self
            .gltf
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .try_fold(0, |acc, primitive| {
                Result::<usize, Box<dyn std::error::Error>>::Ok(
                    acc + primitive
                        .reader(|_| Some(&self.bin))
                        .read_indices()
                        .ok_or("failed to read positions".to_string())?
                        .into_u32()
                        .len(),
                )
            })? as u32)
    }

    fn material_count(&self) -> u32 {
        self.gltf.document.materials().len() as u32
    }

    fn get_primitive_vertex_count(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        self.gltf
            .document
            .meshes()
            .nth(mesh_index)
            .unwrap()
            .primitives()
            .nth(primitive_index)
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(&self.bin));

                Ok(reader
                    .read_positions()
                    .ok_or("failed to read positions".to_string())?
                    .len() as u32)
            })
            .unwrap()
    }

    fn get_primitive_first_vertex(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(self
            .gltf
            .document
            .meshes()
            .take(mesh_index)
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(&self.bin));

                Ok(reader
                    .read_positions()
                    .ok_or("failed to read positions")?
                    .len() as u32)
            })
            .try_fold(0, |acc, e: Result<_, Box<dyn std::error::Error>>| {
                Result::<_, Box<dyn std::error::Error>>::Ok(acc + e?)
            })?
            + self
                .gltf
                .document
                .meshes()
                .nth(mesh_index)
                .unwrap()
                .primitives()
                .take(primitive_index)
                .map(|primitive| {
                    let reader = primitive.reader(|_| Some(&self.bin));

                    Ok(reader
                        .read_positions()
                        .ok_or("failed to read positions")?
                        .len() as u32)
                })
                .try_fold(0, |acc, e: Result<_, Box<dyn std::error::Error>>| {
                    Result::<_, Box<dyn std::error::Error>>::Ok(acc + e?)
                })?)
    }

    fn get_primitive_index_count(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        self.gltf
            .document
            .meshes()
            .nth(mesh_index)
            .unwrap()
            .primitives()
            .nth(primitive_index)
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(&self.bin));

                Ok(reader
                    .read_indices()
                    .ok_or("failed to read indices".to_string())?
                    .into_u32()
                    .len() as u32)
            })
            .unwrap()
    }

    fn get_primitive_first_index(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Box<dyn std::error::Error>> {
        Ok(self
            .gltf
            .document
            .meshes()
            .take(mesh_index)
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(&self.bin));

                Ok(reader
                    .read_indices()
                    .ok_or("failed to read indices")?
                    .into_u32()
                    .len() as u32)
            })
            .try_fold(0, |acc, e: Result<_, Box<dyn std::error::Error>>| {
                Result::<_, Box<dyn std::error::Error>>::Ok(acc + e?)
            })?
            + self
                .gltf
                .document
                .meshes()
                .nth(mesh_index)
                .unwrap()
                .primitives()
                .take(primitive_index)
                .map(|primitive| {
                    let reader = primitive.reader(|_| Some(&self.bin));

                    Ok(reader
                        .read_indices()
                        .ok_or("failed to read indices")?
                        .into_u32()
                        .len() as u32)
                })
                .try_fold(0, |acc, e: Result<_, Box<dyn std::error::Error>>| {
                    Result::<_, Box<dyn std::error::Error>>::Ok(acc + e?)
                })?)
    }
}

impl<'data> super::Scene for Scene<'data> {
    fn load(
        &self,
        objects: &mut [u8],
        meshes: &mut [u8],
        primitives: &mut [u8],
        vertices: &mut [u8],
        indices: &mut [u8],
        materials: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.load_meshes(meshes, primitives, vertices, indices)?;
        self.load_objects(objects)?;
        self.load_materials(materials)?;

        Ok(())
    }

    fn desc(&self) -> Result<super::SceneDesc, Box<dyn std::error::Error>> {
        let camera = match self.load_camera() {
            Ok(Some(camera)) => camera,
            Ok(None) => Err("failed to load camera".to_string())?,
            Err(e) => Err(e)?,
        };

        Ok(SceneDesc {
            world: camera.world,
            projection: camera.projection,
            objects: self.object_count(),
            meshes: self.mesh_count(),
            primitives: self.primitive_count(),
            vertices: self.vertex_count()?,
            indices: self.index_count()?,
            materials: self.material_count(),
        })
    }

    fn configure_acceleration_structures(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tlas: wgpu::Tlas,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
    ) -> Result<wgpu::TlasPackage, Box<dyn std::error::Error>> {
        let mut tlas_package = wgpu::TlasPackage::new(tlas);

        let blases = self
            .gltf
            .document
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node.transform().matrix(), mesh)))
            .enumerate()
            .map(|(i, (transform, mesh))| {
                let sizes = mesh
                    .primitives()
                    .map(|primitive| {
                        let reader = primitive.reader(|_| Some(&self.bin));
                        let vertex_count = reader
                            .read_positions()
                            .ok_or("failed to read positions".to_string())?
                            .len() as u32;
                        let index_count = reader
                            .read_indices()
                            .ok_or("failed to read indices".to_string())?
                            .into_u32()
                            .len() as u32;

                        Result::<_, Box<dyn std::error::Error>>::Ok((
                            wgpu::BlasTriangleGeometrySizeDescriptor {
                                vertex_format: wgpu::VertexFormat::Float32x3,
                                vertex_count,
                                index_format: Some(wgpu::IndexFormat::Uint32),
                                index_count: Some(index_count),
                                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                            },
                            mesh.index(),
                            primitive.index(),
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let blas = device.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: None,
                        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: sizes
                            .iter()
                            .map(|(size, _, _)| size.clone())
                            .collect::<Vec<_>>(),
                    },
                );

                let transposed =
                    &Matrix4::from_row_slice(bytemuck::cast_slice(transform.as_slice()))
                        .transpose();

                tlas_package[i] = Some(wgpu::TlasInstance::new(
                    &blas,
                    transposed.as_slice()[0..12].try_into().unwrap(),
                    i as u32,
                    0xff,
                ));

                Result::<_, Box<dyn std::error::Error>>::Ok((sizes, blas))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let build_entries = blases
            .iter()
            .map(|(sizes, blas)| {
                let geometries = sizes
                    .iter()
                    .map(|(size, mesh_index, primitive_index)| {
                        let first_vertex =
                            self.get_primitive_first_vertex(*mesh_index, *primitive_index)?;
                        let first_index =
                            self.get_primitive_first_index(*mesh_index, *primitive_index)?;

                        Result::<_, Box<dyn std::error::Error>>::Ok(wgpu::BlasTriangleGeometry {
                            size,
                            vertex_buffer,
                            first_vertex,
                            vertex_stride: std::mem::size_of::<Vertex>() as u64,
                            index_buffer: Some(index_buffer),
                            index_buffer_offset: Some((first_index * 4) as u64),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Result::<_, Box<dyn std::error::Error>>::Ok(wgpu::BlasBuildEntry {
                    blas,
                    geometry: wgpu::BlasGeometries::TriangleGeometries(geometries),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(build_entries.iter(), iter::once(&tlas_package));

        queue.submit(Some(encoder.finish()));

        Ok(tlas_package)
    }
}
