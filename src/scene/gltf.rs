use std::{
    fs::File,
    io::{self, Write},
    iter,
};

use nalgebra::{Matrix4, Perspective3};

use crate::scene::{BlasGeometry, Material, Object};

use super::{BlasEntry, Camera, Mesh, Primitive, SceneDesc, TextureDesc, Vertex};

#[derive(Debug)]
pub struct Error {
    message: String,
    source: Option<Box<dyn std::error::Error>>,
}

#[derive(Debug)]
enum BufferReader<'a> {
    Bin(&'a [u8]),
    File(memmap2::Mmap),
}

#[derive(Clone, Debug)]
pub struct Scene<'a> {
    document: gltf::Document,
    bin: &'a [u8],
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "error loading glb or gltf scene: {}", self.message)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.source.as_ref() {
            Some(e) => Some(&**e),
            None => None,
        }
    }
}

impl<'a> Scene<'a> {
    pub fn new(json: &'a [u8], bin: &'a [u8]) -> Result<Self, Error> {
        let document = gltf::Gltf::from_slice(json)
            .map_err(|e| Error {
                message: "failed to validate scene".to_string(),
                source: Some(Box::new(e)),
            })?
            .document;

        Ok(Self { document, bin })
    }

    fn read_buffer(&self, buffer: &gltf::Buffer) -> Result<BufferReader, Error> {
        Ok(match buffer.source() {
            gltf::buffer::Source::Bin => BufferReader::Bin(self.bin),
            gltf::buffer::Source::Uri(uri) => {
                let file = File::open(uri).map_err(|e| Error {
                    message: format!("failed to open {uri}"),
                    source: Some(Box::new(e)),
                })?;

                let mmap = unsafe {
                    memmap2::Mmap::map(&file).map_err(|e| Error {
                        message: format!("failed to mmap {uri}"),
                        source: Some(Box::new(e)),
                    })?
                };

                BufferReader::File(mmap)
            }
        })
    }

    fn load_meshes(
        &self,
        mut meshes: &mut [u8],
        mut primitives: &mut [u8],
        mut vertices: &mut [u8],
        mut indices: &mut [u8],
    ) -> Result<(), Error> {
        let mut vertex_counter = 0;
        let mut index_counter = 0;

        for mesh in self.document.meshes() {
            self.load_mesh(
                &mesh,
                &mut meshes,
                &mut primitives,
                &mut vertices,
                &mut indices,
                &mut vertex_counter,
                &mut index_counter,
            )?;
        }

        Ok(())
    }

    fn load_mesh(
        &self,
        mesh: &gltf::Mesh,
        meshes: &mut &mut [u8],
        primitives: &mut &mut [u8],
        vertices: &mut &mut [u8],
        indices: &mut &mut [u8],
        vertex_counter: &mut u32,
        index_counter: &mut u32,
    ) -> Result<(), Error> {
        let primitive_start = self
            .document
            .meshes()
            .take(mesh.index())
            .flat_map(|mesh| mesh.primitives())
            .count() as u32;

        let primitive_count = mesh.primitives().count() as u32;

        meshes
            .write_all(bytemuck::bytes_of(&Mesh {
                primitive_start,
                primitive_count,
            }))
            .map_err(|e| Error {
                message: "failed to write mesh to staging buffer".to_string(),
                source: Some(Box::new(e)),
            })?;

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
        primitives: &mut &mut [u8],
        vertices: &mut &mut [u8],
        indices: &mut &mut [u8],
        vertex_counter: &mut u32,
        index_counter: &mut u32,
    ) -> Result<(), Error> {
        let reader = primitive.reader(|_| Some(self.bin));

        let vertex_start = *vertex_counter;
        let vertex_count = reader
            .read_positions()
            .ok_or(Error {
                message: "failed to read positions".to_string(),
                source: None,
            })?
            .len() as u32;
        *vertex_counter += vertex_count;

        let index_start = *index_counter;
        let index_count = reader
            .read_indices()
            .ok_or(Error {
                message: "failed to read indices".to_string(),
                source: None,
            })?
            .into_u32()
            .len() as u32;
        *index_counter += index_count;

        let material = primitive.material().index().ok_or(Error {
            message: "no material found for primitive".to_string(),
            source: None,
        })? as u32;

        primitives
            .write_all(bytemuck::bytes_of(&Primitive {
                vertex_start,
                vertex_count,
                index_start,
                index_count,
                material,
            }))
            .map_err(|e| Error {
                message: "failed to write primitive to staging buffer".to_string(),
                source: Some(Box::new(e)),
            })?;

        for ((position, normal), uv) in reader
            .read_positions()
            .ok_or(Error {
                message: "failed to read positions".to_string(),
                source: None,
            })?
            .zip(reader.read_normals().ok_or(Error {
                message: "failed to read normals".to_string(),
                source: None,
            })?)
            .zip(match reader.read_tex_coords(0) {
                Some(texture_coords) => {
                    Box::new(texture_coords.into_f32()) as Box<dyn Iterator<Item = [f32; 2]>>
                }
                None => {
                    Box::new(iter::repeat(Default::default())) as Box<dyn Iterator<Item = [f32; 2]>>
                }
            })
        {
            vertices
                .write_all(bytemuck::bytes_of(&Vertex::new(position, normal, uv)))
                .map_err(|e| Error {
                    message: "failed to write vertices to staging buffer".to_string(),
                    source: Some(Box::new(e)),
                })?;
        }

        for index in reader
            .read_indices()
            .ok_or(Error {
                message: "failed to read indices".to_string(),
                source: None,
            })?
            .into_u32()
        {
            indices
                .write_all(bytemuck::bytes_of(&index))
                .map_err(|e| Error {
                    message: "failed to write indices to staging buffer".to_string(),
                    source: Some(Box::new(e)),
                })?;
        }

        Ok(())
    }

    fn load_materials(&self, mut materials: &mut [u8]) -> Result<(), Error> {
        for material in self.document.materials() {
            materials
                .write_all(bytemuck::bytes_of(&Material::new(
                    material.pbr_metallic_roughness().metallic_factor(),
                    material.pbr_metallic_roughness().roughness_factor(),
                    material.emissive_strength().unwrap_or(0.0),
                    material.ior().unwrap_or(0.0),
                    material
                        .pbr_metallic_roughness()
                        .base_color_texture()
                        .map(|info| info.texture().index() as u32)
                        .unwrap_or(0),
                    if material
                        .pbr_metallic_roughness()
                        .base_color_texture()
                        .is_some()
                    {
                        1
                    } else {
                        0
                    },
                    material.pbr_metallic_roughness().base_color_factor(),
                )))
                .map_err(|e| Error {
                    message: "failed write material to staging buffer".to_string(),
                    source: Some(Box::new(e)),
                })?;
        }

        Ok(())
    }

    fn load_objects(&self, mut objects: &mut [u8]) -> Result<(), Error> {
        for node in self.document.nodes() {
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

                    objects
                        .write_all(bytemuck::bytes_of(&Object::new(
                            transform,
                            mesh.index() as u32,
                        )))
                        .map_err(|e| Error {
                            message: "failed to write object to staging buffer".to_string(),
                            source: Some(Box::new(e)),
                        })?;
                }
                None => continue,
            }
        }

        Ok(())
    }

    fn load_textures(&self, queue: &wgpu::Queue, textures: &[wgpu::Texture]) -> Result<(), Error> {
        for (gltf_texture, gpu_texture) in self.document.textures().zip(textures) {
            match gltf_texture.source().source() {
                gltf::image::Source::View { view, .. } => {
                    let reader = self.read_buffer(&view.buffer())?;
                    let slice = &reader.as_ref()[view.offset()..view.offset() + view.length()];

                    let image = image::load_from_memory(slice)
                        .map_err(|e| Error {
                            message: "failed to load image".to_string(),
                            source: Some(Box::new(e)),
                        })?
                        .into_rgba8();
                    let width = image.width();
                    let height = image.height();
                    let image_data = image.to_vec();

                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: gpu_texture,
                            mip_level: 0,
                            aspect: wgpu::TextureAspect::All,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        bytemuck::cast_slice(image_data.as_slice()),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(width * 4),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                gltf::image::Source::Uri { uri, .. } => {
                    let file = File::open(uri).map_err(|e| Error {
                        message: format!("failed to open {uri}"),
                        source: Some(Box::new(e)),
                    })?;
                    let slice = unsafe {
                        &memmap2::Mmap::map(&file).map_err(|e| Error {
                            message: format!("failed to mmap {uri}"),
                            source: Some(Box::new(e)),
                        })?
                    };

                    let image = image::load_from_memory(slice)
                        .map_err(|e| Error {
                            message: format!("failed to load {uri} as image"),
                            source: Some(Box::new(e)),
                        })?
                        .into_rgba8();
                    let width = image.width();
                    let height = image.height();
                    let image_data = image.to_vec();

                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: gpu_texture,
                            mip_level: 0,
                            aspect: wgpu::TextureAspect::All,
                            origin: wgpu::Origin3d::ZERO,
                        },
                        bytemuck::cast_slice(image_data.as_slice()),
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(width * 4),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
        }

        Ok(())
    }

    fn load_camera(&self) -> Result<Option<Camera>, Error> {
        let node = match self.document.nodes().find(|node| node.camera().is_some()) {
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
            gltf::camera::Projection::Orthographic(_) => Err(Error {
                message: "todo: support for orthographic projection".to_string(),
                source: None,
            })?,
            gltf::camera::Projection::Perspective(perspective) => bytemuck::cast_slice(
                Perspective3::new(
                    perspective.aspect_ratio().ok_or(Error {
                        message: "failed to load aspect ratio from camera".to_string(),
                        source: None,
                    })?,
                    perspective.yfov(),
                    perspective.znear(),
                    perspective.zfar().ok_or(Error {
                        message: "failed to load zfar from camera".to_string(),
                        source: None,
                    })?,
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
        self.document
            .nodes()
            .filter(|object| object.mesh().is_some())
            .count() as u32
    }

    fn mesh_count(&self) -> u32 {
        self.document.meshes().count() as u32
    }

    fn primitive_count(&self) -> u32 {
        self.document
            .meshes()
            .map(|mesh| mesh.primitives().len())
            .sum::<usize>() as u32
    }

    fn vertex_count(&self) -> Result<u32, Error> {
        Ok(self
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .try_fold(0, |acc, primitive| {
                Result::<usize, Error>::Ok(
                    acc + primitive
                        .reader(|_| Some(self.bin))
                        .read_positions()
                        .ok_or(Error {
                            message: "failed to read positions".to_string(),
                            source: None,
                        })?
                        .len(),
                )
            })? as u32)
    }

    fn index_count(&self) -> Result<u32, Error> {
        Ok(self
            .document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .try_fold(0, |acc, primitive| {
                Result::<usize, Error>::Ok(
                    acc + primitive
                        .reader(|_| Some(self.bin))
                        .read_indices()
                        .ok_or(Error {
                            message: "failed to read indices".to_string(),
                            source: None,
                        })?
                        .into_u32()
                        .len(),
                )
            })? as u32)
    }

    fn material_count(&self) -> u32 {
        self.document.materials().len() as u32
    }

    fn get_primitive_vertex_count(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Error> {
        self.document
            .meshes()
            .nth(mesh_index)
            .unwrap()
            .primitives()
            .nth(primitive_index)
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(self.bin));

                Ok(reader
                    .read_positions()
                    .ok_or(Error {
                        message: "failed to read positions".to_string(),
                        source: None,
                    })?
                    .len() as u32)
            })
            .unwrap()
    }

    fn get_primitive_first_vertex(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Error> {
        Ok(self
            .document
            .meshes()
            .take(mesh_index)
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(&self.bin));

                Ok(reader
                    .read_positions()
                    .ok_or(Error {
                        message: "failed to read positions".to_string(),
                        source: None,
                    })?
                    .len() as u32)
            })
            .try_fold(0, |acc, e: Result<_, Error>| {
                Result::<_, Error>::Ok(acc + e?)
            })?
            + self
                .document
                .meshes()
                .nth(mesh_index)
                .unwrap()
                .primitives()
                .take(primitive_index)
                .map(|primitive| {
                    let reader = primitive.reader(|_| Some(self.bin));

                    Ok(reader
                        .read_positions()
                        .ok_or(Error {
                            message: "failed to read positions".to_string(),
                            source: None,
                        })?
                        .len() as u32)
                })
                .try_fold(0, |acc, e: Result<_, Error>| {
                    Result::<_, Error>::Ok(acc + e?)
                })?)
    }

    fn get_primitive_index_count(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Error> {
        self.document
            .meshes()
            .nth(mesh_index)
            .unwrap()
            .primitives()
            .nth(primitive_index)
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(self.bin));

                Ok(reader
                    .read_indices()
                    .ok_or(Error {
                        message: "failed to read indices".to_string(),
                        source: None,
                    })?
                    .into_u32()
                    .len() as u32)
            })
            .unwrap()
    }

    fn get_primitive_first_index(
        &self,
        mesh_index: usize,
        primitive_index: usize,
    ) -> Result<u32, Error> {
        Ok(self
            .document
            .meshes()
            .take(mesh_index)
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| {
                let reader = primitive.reader(|_| Some(&self.bin));

                Ok(reader
                    .read_indices()
                    .ok_or(Error {
                        message: "failed to load indices".to_string(),
                        source: None,
                    })?
                    .into_u32()
                    .len() as u32)
            })
            .try_fold(0, |acc, e: Result<_, Error>| {
                Result::<_, Error>::Ok(acc + e?)
            })?
            + self
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
                        .ok_or(Error {
                            message: "failed to read indices".to_string(),
                            source: None,
                        })?
                        .into_u32()
                        .len() as u32)
                })
                .try_fold(0, |acc, e: Result<_, Error>| {
                    Result::<_, Error>::Ok(acc + e?)
                })?)
    }

    pub fn texture_descriptor(&self, texture: &gltf::Texture) -> Result<TextureDesc, Error> {
        match texture.source().source() {
            gltf::image::Source::View { view, .. } => {
                let reader = self.read_buffer(&view.buffer())?;

                let slice = &reader.as_ref()[view.offset()..view.offset() + view.length()];

                let (width, height) = image::ImageReader::new(io::Cursor::new(slice))
                    .with_guessed_format()
                    .map_err(|e| Error {
                        message: "failed load texture image".to_string(),
                        source: Some(Box::new(e)),
                    })?
                    .into_dimensions()
                    .map_err(|e| Error {
                        message: "failed to load texture image".to_string(),
                        source: Some(Box::new(e)),
                    })?;

                Ok(TextureDesc { width, height })
            }
            gltf::image::Source::Uri { uri, .. } => {
                let file = File::open(uri).map_err(|e| Error {
                    message: format!("failed to load {uri}"),
                    source: Some(Box::new(e)),
                })?;

                let mmap = unsafe {
                    memmap2::Mmap::map(&file).map_err(|e| Error {
                        message: format!("failed to mmap {uri}"),
                        source: Some(Box::new(e)),
                    })?
                };

                let (width, height) = image::ImageReader::new(io::Cursor::new(&mmap))
                    .with_guessed_format()
                    .map_err(|e| Error {
                        message: "failed load texture image".to_string(),
                        source: Some(Box::new(e)),
                    })?
                    .into_dimensions()
                    .map_err(|e| Error {
                        message: "failed to load texture image".to_string(),
                        source: Some(Box::new(e)),
                    })?;

                Ok(TextureDesc { width, height })
            }
        }
    }

    fn texture_descriptors(&self) -> Result<Vec<TextureDesc>, Error> {
        self.document
            .textures()
            .map(|texture| self.texture_descriptor(&texture))
            .collect::<Result<Vec<_>, _>>()
    }
}

impl<'a> AsRef<[u8]> for BufferReader<'a> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Bin(bin) => bin,
            Self::File(mmap) => mmap,
        }
    }
}

impl<'data> super::Scene for Scene<'data> {
    fn load(
        &self,
        queue: &wgpu::Queue,
        objects: &mut [u8],
        meshes: &mut [u8],
        primitives: &mut [u8],
        vertices: &mut [u8],
        indices: &mut [u8],
        materials: &mut [u8],
        textures: &[wgpu::Texture],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.load_meshes(meshes, primitives, vertices, indices)?;
        self.load_objects(objects)?;
        self.load_materials(materials)?;
        self.load_textures(queue, textures)?;

        Ok(())
    }

    fn desc(&self) -> Result<super::SceneDesc, Box<dyn std::error::Error>> {
        let camera = match self.load_camera() {
            Ok(Some(camera)) => camera,
            Ok(None) => Err(Error {
                message: "failed to load camera from scene".to_string(),
                source: None,
            })?,
            Err(e) => Err(e)?,
        };

        let blas_entries = self
            .document
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (mesh, node.transform().matrix())))
            .enumerate()
            .map(|(mesh_index, (mesh, transform))| {
                let geometries = mesh
                    .primitives()
                    .enumerate()
                    .map(|(primitive_index, _)| {
                        let first_vertex =
                            self.get_primitive_first_vertex(mesh_index, primitive_index)?;
                        let vertex_count =
                            self.get_primitive_vertex_count(mesh_index, primitive_index)?;
                        let first_index =
                            self.get_primitive_first_index(mesh_index, primitive_index)?;
                        let index_count =
                            self.get_primitive_index_count(mesh_index, primitive_index)?;

                        Ok(BlasGeometry {
                            first_vertex,
                            vertex_count,
                            first_index,
                            index_count,
                        })
                    })
                    .collect::<Result<_, Box<dyn std::error::Error>>>()?;

                Ok(BlasEntry {
                    transform,
                    geometries,
                })
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;

        Ok(SceneDesc {
            world: camera.world,
            projection: camera.projection,
            objects: self.object_count(),
            meshes: self.mesh_count(),
            primitives: self.primitive_count(),
            vertices: self.vertex_count()?,
            indices: self.index_count()?,
            materials: self.material_count(),
            blas_entries,
            textures: self.texture_descriptors()?,
        })
    }
}
