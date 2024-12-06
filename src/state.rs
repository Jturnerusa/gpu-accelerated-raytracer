use std::{collections::HashMap, iter, sync};

use encase::{ShaderSize, ShaderType, StorageBuffer, UniformBuffer};
use nalgebra::Matrix4;

use crate::scene::{Camera, Material, Mesh, Object, Primitive, Vertex};

#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct Uniforms {
    view: Matrix4<f32>,
    perspective: Matrix4<f32>,
    width: u32,
    height: u32,
    objects: u32,
    chunk_size: u32,
    bounces: u32,
    seed: u32,
    current_chunk: u32,
    samples: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct WindowDesc<W> {
    pub window: W,
    pub width: u32,
    pub height: u32,
}

pub struct State<'surface, W> {
    window: Option<WindowDesc<W>>,
    surface: Option<wgpu::Surface<'surface>>,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: Option<wgpu::RenderPipeline>,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: wgpu::BindGroupLayout,
    uniforms: Option<Uniforms>,
    uniforms_buffer: Option<wgpu::Buffer>,
    objects_buffer: Option<wgpu::Buffer>,
    meshes_buffer: Option<wgpu::Buffer>,
    primitives_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    materials_buffer: Option<wgpu::Buffer>,
    tlas_vertices_buffer: Option<wgpu::Buffer>,
    tlas: Option<wgpu::TlasPackage>,
    samples: Option<wgpu::Texture>,
    pixel_buffer: Option<wgpu::Buffer>,
}

impl<'surface> State<'surface, &'surface sdl2::video::Window> {
    pub async fn new(
        window_desc: Option<WindowDesc<&'surface sdl2::video::Window>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = match window_desc {
            Some(w) => Some(unsafe {
                instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(w.window)?)?
            }),
            None => None,
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("failed to request adapter".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::EXPERIMENTAL_RAY_QUERY
                        | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE,
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        let surface_config = match &surface {
            Some(surface) => {
                let caps = surface.get_capabilities(&adapter);

                let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap();

                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format,
                    width: window_desc.unwrap().width,
                    height: window_desc.unwrap().height,
                    present_mode: caps.present_modes[0],
                    alpha_mode: caps.alpha_modes[0],
                    view_formats: Vec::new(),
                    desired_maximum_frame_latency: 2,
                };

                surface.configure(&device, &config);

                Some(config)
            }
            None => None,
        };

        let bind_group_layout = make_bind_group_layout(&device);

        let compute_pipeline = make_compute_pipeline(&device, &bind_group_layout);

        let render_pipeline = if window_desc.is_some() {
            Some(make_render_pipeline(
                &device,
                &bind_group_layout,
                surface_config.as_ref().unwrap(),
            ))
        } else {
            None
        };

        Ok(Self {
            window: window_desc,
            surface,
            instance,
            device,
            queue,
            render_pipeline,
            compute_pipeline,
            bind_group: None,
            bind_group_layout,
            uniforms: None,
            uniforms_buffer: None,
            objects_buffer: None,
            meshes_buffer: None,
            primitives_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            materials_buffer: None,
            tlas_vertices_buffer: None,
            tlas: None,
            samples: None,
            pixel_buffer: None,
        })
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let output = self.surface.as_ref().unwrap().get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(self.render_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));

        output.present();

        Ok(())
    }
}

impl<'surface, W> State<'surface, W> {
    pub fn wait(&self) {
        self.device.poll(wgpu::MaintainBase::Wait);
    }

    pub fn download_frame(&mut self, data: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        self.copy_texture_to_buffer();

        let (sender, receiver) = sync::mpsc::channel();

        self.pixel_buffer.as_ref().unwrap().slice(..).map_async(
            wgpu::MapMode::Read,
            move |result| {
                sender.send(result).unwrap();
            },
        );

        self.wait();

        receiver.recv()??;

        data.extend_from_slice(
            &self
                .pixel_buffer
                .as_ref()
                .unwrap()
                .slice(..)
                .get_mapped_range(),
        );

        Ok(())
    }

    fn copy_texture_to_buffer(&mut self) {
        let uniforms = self.uniforms.unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: self.samples.as_ref().unwrap(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: self.pixel_buffer.as_ref().unwrap(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(uniforms.width * std::mem::size_of::<f32>() as u32 * 4),
                        rows_per_image: Some(uniforms.height),
                    },
                },
                wgpu::Extent3d {
                    width: uniforms.width,
                    height: uniforms.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.submit(iter::once(encoder.finish()));
    }

    pub fn is_finished(&self) -> bool {
        let uniforms = self.uniforms.unwrap();

        uniforms.current_chunk >= (uniforms.width * uniforms.height) / uniforms.chunk_size
    }

    pub fn process_chunk(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        {
            let uniforms = self.uniforms.as_mut().unwrap();

            let chunks_per_frame = (uniforms.width * uniforms.height) / uniforms.chunk_size;

            if uniforms.current_chunk >= chunks_per_frame {
                return Ok(());
            }

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.compute_pipeline);
                compute_pass.set_bind_group(0, &self.bind_group, &[]);
                compute_pass.dispatch_workgroups(
                    uniforms.chunk_size / 8,
                    uniforms.chunk_size / 8,
                    1,
                );
            }

            self.queue.submit(iter::once(encoder.finish()));
        }

        self.wait();

        self.uniforms.as_mut().unwrap().current_chunk += 1;

        write_uniforms_to_buffer(
            &self.queue,
            self.uniforms_buffer.as_ref().unwrap(),
            self.uniforms.unwrap(),
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_scene(
        &mut self,
        width: u32,
        height: u32,
        seed: u32,
        samples: u32,
        bounces: u32,
        chunk_size: u32,
        camera: Camera,
        objects: &[Object],
        meshes: &[Mesh],
        primitives: &[Primitive],
        vertices: &[Vertex],
        indices: &[u32],
        materials: &[Material],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let uniforms = Uniforms {
            view: camera.world,
            perspective: camera.projection,
            width,
            height,
            objects: objects.len() as u32,
            bounces,
            current_chunk: 0,
            chunk_size,
            seed,
            samples,
        };

        let uniforms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: Uniforms::SHADER_SIZE.get(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let objects_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("objects buffer"),
            size: objects.size().get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let meshes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh buffer"),
            size: meshes.size().get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let primitives_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("primitives buffer"),
            size: primitives.size().get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex buffer"),
            size: vertices.size().get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: indices.size().get(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });

        let materials_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("materials buffer"),
            size: materials.size().get(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        write_uniforms_to_buffer(&self.queue, &uniforms_buffer, uniforms)?;
        write_encased_to_buffer(&self.queue, &objects_buffer, objects)?;
        write_encased_to_buffer(&self.queue, &meshes_buffer, meshes)?;
        write_encased_to_buffer(&self.queue, &primitives_buffer, primitives)?;
        write_encased_to_buffer(&self.queue, &vertex_buffer, vertices)?;
        write_encased_to_buffer(&self.queue, &index_buffer, indices)?;
        write_encased_to_buffer(&self.queue, &materials_buffer, materials)?;

        let tlas_vertices = vertices
            .iter()
            .map(|vertex| [vertex.position[0], vertex.position[1], vertex.position[2]])
            .collect::<Vec<_>>();

        let tlas_vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tlas vertices buffer"),
            size: tlas_vertices.size().get(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });

        write_encased_to_buffer(&self.queue, &tlas_vertex_buffer, &tlas_vertices)?;

        let tlas_package = configure_tlas(
            &self.device,
            &self.queue,
            &index_buffer,
            &tlas_vertex_buffer,
            objects,
            meshes,
            primitives,
        )?;

        let texture = create_texture(&self.device, width, height);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group 1"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniforms_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &objects_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &meshes_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &primitives_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &vertex_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &index_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &materials_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas_package.tlas()),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });

        let pixel_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels"),
            size: u64::from(width) * u64::from(height) * std::mem::size_of::<f32>() as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.bind_group = Some(bind_group);
        self.uniforms = Some(uniforms);
        self.uniforms_buffer = Some(uniforms_buffer);
        self.objects_buffer = Some(objects_buffer);
        self.meshes_buffer = Some(meshes_buffer);
        self.primitives_buffer = Some(primitives_buffer);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.materials_buffer = Some(materials_buffer);
        self.tlas_vertices_buffer = Some(tlas_vertex_buffer);
        self.tlas = Some(tlas_package);
        self.samples = Some(texture);
        self.pixel_buffer = Some(pixel_buffer);
        Ok(())
    }
}

fn create_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("storage texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn write_uniforms_to_buffer(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    uniforms: Uniforms,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = queue
        .write_buffer_with(buffer, 0, Uniforms::SHADER_SIZE)
        .unwrap();

    let mut storage = UniformBuffer::new(memory.as_mut());

    storage.write(&uniforms)?;

    drop(memory);

    queue.submit(iter::empty());

    Ok(())
}

fn write_encased_to_buffer<T: encase::ShaderType + encase::internal::WriteInto>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    data: T,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut memory = queue.write_buffer_with(buffer, 0, data.size()).unwrap();

    let mut storage = StorageBuffer::new(memory.as_mut());

    storage.write(&data)?;

    drop(memory);

    queue.submit(iter::empty());

    Ok(())
}

fn configure_tlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    index_buffer: &wgpu::Buffer,
    tlas_vertext_buffer: &wgpu::Buffer,
    objects: &[Object],
    meshes: &[Mesh],
    primitives: &[Primitive],
) -> Result<wgpu::TlasPackage, Box<dyn std::error::Error>> {
    let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        max_instances: objects.len() as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    let mut tlas_package = wgpu::TlasPackage::new(tlas);

    let blases = objects
        .iter()
        .copied()
        .enumerate()
        .map(|(i, object)| {
            let mesh = meshes[object.mesh as usize];
            let primitives = &primitives[mesh.primitive_start as usize
                ..mesh.primitive_start as usize + mesh.primitive_count as usize];

            let sizes = primitives
                .iter()
                .map(|p| {
                    (
                        wgpu::BlasTriangleGeometrySizeDescriptor {
                            vertex_format: wgpu::VertexFormat::Float32x3,
                            vertex_count: p.vertex_count,
                            index_format: Some(wgpu::IndexFormat::Uint32),
                            index_count: Some(p.index_count),
                            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                        },
                        p,
                    )
                })
                .collect::<Vec<_>>();

            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: sizes
                        .iter()
                        .map(|(size, _)| size.clone())
                        .collect::<Vec<_>>(),
                },
            );

            tlas_package[i] = Some(wgpu::TlasInstance::new(
                &blas,
                object.transform.transpose().as_slice()[0..12]
                    .try_into()
                    .unwrap(),
                i as u32,
                0xff,
            ));

            (sizes, blas)
        })
        .collect::<Vec<_>>();

    let build_entries = blases
        .iter()
        .map(|(sizes, blas)| {
            let geometries = sizes
                .iter()
                .map(|(size, p)| wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: tlas_vertext_buffer,
                    first_vertex: p.vertex_start,
                    vertex_stride: std::mem::size_of::<[f32; 3]>() as u64,
                    index_buffer: Some(index_buffer),
                    index_buffer_offset: Some(p.index_start as u64 * 4),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                })
                .collect::<Vec<_>>();

            wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(geometries),
            }
        })
        .collect::<Vec<_>>();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(build_entries.iter(), iter::once(&tlas_package));

    queue.submit(Some(encoder.finish()));

    Ok(tlas_package)
}

fn make_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // objects
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // meshes
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // primitives
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // vertices
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // indices
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // materials
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // tlas
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure,
                count: None,
            },
            // texture
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

fn make_compute_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute pipeline layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &HashMap::new(),
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
}

fn make_render_pipeline(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    surface_config: &wgpu::SurfaceConfiguration,
) -> wgpu::RenderPipeline {
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}
