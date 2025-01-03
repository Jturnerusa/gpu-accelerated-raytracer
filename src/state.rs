use std::{
    collections::HashMap,
    iter,
    num::{NonZeroU32, NonZeroU64},
    sync,
};

use nalgebra::Matrix4;

use crate::scene::{
    BlasEntry, Camera, Light, Material, Mesh, Object, Primitive, Scene, TextureDesc, Vertex,
};

#[derive(Debug)]
pub struct Error {
    message: String,
    source: Option<Box<dyn std::error::Error>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view: [[f32; 4]; 4],
    perspective: [[f32; 4]; 4],
    width: u32,
    height: u32,
    objects: u32,
    lights: u32,
    chunk_size: u32,
    bounces: u32,
    seed: u32,
    current_chunk: u32,
    samples: u32,
    p0: [u32; 3],
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
    surface_config: Option<wgpu::SurfaceConfiguration>,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: Option<wgpu::RenderPipeline>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    uniforms: Option<Uniforms>,
    uniforms_buffer: Option<wgpu::Buffer>,
    objects_buffer: Option<wgpu::Buffer>,
    meshes_buffer: Option<wgpu::Buffer>,
    primitives_buffer: Option<wgpu::Buffer>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    materials_buffer: Option<wgpu::Buffer>,
    lights_buffer: Option<wgpu::Buffer>,
    texture_descriptors_buffer: Option<wgpu::Buffer>,
    textures: Option<Vec<wgpu::Texture>>,
    tlas: Option<wgpu::TlasPackage>,
    samples: Option<wgpu::Texture>,
    sampler: Option<wgpu::Sampler>,
    pixel_buffer: Option<wgpu::Buffer>,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "state error: {}", self.message)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.source.as_ref() {
            Some(source) => Some(&**source),
            None => None,
        }
    }
}

impl<'surface> State<'surface, &'surface sdl2::video::Window> {
    pub async fn new(
        window_desc: Option<WindowDesc<&'surface sdl2::video::Window>>,
    ) -> Result<Self, Error> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = match window_desc {
            Some(w) => Some(unsafe {
                instance
                    .create_surface_unsafe(
                        wgpu::SurfaceTargetUnsafe::from_window(w.window).map_err(|e| Error {
                            message: "failed to create surface from sdl2 window".to_string(),
                            source: Some(Box::new(e)),
                        })?,
                    )
                    .map_err(|e| Error {
                        message: "failed to create surface".to_string(),
                        source: Some(Box::new(e)),
                    })?
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
            .ok_or(Error {
                message: "failed to request adapter".to_string(),
                source: None,
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::EXPERIMENTAL_RAY_QUERY
                        | wgpu::Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE
                        | wgpu::Features::TEXTURE_BINDING_ARRAY
                        | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                        | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| Error {
                message: "failed to request device".to_string(),
                source: Some(Box::new(e))
            })?;

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

        Ok(Self {
            window: window_desc,
            surface,
            surface_config,
            instance,
            device,
            queue,
            render_pipeline: None,
            compute_pipeline: None,
            bind_group: None,
            bind_group_layout: None,
            uniforms: None,
            uniforms_buffer: None,
            objects_buffer: None,
            meshes_buffer: None,
            primitives_buffer: None,
            vertex_buffer: None,
            index_buffer: None,
            materials_buffer: None,
            lights_buffer: None,
            textures: None,
            texture_descriptors_buffer: None,
            tlas: None,
            samples: None,
            sampler: None,
            pixel_buffer: None,
        })
    }

    pub fn render(&mut self) -> Result<(), Error> {
        let output = self
            .surface
            .as_ref()
            .unwrap()
            .get_current_texture()
            .map_err(|e| Error {
                message: "failed to get texture".to_string(),
                source: Some(Box::new(e)),
            })?;

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

    pub fn download_frame(&mut self, data: &mut Vec<u8>) -> Result<(), Error> {
        self.copy_texture_to_buffer();

        let (sender, receiver) = sync::mpsc::channel();

        self.pixel_buffer.as_ref().unwrap().slice(..).map_async(
            wgpu::MapMode::Read,
            move |result| {
                sender.send(result).unwrap();
            },
        );

        self.wait();

        receiver
            .recv()
            .map_err(|e| Error {
                message: "failed to map buffer".to_string(),
                source: Some(Box::new(e)),
            })?
            .map_err(|e| Error {
                message: "failed to receive from sender".to_string(),
                source: Some(Box::new(e)),
            })?;

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

    pub fn process_chunk(&mut self) -> Result<(), Error> {
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

                compute_pass.set_pipeline(self.compute_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &self.bind_group, &[]);
                compute_pass.dispatch_workgroups(
                    uniforms.chunk_size / 8,
                    uniforms.chunk_size / 8,
                    1,
                );
            }

            self.queue.submit(iter::once(encoder.finish()));
        }

        self.uniforms.as_mut().unwrap().current_chunk += 1;

        self.queue.write_buffer(
            self.uniforms_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(self.uniforms.as_ref().unwrap()),
        );

        self.queue.submit(iter::empty());

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
        camera: Option<Camera>,
        scene: &dyn Scene,
    ) -> Result<(), Error> {
        let scene_desc = scene.desc().map_err(|e| Error {
            message: "failed to read scene description".to_string(),
            source: Some(e),
        })?;

        let camera = match camera {
            Some(camera) => camera,
            None => match scene.load_camera() {
                Ok(Some(camera)) => camera,
                Ok(None) => Err(Error {
                    message: "failed to load camera from scene".to_string(),
                    source: None,
                })?,
                Err(e) => Err(Error {
                    message: "failed to load camera from scene".to_string(),
                    source: Some(e),
                })?,
            },
        };

        let uniforms = Uniforms {
            view: camera.world,
            perspective: camera.projection,
            width,
            height,
            objects: scene_desc.objects,
            lights: scene_desc.lights,
            bounces,
            current_chunk: 0,
            chunk_size,
            seed,
            samples,
            p0: Default::default(),
        };

        let uniforms_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let objects_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("objects buffer"),
            size: (std::mem::size_of::<Object>() as u32 * scene_desc.objects) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let meshes_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh buffer"),
            size: (std::mem::size_of::<Mesh>() as u32 * scene_desc.meshes) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let primitives_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("primitives buffer"),
            size: (std::mem::size_of::<Primitive>() as u32 * scene_desc.primitives) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex buffer"),
            size: (std::mem::size_of::<Vertex>() as u32 * scene_desc.vertices) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });

        let index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: (std::mem::size_of::<u32>() as u32 * scene_desc.indices) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });

        let materials_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("materials buffer"),
            size: (std::mem::size_of::<Material>() as u32 * scene_desc.materials) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lights_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights buffer"),
            size: (std::mem::size_of::<Light>() as u32 * scene_desc.lights) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_descriptors_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture descriptors buffer"),
            size: (std::mem::size_of::<TextureDesc>() as u32 * scene_desc.materials) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pixel_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixels"),
            size: u64::from(width) * u64::from(height) * std::mem::size_of::<f32>() as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut uniforms_mapped = self
            .queue
            .write_buffer_with(
                &uniforms_buffer,
                0,
                NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64).unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut objects_mapped = self
            .queue
            .write_buffer_with(
                &objects_buffer,
                0,
                NonZeroU64::new(
                    (std::mem::size_of::<Object>() * scene_desc.objects as usize) as u64,
                )
                .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut meshes_mapped = self
            .queue
            .write_buffer_with(
                &meshes_buffer,
                0,
                NonZeroU64::new((std::mem::size_of::<Mesh>() * scene_desc.meshes as usize) as u64)
                    .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut primitives_mapped = self
            .queue
            .write_buffer_with(
                &primitives_buffer,
                0,
                NonZeroU64::new(
                    (std::mem::size_of::<Primitive>() * scene_desc.primitives as usize) as u64,
                )
                .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut vertices_mapped = self
            .queue
            .write_buffer_with(
                &vertex_buffer,
                0,
                NonZeroU64::new(
                    (std::mem::size_of::<Vertex>() * scene_desc.vertices as usize) as u64,
                )
                .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut indices_mapped = self
            .queue
            .write_buffer_with(
                &index_buffer,
                0,
                NonZeroU64::new((std::mem::size_of::<u32>() * scene_desc.indices as usize) as u64)
                    .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut materials_mapped = self
            .queue
            .write_buffer_with(
                &materials_buffer,
                0,
                NonZeroU64::new(
                    (std::mem::size_of::<Material>() * scene_desc.materials as usize) as u64,
                )
                .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let mut lights_mapped = self
            .queue
            .write_buffer_with(
                &lights_buffer,
                0,
                NonZeroU64::new((std::mem::size_of::<Light>() * scene_desc.lights as usize) as u64)
                    .unwrap(),
            )
            .ok_or(Error {
                message: "failed to map buffer".to_string(),
                source: None,
            })?;

        let textures = if scene_desc.textures.is_empty() {
            iter::once(create_texture(
                &self.device,
                1,
                1,
                wgpu::TextureFormat::Rgba8Unorm,
            ))
            .collect::<Vec<_>>()
        } else {
            scene_desc
                .textures
                .iter()
                .map(|desc| {
                    create_texture(
                        &self.device,
                        desc.width,
                        desc.height,
                        wgpu::TextureFormat::Rgba8Unorm,
                    )
                })
                .collect::<Vec<_>>()
        };

        let texture_views = textures
            .iter()
            .map(|texture| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    ..Default::default()
                })
            })
            .collect::<Vec<_>>();

        uniforms_mapped
            .as_mut()
            .copy_from_slice(bytemuck::bytes_of(&uniforms));

        scene
            .load(
                &self.queue,
                objects_mapped.as_mut(),
                meshes_mapped.as_mut(),
                primitives_mapped.as_mut(),
                vertices_mapped.as_mut(),
                indices_mapped.as_mut(),
                materials_mapped.as_mut(),
                lights_mapped.as_mut(),
                textures.as_slice(),
            )
            .map_err(|e| Error {
                message: "failed to upload scene".to_string(),
                source: Some(e),
            })?;

        self.queue.write_buffer(
            &texture_descriptors_buffer,
            0,
            bytemuck::cast_slice(scene_desc.textures.as_slice()),
        );

        drop(uniforms_mapped);
        drop(objects_mapped);
        drop(meshes_mapped);
        drop(primitives_mapped);
        drop(vertices_mapped);
        drop(indices_mapped);
        drop(materials_mapped);
        drop(lights_mapped);

        self.queue.submit(iter::empty());

        let tlas_package = configure_acceleration_structures(
            &self.device,
            &self.queue,
            &vertex_buffer,
            &index_buffer,
            scene_desc.blas_entries.as_slice(),
        );

        let samples = create_texture(
            &self.device,
            width,
            height,
            wgpu::TextureFormat::Rgba32Float,
        );
        let samples_view = samples.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = make_bind_group_layout(&self.device, textures.len() as u32);

        let bind_group = make_bind_group(
            &self.device,
            &bind_group_layout,
            &uniforms_buffer,
            &objects_buffer,
            &meshes_buffer,
            &primitives_buffer,
            &vertex_buffer,
            &index_buffer,
            &materials_buffer,
            &lights_buffer,
            &texture_descriptors_buffer,
            &samples_view,
            texture_views.iter().collect::<Vec<_>>().as_slice(),
            &tlas_package,
            &sampler,
        );

        let compute_pipeline = make_compute_pipeline(&self.device, &bind_group_layout);

        let render_pipeline = match &self.surface_config {
            Some(config) => Some(make_render_pipeline(
                &self.device,
                &bind_group_layout,
                config,
            )),
            None => None,
        };

        self.bind_group = Some(bind_group);
        self.uniforms = Some(uniforms);
        self.uniforms_buffer = Some(uniforms_buffer);
        self.compute_pipeline = Some(compute_pipeline);
        self.render_pipeline = render_pipeline;
        self.bind_group_layout = Some(bind_group_layout);
        self.objects_buffer = Some(objects_buffer);
        self.meshes_buffer = Some(meshes_buffer);
        self.primitives_buffer = Some(primitives_buffer);
        self.vertex_buffer = Some(vertex_buffer);
        self.index_buffer = Some(index_buffer);
        self.materials_buffer = Some(materials_buffer);
        self.lights_buffer = Some(lights_buffer);
        self.texture_descriptors_buffer = Some(texture_descriptors_buffer);
        self.textures = Some(textures);
        self.tlas = Some(tlas_package);
        self.samples = Some(samples);
        self.pixel_buffer = Some(pixel_buffer);
        self.sampler = Some(sampler);

        Ok(())
    }
}

fn create_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
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
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn make_bind_group_layout(device: &wgpu::Device, textures_count: u32) -> wgpu::BindGroupLayout {
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
            // lights
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // texture descriptors
            wgpu::BindGroupLayoutEntry {
                binding: 8,
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
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure,
                count: None,
            },
            // samples
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // textures
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: Some(NonZeroU32::new(textures_count).unwrap()),
            },
            // sampler
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn make_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    uniforms_buffer: &wgpu::Buffer,
    objects_buffer: &wgpu::Buffer,
    meshes_buffer: &wgpu::Buffer,
    primitives_buffer: &wgpu::Buffer,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    materials_buffer: &wgpu::Buffer,
    lights_buffer: &wgpu::Buffer,
    texture_descriptors_buffer: &wgpu::Buffer,
    samples_view: &wgpu::TextureView,
    texture_views: &[&wgpu::TextureView],
    tlas_package: &wgpu::TlasPackage,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group 1"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniforms_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: objects_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: meshes_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: primitives_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: vertex_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: index_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: materials_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: lights_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: texture_descriptors_buffer,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::AccelerationStructure(tlas_package.tlas()),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(samples_view),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureViewArray(texture_views),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::Sampler(sampler),
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

fn write_to_buffer(queue: &wgpu::Queue, buffer: &wgpu::Buffer, data: &[u8]) -> Result<(), Error> {
    let mut mapped = queue
        .write_buffer_with(
            buffer,
            0,
            NonZeroU64::new(std::mem::size_of_val(data) as u64).unwrap(),
        )
        .ok_or(Error {
            message: "failed to map buffer".to_string(),
            source: None,
        })?;

    mapped.as_mut().copy_from_slice(data);

    drop(mapped);

    queue.submit(iter::empty());

    Ok(())
}

fn configure_acceleration_structures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    blas_entries: &[BlasEntry],
) -> wgpu::TlasPackage {
    let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("tlas"),
        max_instances: blas_entries.len() as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });

    let mut tlas_package = wgpu::TlasPackage::new(tlas);

    let blases = blas_entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let descriptors = entry
                .geometries
                .iter()
                .map(|geometry| {
                    (
                        wgpu::BlasTriangleGeometrySizeDescriptor {
                            vertex_format: wgpu::VertexFormat::Float32x3,
                            vertex_count: geometry.vertex_count,
                            index_format: Some(wgpu::IndexFormat::Uint32),
                            index_count: Some(geometry.index_count),
                            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                        },
                        geometry,
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
                    descriptors: descriptors
                        .iter()
                        .map(|(descriptor, _)| descriptor.clone())
                        .collect(),
                },
            );

            // The raw transform array is in the wrong order.
            // Matrix4 storage happens to be in the correct order so
            // we can just use that.
            let m = Matrix4::from_row_slice(bytemuck::cast_slice(entry.transform.as_slice()));

            tlas_package[i] = Some(wgpu::TlasInstance::new(
                &blas,
                m.as_slice()[0..12].try_into().unwrap(),
                i as u32,
                0xff,
            ));

            (descriptors, blas)
        })
        .collect::<Vec<_>>();

    let build_entries = blases
        .iter()
        .map(|(descriptors, blas)| {
            let geometries = descriptors
                .iter()
                .map(|(descriptor, geometry)| wgpu::BlasTriangleGeometry {
                    size: descriptor,
                    vertex_buffer,
                    first_vertex: geometry.first_vertex,
                    vertex_stride: std::mem::size_of::<Vertex>() as u64,
                    index_buffer: Some(index_buffer),
                    index_buffer_offset: Some(
                        geometry.first_index as u64 * std::mem::size_of::<u32>() as u64,
                    ),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                })
                .collect::<Vec<_>>();

            wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(geometries),
            }
        })
        .collect::<Vec<_>>();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(build_entries.iter(), iter::once(&tlas_package));

    queue.submit(Some(encoder.finish()));

    tlas_package
}
