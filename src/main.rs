#![allow(dead_code)]

mod deserialize;

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    iter,
    path::{Path, PathBuf},
};

use clap::Parser;
use encase::{ShaderSize, ShaderType, StorageBuffer, UniformBuffer};
use nalgebra::{Matrix4, Perspective3, Vector3, Vector4};
use winit::application::ApplicationHandler;

const WORKGROUP_SIZE_X: usize = 8;
const WORKGROUP_SIZE_Y: usize = 8;
const WORKGROUP_SIZE_Z: usize = 1;

#[derive(Clone, Debug, clap::Parser)]
struct Args {
    #[arg(long)]
    width: u32,
    #[arg(long)]
    aspect_ratio: f32,
    #[arg(long)]
    seed: u32,
    #[arg(long)]
    scene: PathBuf,
    #[arg(long)]
    chunk_size: u32,
    #[arg(long)]
    samples_per_pass: u32,
    #[arg(long)]
    bounces: u32,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct Uniforms {
    view: Matrix4<f32>,
    view_inverse: Matrix4<f32>,
    perspective: Matrix4<f32>,
    perspective_inverse: Matrix4<f32>,
    width: u32,
    height: u32,
    objects: u32,
    lights: u32,
    chunk_size: u32,
    bounces: u32,
    seed: u32,
    current_chunk: u32,
    samples_per_pass: u32,
    passes: u32,
}

#[derive(Clone, Copy, encase::ShaderType)]
struct Vertex {
    p: Vector3<f32>,
    n: Vector3<f32>,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct Light {
    color: Vector4<f32>,
    power: f32,
    size: f32,
    transform: Matrix4<f32>,
    transform_inverse: Matrix4<f32>,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct Material {
    metalic: f32,
    roughness: f32,
    color: Vector4<f32>,
}

#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct Object {
    transform: Matrix4<f32>,
    vertex_start: u32,
    vertex_count: u32,
    index_start: u32,
    index_count: u32,
    material: u32,
}

#[derive(Clone)]
struct Scene {
    perspective: Matrix4<f32>,
    view: Matrix4<f32>,
    vertex_buffer: Vec<Vertex>,
    index_buffer: Vec<u32>,
    lights: Vec<Light>,
    materials: Vec<Material>,
    objects: Vec<Object>,
}

struct State<'surface, W> {
    width: u32,
    height: u32,
    chunk_size: u32,
    surface: wgpu::Surface<'surface>,
    instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    uniforms_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    materials_buffer: wgpu::Buffer,
    objects_buffer: wgpu::Buffer,
    tlas: wgpu::TlasPackage,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    window: W,
    uniforms: Uniforms,
}

impl<'surface> State<'surface, &'surface winit::window::Window> {
    async fn new<'data>(
        width: u32,
        height: u32,
        window: &'surface winit::window::Window,
        vertices: &'data [Vertex],
        indices: &'data [u32],
        materials: &'data [Material],
        lights: &'data [Light],
        objects: &'data [Object],
        view: Matrix4<f32>,
        perspective: Matrix4<f32>,
        chunk_size: u32,
        bounces: u32,
        seed: u32,
        samples_per_pass: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
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

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap();

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: Vec::new(),
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        let uniforms = Uniforms {
            view,
            view_inverse: view.try_inverse().unwrap(),
            perspective,
            perspective_inverse: perspective.try_inverse().unwrap(),
            width,
            height,
            objects: objects.len() as u32,
            lights: lights.len() as u32,
            chunk_size,
            bounces,
            seed,
            current_chunk: 0,
            samples_per_pass,
            passes: 1,
        };

        let uniforms_buffer = configure_uniforms(&device, uniforms)?;
        let vertex_buffer = configure_buffer(
            &device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
            "vertices",
            vertices,
        )?;
        let index_buffer = configure_buffer(
            &device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
            "indices",
            indices,
        )?;
        let materials_buffer =
            configure_buffer(&device, wgpu::BufferUsages::STORAGE, "materials", materials)?;
        let objects_buffer =
            configure_buffer(&device, wgpu::BufferUsages::STORAGE, "objects", objects)?;
        let lights_buffer =
            configure_buffer(&device, wgpu::BufferUsages::STORAGE, "lights", lights)?;

        let tlas_package =
            build_acceleration_structures(&device, &queue, &vertex_buffer, &index_buffer, objects);

        let texture = create_texture(&device, width, height);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                // verticies
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // faces
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // materials
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // objects
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // tlas
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure,
                    count: None,
                },
                // texture
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group 1"),
            layout: &bind_group_layout,
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
                        buffer: &vertex_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &index_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &materials_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &objects_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &lights_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas_package.tlas()),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });

        Ok(Self {
            width,
            height,
            chunk_size,
            surface,
            instance,
            device,
            queue,
            uniforms_buffer,
            vertex_buffer,
            index_buffer,
            materials_buffer,
            objects_buffer,
            render_pipeline,
            compute_pipeline,
            bind_group,
            tlas: tlas_package,
            window,
            uniforms,
        })
    }

    fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let output = self.surface.get_current_texture()?;

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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));

        output.present();

        Ok(())
    }

    fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
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
                self.chunk_size / WORKGROUP_SIZE_X as u32,
                self.chunk_size / WORKGROUP_SIZE_Y as u32,
                WORKGROUP_SIZE_Z as u32,
            );
        }

        self.queue.submit(iter::once(encoder.finish()));

        let chunks_per_pass = (self.width * self.height) / self.uniforms.chunk_size;

        self.uniforms.current_chunk += 1;

        if self.uniforms.current_chunk >= chunks_per_pass {
            self.uniforms.current_chunk = 0;
            self.uniforms.passes += 1
        }

        self.write_uniforms()?;

        Ok(())
    }

    fn write_uniforms(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut encased_uniforms = UniformBuffer::new(Vec::new());

        encased_uniforms.write(&self.uniforms)?;

        self.queue
            .write_buffer(&self.uniforms_buffer, 0, &encased_uniforms.into_inner());

        Ok(())
    }
}

impl<'surface> ApplicationHandler for State<'surface, &'surface winit::window::Window> {
    fn resumed(&mut self, _: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            winit::event::WindowEvent::RedrawRequested => {
                match self.compute() {
                    Ok(()) => (),
                    Err(e) => {
                        eprintln!("{e}");
                        event_loop.exit();
                    }
                }
                match self.render() {
                    Ok(()) => (),
                    Err(e) => {
                        eprintln!("{e}");
                        event_loop.exit();
                    }
                }
                self.device.poll(wgpu::Maintain::Wait);
                self.window.request_redraw();
            }
            winit::event::WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        state: winit::event::ElementState::Pressed,
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            _ => (),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let height = (args.width as f32 / args.aspect_ratio) as u32;
    let scene = load_scene(&args.scene)?;

    let event_loop = winit::event_loop::EventLoop::new()?;
    let window = event_loop.create_window(winit::window::WindowAttributes::default())?;

    let mut state = State::new(
        args.width,
        height,
        &window,
        &scene.vertex_buffer,
        &scene.index_buffer,
        &scene.materials,
        &scene.lights,
        &scene.objects,
        scene.view,
        scene.perspective,
        find_best_chunk_size(
            args.chunk_size,
            args.width,
            WORKGROUP_SIZE_X as u32,
            WORKGROUP_SIZE_Y as u32,
        ),
        args.bounces,
        args.seed,
        args.samples_per_pass,
    )
    .await?;

    event_loop.run_app(&mut state)?;

    Ok(())
}

fn find_best_chunk_size(max: u32, width: u32, wgx: u32, wgy: u32) -> u32 {
    for x in (1..=max).rev() {
        if width % x == 0 && x % wgx == 0 && x % wgy == 0 {
            return x;
        }
    }
    1
}

fn load_scene(path: &Path) -> Result<Scene, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buff = String::new();

    file.read_to_string(&mut buff)?;

    let deserialized_scene: deserialize::Scene = serde_json::from_str(buff.as_str())?;

    let mesh_keys = deserialized_scene.meshes.keys().collect::<Vec<_>>();
    let material_keys = deserialized_scene.materials.keys().collect::<Vec<_>>();

    let meshes = mesh_keys
        .iter()
        .map(|key| deserialized_scene.meshes[key.as_str()].clone())
        .collect::<Vec<_>>();

    let materials = material_keys
        .iter()
        .map(|key| deserialized_scene.materials[key.as_str()])
        .map(|material| Material {
            metalic: material.metalic,
            roughness: material.roughness,
            color: material.color,
        })
        .collect::<Vec<_>>();

    let vertex_buffer = meshes
        .iter()
        .flat_map(|mesh| {
            mesh.verts
                .iter()
                .copied()
                .zip(mesh.normals.iter().copied())
                .map(|(p, n)| Vertex { p, n })
        })
        .collect::<Vec<_>>();

    let index_buffer = meshes
        .iter()
        .flat_map(|mesh| {
            mesh.faces
                .iter()
                .flat_map(|[a, b, c]| [*a as u32, *b as u32, *c as u32])
        })
        .collect::<Vec<u32>>();

    let objects = deserialized_scene
        .objects
        .iter()
        .map(|object| {
            let mesh_index = mesh_keys
                .iter()
                .enumerate()
                .find_map(|(i, key)| {
                    if key.as_str() == object.mesh.as_str() {
                        Some(i)
                    } else {
                        None
                    }
                })
                .unwrap();

            let vertex_start: usize = meshes[0..mesh_index]
                .iter()
                .map(|mesh| mesh.verts.len())
                .sum();

            let vertex_count = meshes[mesh_index].verts.len();

            let index_start: usize = meshes[0..mesh_index]
                .iter()
                .map(|mesh| mesh.faces.len())
                .sum::<usize>()
                * 3;

            let index_count = meshes[mesh_index].faces.len() * 3;

            let material = material_keys
                .iter()
                .enumerate()
                .find_map(|(i, key)| {
                    if key.as_str() == object.material.as_str() {
                        Some(i)
                    } else {
                        None
                    }
                })
                .unwrap();

            Object {
                transform: object.transform,
                vertex_start: vertex_start as u32,
                vertex_count: vertex_count as u32,
                index_start: index_start as u32,
                index_count: index_count as u32,
                material: material as u32,
            }
        })
        .collect::<Vec<_>>();

    let lights = deserialized_scene
        .lights
        .iter()
        .map(|light| match *light {
            deserialize::Light::Area {
                size,
                power,
                transform,
                color,
            } => Light {
                color,
                power,
                size,
                transform,
                transform_inverse: transform.try_inverse().unwrap(),
            },
        })
        .collect::<Vec<_>>();

    let perspective = Perspective3::new(
        deserialized_scene.camera.aspect_ratio,
        deserialized_scene.camera.fov,
        deserialized_scene.camera.znear,
        deserialized_scene.camera.zfar,
    )
    .into_inner();

    Ok(Scene {
        perspective,
        view: deserialized_scene.camera.transform,
        vertex_buffer,
        index_buffer,
        lights,
        materials,
        objects,
    })
}

fn configure_uniforms<T: ShaderType + encase::internal::WriteInto>(
    device: &wgpu::Device,
    uniforms: T,
) -> Result<wgpu::Buffer, Box<dyn std::error::Error>> {
    let mut encased_uniforms = UniformBuffer::new(Vec::new());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: uniforms.size().get(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    encased_uniforms.write(&uniforms)?;

    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(&encased_uniforms.into_inner());

    buffer.unmap();

    Ok(buffer)
}

fn configure_buffer<T: ShaderType + encase::internal::WriteInto>(
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    label: &str,
    data: T,
) -> Result<wgpu::Buffer, Box<dyn std::error::Error>> {
    let mut encased_data = StorageBuffer::new(Vec::new());

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data.size().get(),
        usage,
        mapped_at_creation: true,
    });

    encased_data.write(&data)?;

    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(&encased_data.into_inner());

    buffer.unmap();

    Ok(buffer)
}

fn build_acceleration_structures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    objects: &[Object],
) -> wgpu::TlasPackage {
    let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("tlas"),
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
            let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: object.vertex_count,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(object.index_count),
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            };

            let blas = device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: None,
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size_desc.clone()],
                },
            );

            let transform = object.transform.transpose().as_slice()[0..12]
                .try_into()
                .unwrap();

            tlas_package[i] = Some(wgpu::TlasInstance::new(&blas, transform, i as u32, 0xff));

            (size_desc, blas)
        })
        .collect::<Vec<_>>();

    let blas_build_entries = blases
        .iter()
        .zip(objects)
        .map(|((size_desc, blas), object)| {
            let geometries = wgpu::BlasTriangleGeometry {
                size: size_desc,
                vertex_buffer,
                first_vertex: object.vertex_start,
                vertex_stride: Vertex::SHADER_SIZE.get(),
                index_buffer: Some(index_buffer),
                // this seems to be an offset in bytes?
                index_buffer_offset: Some(object.index_start as u64 * 4),
                transform_buffer: None,
                transform_buffer_offset: None,
            };

            wgpu::BlasBuildEntry {
                blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![geometries]),
            }
        })
        .collect::<Vec<_>>();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    encoder.build_acceleration_structures(blas_build_entries.iter(), iter::once(&tlas_package));

    queue.submit(Some(encoder.finish()));

    tlas_package
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
