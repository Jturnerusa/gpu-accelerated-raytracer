#![allow(dead_code)]

mod scene;
mod state;

use std::{fs::File, path::PathBuf};

use clap::Parser;
use scene::Scene;
use state::{State, WindowDesc};

const WORKGROUP_SIZE_X: usize = 8;
const WORKGROUP_SIZE_Y: usize = 8;
const WORKGROUP_SIZE_Z: usize = 1;

#[derive(Debug)]
struct Error {
    message: String,
    source: Option<Box<dyn std::error::Error>>,
}

#[derive(Clone, Debug, clap::Parser)]
struct Args {
    #[arg(long)]
    width: u32,
    #[arg(long)]
    height: u32,
    #[arg(long)]
    seed: u32,
    #[arg(long)]
    scene: PathBuf,
    #[arg(long)]
    chunk_size: u32,
    #[arg(long)]
    samples: u32,
    #[arg(long)]
    bounces: u32,
    #[arg(long)]
    gui: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ray tracer error: {}", self.message)
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

fn print_error_chain(top: &dyn std::error::Error) {
    eprintln!("{top}");

    let mut error = top.source();
    let mut n = 1usize;

    while let Some(e) = &error {
        let indent = " ".repeat(n);

        eprintln!("{indent}caused by: {e}");

        error = e.source();
        n += 1;
    }
}

#[tokio::main]
async fn main() {
    match run().await {
        Ok(_) => (),
        Err(e) => {
            print_error_chain(&e);
            std::process::exit(1);
        }
    }
}

async fn run() -> Result<(), Error> {
    let args = Args::parse();

    let file = File::open(args.scene.as_path()).map_err(|e| Error {
        message: format!(
            "failed to load scene file {}",
            args.scene.as_path().to_str().unwrap()
        ),
        source: Some(Box::new(e)),
    })?;

    let data = unsafe {
        memmap2::Mmap::map(&file).map_err(|e| Error {
            message: format!(
                "failed to mmap scene file {}",
                args.scene.as_path().to_str().unwrap()
            ),
            source: Some(Box::new(e)),
        })?
    };

    match args.scene.as_path().extension().and_then(|s| s.to_str()) {
        Some("glb") => {
            let glb = gltf::Glb::from_slice(&data).map_err(|e| Error {
                message: "failed to load scene".to_string(),
                source: Some(Box::new(e)),
            })?;

            let bin = &glb.bin.ok_or(Error {
                message: "no binary data found in glb file".to_string(),
                source: None,
            })?;

            let scene =
                Box::new(
                    crate::scene::gltf::Scene::new(&data, bin).map_err(|e| Error {
                        message: "failed to open scene file".to_string(),
                        source: Some(Box::new(e)),
                    })?,
                );

            if args.gui {
                run_with_gui(&args, &*scene).await?;
            } else {
                run_headless(&args, &*scene).await?;
            }

            Ok(())
        }
        Some("gltf") => {
            let bin_path = args.scene.with_extension("bin");

            let bin_file = File::open(bin_path.as_path()).map_err(|e| Error {
                message: format!(
                    "failed to open gltf binary data file {}",
                    bin_path.to_str().unwrap()
                ),
                source: Some(Box::new(e)),
            })?;

            let bin = unsafe {
                memmap2::Mmap::map(&bin_file).map_err(|e| Error {
                    message: format!(
                        "failed to mmap scene file {}",
                        bin_path.as_path().to_str().unwrap()
                    ),
                    source: Some(Box::new(e)),
                })?
            };

            let scene =
                Box::new(
                    crate::scene::gltf::Scene::new(&data, &bin).map_err(|e| Error {
                        message: format!(
                            "failed to open scene file {}",
                            args.scene.as_path().to_str().unwrap()
                        ),
                        source: Some(Box::new(e)),
                    })?,
                );

            if args.gui {
                run_with_gui(&args, &*scene).await?;
            } else {
                run_headless(&args, &*scene).await?;
            }

            Ok(())
        }
        _ => Err(Error {
            message: "failed to recognize file format".to_string(),
            source: None,
        })?,
    }
}

async fn run_with_gui(args: &Args, scene: &dyn Scene) -> Result<(), Error> {
    let sdl2_context = sdl2::init().map_err(|e| Error {
        message: "failed to load sdl2".to_string(),
        source: Some(e.into()),
    })?;

    let sdl2_video = sdl2_context.video().map_err(|e| Error {
        message: "failed to load sdl2 video subsystem".to_string(),
        source: Some(e.into()),
    })?;

    let window = sdl2_video
        .window("raytracer", args.width, args.height)
        .position_centered()
        .resizable()
        .build()
        .map_err(|e| Error {
            message: "failed to load sdl2 window".to_string(),
            source: Some(e.into()),
        })?;

    let mut events = sdl2_context.event_pump().map_err(|e| Error {
        message: "failed to load sdl2 event pump".to_string(),
        source: Some(e.into()),
    })?;

    let mut state = State::new(Some(WindowDesc {
        width: args.width,
        height: args.height,
        window: &window,
    }))
    .await
    .map_err(|e| Error {
        message: "failed to setup state".to_string(),
        source: Some(Box::new(e)),
    })?;

    state
        .load_scene(
            args.width,
            args.height,
            args.seed,
            args.samples,
            args.bounces,
            args.chunk_size,
            scene,
        )
        .map_err(|e| Error {
            message: "failed to upload scene to gpu".to_string(),
            source: Some(Box::new(e)),
        })?;

    while !state.is_finished() {
        state.process_chunk().map_err(|e| Error {
            message: "failed to process chunk".to_string(),
            source: Some(Box::new(e)),
        })?;

        state.render().map_err(|e| Error {
            message: "failed to render frame".to_string(),
            source: Some(Box::new(e)),
        })?;

        for event in events.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. } => return Ok(()),
                _ => continue,
            }
        }

        state.wait();
    }

    'main: loop {
        for event in events.poll_iter() {
            match event {
                sdl2::event::Event::Quit { .. }
                | sdl2::event::Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::ESCAPE),
                    ..
                } => break 'main,
                _ => continue,
            }
        }
    }

    write_output(args, &mut state)?;

    Ok(())
}

async fn run_headless(args: &Args, scene: &dyn Scene) -> Result<(), Error> {
    let mut state = State::new(None).await.map_err(|e| Error {
        message: "failed to setup state".to_string(),
        source: Some(Box::new(e)),
    })?;

    state
        .load_scene(
            args.width,
            args.height,
            args.seed,
            args.samples,
            args.bounces,
            args.chunk_size,
            scene,
        )
        .map_err(|e| Error {
            message: "failed to upload scene to gpu".to_string(),
            source: Some(Box::new(e)),
        })?;

    while !state.is_finished() {
        state.process_chunk().map_err(|e| Error {
            message: "failed to process chunk".to_string(),
            source: Some(Box::new(e)),
        })?;

        state.wait();
    }

    write_output(args, &mut state)?;

    Ok(())
}

fn write_output<W>(args: &Args, state: &mut State<W>) -> Result<(), Error> {
    if let Some(output) = args.output.as_ref() {
        let mut float_pixel_data = Vec::new();
        let mut rgba_pixel_data = Vec::new();

        state
            .download_frame(&mut float_pixel_data)
            .map_err(|e| Error {
                message: "failed to download frame from gpu".to_string(),
                source: Some(Box::new(e)),
            })?;

        rgba32float_to_rgba8888(float_pixel_data.as_slice(), &mut rgba_pixel_data);

        image::save_buffer_with_format(
            output,
            rgba_pixel_data.as_slice(),
            args.width,
            args.height,
            image::ColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .map_err(|e| Error {
            message: "failed to save output".to_string(),
            source: Some(Box::new(e)),
        })?;
    }
    Ok(())
}

fn rgba32float_to_rgba8888(floats: &[u8], output: &mut Vec<u8>) {
    for [r, g, b, _] in bytemuck::cast_slice::<u8, f32>(floats)
        .chunks(4)
        .map(|chunk| <[f32; 4]>::try_from(chunk).unwrap())
    {
        let r = (r * 255.0) as u8;
        let g = (g * 255.0) as u8;
        let b = (b * 255.0) as u8;

        output.extend_from_slice(&[r, g, b]);
    }
}

fn find_best_chunk_size(max: u32, width: u32, wgx: u32, wgy: u32) -> u32 {
    for x in (1..=max).rev() {
        if width % x == 0 && x % wgx == 0 && x % wgy == 0 {
            return x;
        }
    }
    1
}
