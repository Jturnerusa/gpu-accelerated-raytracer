#![allow(dead_code)]

mod scene;
mod state;

use std::{
    fs::File,
    io::{self},
    path::PathBuf,
};

use clap::Parser;
use scene::Scene;
use state::{State, WindowDesc};

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
    samples: u32,
    #[arg(long)]
    bounces: u32,
    #[arg(long)]
    gui: bool,
}

#[tokio::main]
async fn main() {
    match run().await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let file = File::open(args.scene.as_path())?;
    let data = unsafe { memmap2::Mmap::map(&file)? };

    match args.scene.as_path().extension().and_then(|s| s.to_str()) {
        Some("glb") => {
            let glb = gltf::Glb::from_slice(&data)?;
            let bin = &glb
                .bin
                .ok_or("no binary data found in glb file".to_string())?;

            let scene = Box::new(crate::scene::gltf::Scene::new(&glb.json, bin)?);

            if args.gui {
                run_with_gui(&args, &*scene).await?;
            } else {
                run_headless(&args, &*scene).await?;
            }

            Ok(())
        }
        Some("gltf") => {
            let data_file = File::open(args.scene.with_extension("bin"))?;
            let bin = unsafe { memmap2::Mmap::map(&data_file)? };

            let scene = Box::new(crate::scene::gltf::Scene::new(&data, &bin)?);

            if args.gui {
                run_with_gui(&args, &*scene).await?;
            } else {
                run_headless(&args, &*scene).await?;
            }

            Ok(())
        }
        _ => Err("failed to recognize file format".to_string())?,
    }
}

async fn run_with_gui(args: &Args, scene: &dyn Scene) -> Result<(), Box<dyn std::error::Error>> {
    let sdl2_context = sdl2::init()?;
    let sdl2_video = sdl2_context.video()?;
    let window = sdl2_video
        .window(
            "raytracer",
            args.width,
            (args.width as f32 / args.aspect_ratio) as u32,
        )
        .position_centered()
        .resizable()
        .build()?;

    let mut events = sdl2_context.event_pump()?;

    let mut state = State::new(Some(WindowDesc {
        width: args.width,
        height: (args.width as f32 / args.aspect_ratio) as u32,
        window: &window,
    }))
    .await?;

    state.load_scene(
        args.width,
        (args.width as f32 / args.aspect_ratio) as u32,
        args.seed,
        args.samples,
        args.bounces,
        args.chunk_size,
        scene,
    )?;

    while !state.is_finished() {
        state.process_chunk()?;
        state.render()?;

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

async fn run_headless(args: &Args, scene: &dyn Scene) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = State::new(None).await?;

    state.load_scene(
        args.width,
        (args.width as f32 / args.aspect_ratio) as u32,
        args.seed,
        args.samples,
        args.bounces,
        args.chunk_size,
        scene,
    )?;

    while !state.is_finished() {
        state.process_chunk()?;
        state.wait();
    }

    write_output(args, &mut state)?;

    Ok(())
}

fn write_output<W>(args: &Args, state: &mut State<W>) -> Result<(), Box<dyn std::error::Error>> {
    let mut float_pixel_data = Vec::new();
    let mut rgba_pixel_data = Vec::new();

    state.download_frame(&mut float_pixel_data)?;

    rgba32float_to_rgba8888(float_pixel_data.as_slice(), &mut rgba_pixel_data);

    let mut encoder = png::Encoder::new(
        io::stdout(),
        args.width,
        (args.width as f32 / args.aspect_ratio) as u32,
    );

    encoder.set_color(png::ColorType::Rgb);

    let mut writer = encoder.write_header()?;

    writer.write_image_data(rgba_pixel_data.as_slice())?;

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
