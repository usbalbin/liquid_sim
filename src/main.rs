extern crate glium;

use glium::Surface;
use std::f32::consts::FRAC_1_SQRT_2;

const DIRS: [(i32, i32); 8] = [
    ( 0,  1),
    ( 1,  0),
    ( 0, -1),
    (-1,  0),

    (-1,  1),
    ( 1,  1),
    ( 1, -1),
    (-1, -1),
];

const DIAGONAL_FACTOR: f32 = 0.0;

const NORMALIZED_DIRS: [(f32, f32); 8] = [
    ( 0.0,  1.0),
    ( 1.0,  0.0),
    ( 0.0, -1.0),
    (-1.0,  0.0),

    (-FRAC_1_SQRT_2 * DIAGONAL_FACTOR, FRAC_1_SQRT_2 * DIAGONAL_FACTOR),
    ( FRAC_1_SQRT_2 * DIAGONAL_FACTOR, FRAC_1_SQRT_2 * DIAGONAL_FACTOR),
    ( FRAC_1_SQRT_2 * DIAGONAL_FACTOR, -FRAC_1_SQRT_2 * DIAGONAL_FACTOR),
    (-FRAC_1_SQRT_2 * DIAGONAL_FACTOR, -FRAC_1_SQRT_2 * DIAGONAL_FACTOR),
];




fn setup() -> (glium::glutin::EventsLoop, glium::Display) {
    let events_loop = glium::glutin::EventsLoop::new();

    let window = glium::glutin::WindowBuilder::new()
        .with_dimensions(1024, 768)
        .with_title("Liquid Sim");

    let context = glium::glutin::ContextBuilder::new();

    let display = glium::Display::new(window, context, &events_loop).unwrap();

    (events_loop, display)
}

fn main() {
    let width = 500;
    let height = 500;
    let mut pressure: Vec<Vec<f32>> = (0..height).map(|y| (0..width).map(|x|{
        let center = (width as f32 / 2.0, height as f32 / 2.0);
        let value =  max(0.0, (128.0 - length(sub(&center, &(x as f32, y as f32)))).abs());
        100.0 + value * 1.0
    }).collect()).collect();

    pressure[height / 2][width / 2] = 24.5;

    let mut flow = vec![vec![(0.0, 0.0); width]; height];
    let mut flow2 = vec![vec![(0.0, 0.0); width]; height];

    assert_eq!(pressure.len(), flow.len());
    assert_eq!(pressure[0].len(), flow[0].len());



    let (mut events_loop, display) = setup();



    let mut close = false;
    while !close {

        let texture = update(&mut pressure, &mut flow, &mut flow2,&display);
        std::mem::swap(&mut flow, &mut flow2);

        let target = display.draw();
        texture.as_surface().fill(&target, glium::uniforms::MagnifySamplerFilter::Linear);
        target.finish().unwrap();

        events_loop.poll_events(|ev| {
            match ev {
                glium::glutin::Event::WindowEvent { event, .. } => match event {
                    glium::glutin::WindowEvent::Closed => close = true,
                    _ => (),
                },
                _ => (),
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    println!("Hello, world!");
}


fn update(pressure: &mut Vec<Vec<f32>>, flow: &mut Vec<Vec<(f32, f32)>>, old_flow: &mut Vec<Vec<(f32, f32)>>, display: &glium::Display) -> glium::texture::Texture2d {
    update_flow(pressure, flow);
    update_pressure(pressure, flow);
    update_heat(flow, old_flow);

    let mut sum = 0.0;
    let pixels:Vec<Vec<(u8, u8, u8)>> = pressure.iter().zip(flow.iter())
        .map(|(p_row, f_row)| p_row.iter().zip(f_row.iter())
            .map(|(p, f)| {

        sum += p;
        let r = to_u8(128.0 + f.0);
        let g = to_u8(128.0 + f.1);
        let b = to_u8(*p);

        (0x11, 0x11, b)//(r, g, b)
    }).collect()).collect();
    println!("Total pressure: {} (should be constant)", sum);
    glium::texture::Texture2d::new(display, pixels).unwrap()
}

fn update_flow(pressure: &Vec<Vec<f32>>, flow: &mut Vec<Vec<(f32, f32)>>) {
    let width = pressure[0].len();
    let height = pressure.len();

    for y in 0..height {
        for x in 0..width {
            let pos = (x, y);
            flow_kernel(pressure, mut_flow(flow, pos), pos);
        }
    }
}
fn update_heat(old_flow: &Vec<Vec<(f32, f32)>>, flow: &mut Vec<Vec<(f32, f32)>>) {
    let width = flow[0].len();
    let height = flow.len();

    for y in 0..height {
        for x in 0..width {
            let pos = (x, y);
            heat_kernel(old_flow, mut_flow(flow, pos), pos);
        }
    }
}

fn update_pressure(pressure: &mut Vec<Vec<f32>>, flow: &Vec<Vec<(f32, f32)>>) {
    let width = pressure[0].len();
    let height = pressure.len();

    for y in 0..height {
        for x in 0..width {
            let pos = (x, y);
            pressure_kernel(mut_pressure(pressure, pos), flow, pos);
        }
    }
}

fn flow_kernel(pressure: &Vec<Vec<f32>>, flow: &mut(f32, f32), pos: (usize, usize)) {
    const GRAVITY: (f32, f32) = (0.0, 0.0);
    const MAGIC: f32 = 0.4;



    let this_pressure = get_pressure(pressure, &pos, &(0, 0));

    let mut force = GRAVITY;
    for (dir, normalized_dir) in DIRS.iter().zip(NORMALIZED_DIRS.iter()) {
        let f = get_pressure(pressure, &pos, dir) - this_pressure;

        let f = mul(&normalized_dir, f);
        force = add(&force, &f);
    }

    let delta_flow = mul(&force, MAGIC);
    *flow = add(flow, &delta_flow);
}

fn pressure_kernel(pressure: &mut f32, flow: &Vec<Vec<(f32, f32)>>, pos: (usize, usize)) {
    let mut delta_pressure = 0.0;
    for (dir, normalized_dir) in DIRS.iter().zip(NORMALIZED_DIRS.iter()) {
        let f = get_flow(flow, &pos, dir);

        delta_pressure += dot( &f, &normalized_dir) * (1.0 + DIAGONAL_FACTOR);//TODO: adjust this
    }

    *pressure += delta_pressure;
}

fn heat_kernel(old_flow: &Vec<Vec<(f32, f32)>>, flow: &mut(f32, f32), pos: (usize, usize)) {
    const FRICTION_COEF: f32 = 0.0001;

    let this_flow = get_flow(old_flow, &pos, &(0, 0));

    let mut delta_flow = (0.0, 0.0);
    for dir in DIRS.iter() {
        let f = sub(&get_flow(old_flow, &pos, dir), &this_flow);
        delta_flow = add(&delta_flow, &f);
    }

    *flow = add(&this_flow, &mul(&delta_flow, FRICTION_COEF));
}

///Requires at least pos to be in bounds
fn get_flow(flow: &Vec<Vec<(f32, f32)>>, pos: &(usize, usize), delta: &(i32, i32)) -> (f32, f32) {
    let x = pos.0 as isize + delta.0 as isize;
    let y = pos.1 as isize + delta.1 as isize;

    let width = flow[0].len();
    let height = flow.len();

    if x < 0 || y < 0 {
        return (0.0, 0.0)
    }
    let x = x as usize;
    let y = y as usize;
    if x >= width || y as usize >= height {
        return (0.0, 0.0)
    }

    flow[y][x]
}

///Requires at least pos to be in bounds
fn get_pressure(pressure: &Vec<Vec<f32>>, pos: &(usize, usize), delta: &(i32, i32)) -> f32 {
    let x = pos.0 as isize + delta.0 as isize;
    let y = pos.1 as isize + delta.1 as isize;

    let width = pressure[0].len();
    let height = pressure.len();

    if x < 0 || y < 0 {
        return pressure[pos.0][pos.1]
    }
    let x = x as usize;
    let y = y as usize;
    if x >= width || y as usize >= height {
        return pressure[pos.0][pos.1]
    }

    pressure[y][x]
}

fn mut_flow(flow: &mut Vec<Vec<(f32, f32)>>, pos: (usize, usize)) -> &mut (f32, f32) {
    &mut flow[pos.1][pos.0]
}

fn mut_pressure(pressure: &mut Vec<Vec<f32>>, pos: (usize, usize)) -> &mut f32 {
    &mut pressure[pos.1][pos.0]
}

fn dot(a: &(f32, f32), b: &(f32, f32)) -> f32 {
    a.0 * b.0 +
        a.1 * b.1
}

fn add(a: &(f32, f32), b: &(f32, f32)) -> (f32, f32) {
    (
        a.0 + b.0,
        a.1 + b.1
    )
}

fn sub(a: &(f32, f32), b: &(f32, f32)) -> (f32, f32) {
    (
        a.0 - b.0,
        a.1 - b.1
    )
}


fn neg(a: &(f32, f32)) -> (f32, f32) {
    (
        -a.0,
        -a.1
    )
}

fn mul(a: &(f32, f32), b: f32) -> (f32, f32) {
    (
        a.0 * b,
        a.1 * b
    )
}

fn length(f: (f32, f32)) -> f32 {
    dot(&f, &f).sqrt()
}

fn max(a: f32, b: f32) -> f32 {
    if a < b {
        b
    } else {
        a
    }
}

fn to_u8(f: f32) -> u8 {
    if f < 0.0 {
        0x00
    } else if f > 255.0 {
        0xFF
    } else {
        f as u8
    }
}