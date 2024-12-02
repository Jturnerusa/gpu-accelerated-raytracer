const F32_MAX: f32 = 3.40282347e+38f;
const F32_EPSILON: f32 = 1.1920929e-7;

@group(0) @binding(0)
var<uniform> UNIFORMS: Uniforms;

@group(0) @binding(1)
var<storage, read> VERTEX_BUFFER: array<Vertex>;

@group(0) @binding(2)
var<storage, read> INDEX_BUFFER: array<u32>;

@group(0) @binding(3)
var<storage, read> MATERIALS_BUFFER: array<Material>;

@group(0) @binding(4)
var<storage, read> OBJECTS: array<Object>;

@group(0) @binding(5)
var<storage, read> LIGHTS: array<Light>;

@group(0) @binding(6)
var TLAS: acceleration_structure;

@group(0) @binding(7)
var SAMPLES: texture_storage_2d<rgba32float, read_write>;

var<private> RNG: u32 = 0;

var<private> VERTS: array<vec2f, 6> = array (
  vec2f(-1.0, 1.0),
  vec2f(-1.0, -1.0),
  vec2f(1.0, 1.0),
  vec2f(1.0, 1.0),
  vec2f(-1.0, -1.0),
  vec2f(1.0, -1.0)
);

struct Uniforms {
    view: mat4x4f,
    view_inverse: mat4x4f,    
    perspective: mat4x4f,
    perspective_inverse: mat4x4f,
    width: u32,
    height: u32,
    objects: u32,
    lights: u32,
    chunk_size: u32,
    bounces: u32,
    seed: u32,
    current_chunk: u32,
    samples_per_pass: u32,
    passes: u32
}

struct Vertex {
    p: vec3f,
    n: vec3f
}

struct Light {
    color: vec4f,
    power: f32,
    size: f32,
    transform: mat4x4f,
    transform_inverse: mat4x4f
}

struct Material {
    metallic: f32,
    roughness: f32,
    color: vec4f
}

struct Object {
    transform: mat4x4f,
    vertex_start: u32,
    vertext_count: u32,
    index_start: u32,
    index_count: u32,
    material: u32
}

struct LightHit {
    power: f32,
    color: vec4f
}

struct Quad {
    q: vec3f,
    u: vec3f,
    v: vec3f
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct Tri {
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
}

fn hash(input: u32) -> u32 {
    var k = input;
    k *= 0xcc9e2d51u;
    k = (k << 15) | (k >> 17);
    k *= 0x1b873593u;
    return k;
}

// generate random float from 0.0 to 1.0
fn rand() -> f32 {
    RNG = hash(RNG);
    return bitcast<f32>(0x3f800000u | (RNG >> 9u)) - 1.0;
}

fn random_unit_vec() -> vec3f {
    var r = vec3f();
    loop {
      r = vec3f(rand() - rand(), rand() - rand(), rand() - rand());
      if dot(r, r) <= 1.0 {
        break;
      } else {
        continue;
      }
    }
    return normalize(r);
}

fn quad_ray_intersects(quad: Quad,
                       ray: Ray,
                       min: f32,
                       max: f32,
                       hit: ptr<function, f32>) -> bool
{
    let n = cross(quad.u, quad.v);
    let normal = normalize(n);
    let d = dot(normal, quad.q);
    let w = n / dot(n, n);
    let denom = dot(normal, ray.direction);

    if abs(denom) < 1e-8 {
        return false;
    }

    let t = (d - dot(normal, ray.origin)) / denom;

    if t >= max || t <= min {
        return false;
    }

    let intersection = ray_at(ray, t);
    let p = intersection - quad.q;
    let alpha = dot(w, cross(p, quad.v));
    let beta = dot(w, cross(quad.u, p));

    if !(alpha >= 0.0 && alpha <= 1.0) || !(beta >= 0.0 && beta <= 1.0) {
        return false;
    } else {
        *hit = t;
        return true;
    }
}

fn hit_lights(ray: Ray, light_hit: ptr<function, LightHit>) -> bool {
    var found_hit = false;
    var closest = F32_MAX;

    for (var i = 0u; i < UNIFORMS.lights; i++) {
        let light = LIGHTS[i];

        let light_ray = Ray(
            (light.transform_inverse * vec4f(ray.origin.xyz, 1.0)).xyz,
            normalize(light.transform_inverse * vec4f(ray.direction.xyz, 0.0)).xyz
        );

        //
        // p2----
        // |    |
        // |    |
        // p0---p1

        let p0 = vec3f(
            -(light.size / 2.0),
            -(light.size / 2.0),
            0.0
        );

        let p1 = vec3f(
            light.size / 2.0,
            -(light.size / 2.0),
            0.0
        );

        let p2 = vec3f(
            -(light.size / 2.0),
            light.size / 2.0,
            0.0
        );

        let q = p0;
        let u = p2 - p0;
        let v = p1 - p0;

        let quad = Quad(q, u, v);

        if quad_ray_intersects(quad, light_ray, F32_EPSILON, closest, &closest) {
            found_hit = true;
            (*light_hit).power = light.power;
            (*light_hit).color = light.color;
        }
    }

    return found_hit;
}

fn get_intersection_tri(object: Object, intersection: RayIntersection) -> Tri {
   // offset into index buffer relative to the object hit.
   let x = intersection.primitive_index * 3;

   // offset into the index buffer to read vertices from.
   let first_index_index = x + object.index_start;

   // indices from the index buffer
   let i0 = INDEX_BUFFER[first_index_index];
   let i1 = INDEX_BUFFER[first_index_index + 1u];
   let i2 = INDEX_BUFFER[first_index_index + 2u];

   // vertex indices, which is an index buffers index plus
   // the objects vertex start index;
   let vi0 = object.vertex_start + i0;
   let vi1 = object.vertex_start + i1;
   let vi2 = object.vertex_start + i2;   

   // vertices, which are of type Vertex and not
   // a vec3f.
   let v0 = VERTEX_BUFFER[vi0];
   let v1 = VERTEX_BUFFER[vi1];
   let v2 = VERTEX_BUFFER[vi2];   

   return Tri(v0, v1, v2);
}

fn get_interpolated_normal(tri: Tri, intersection: RayIntersection) -> vec3f {
   let v0v1 = tri.v1.p - tri.v0.p;
   let v0v2 = tri.v2.p - tri.v0.p;
   let n = cross(v0v1, v0v2);

   return normalize(n);
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.origin + t * ray.direction;
}

fn cast_ray(pixel: vec2f) -> Ray {
    let clip = pixel
               / vec2f(f32(UNIFORMS.width), f32(UNIFORMS.height))
               * 2.0
               - 1.0;
    let camera = UNIFORMS.perspective_inverse * vec4f(clip.x, -clip.y, 0.0, 1.0);
    let direction = UNIFORMS.view * vec4f(normalize(camera).xyz, 0.0);
    let origin = UNIFORMS.view * vec4f(0.0, 0.0, 0.0, 1.0);

    return Ray(origin.xyz, normalize(direction.xyz));

}

fn ray_query(ray: Ray, min: f32, max: f32) -> RayIntersection {
    var rq: ray_query;  

    rayQueryInitialize(&rq, TLAS, RayDesc(0, 0xFFu, min, max, ray.origin, ray.direction));
    rayQueryProceed(&rq);

    return rayQueryGetCommittedIntersection(&rq);
}

fn pixel_color(pixel: vec2f) -> vec4f {
    var ray = cast_ray(pixel);
    var radiance = vec4f();
    var attenuation = vec4f(1.0, 1.0, 1.0, 0.0);
    var bounces = UNIFORMS.bounces;
    var intersection = ray_query(ray, 0.001, F32_MAX);
    
    while intersection.kind != RAY_QUERY_INTERSECTION_NONE && bounces > 0u {
        let object = OBJECTS[intersection.instance_custom_index];
        let material = MATERIALS_BUFFER[object.material];
        let tri = get_intersection_tri(object, intersection);
        let p = ray_at(ray, intersection.t);
        var normal = get_interpolated_normal(tri, intersection);
        normal = normalize(object.transform * vec4f(normal.xyz, 0.0)).xyz;
        if !(dot(ray.direction, normal) < 0.0) {
            normal = -normal;
        }

        if material.metallic > 0.0 {
            let reflected = reflect(ray.direction, normal) + random_unit_vec() * material.roughness;

            ray = Ray(p, reflected);
            attenuation *= material.color;
        } else {
            let diffused = normalize(normal + random_unit_vec());

            ray = Ray(p, diffused);
            attenuation *= material.color * material.roughness;
        }

        bounces -= 1u;
        intersection = ray_query(ray, 0.001, F32_MAX);
    }

    var light_hit = LightHit();
    if hit_lights(ray, &light_hit) {
        radiance = light_hit.color * light_hit.power;
    }

    return radiance * attenuation;
}

@vertex
fn vs_main(
    @builtin(vertex_index) i: u32,
) -> @builtin(position) vec4f {
    return vec4f(VERTS[i], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4<f32> {
    let color = textureLoad(SAMPLES, vec2u(pos.xy));

    let samples = UNIFORMS.passes * UNIFORMS.samples_per_pass;

    return color / f32(samples);
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    RNG = (gid.x + 1) * (gid.y + 1) * (UNIFORMS.current_chunk + 1) * UNIFORMS.passes * UNIFORMS.seed;

    let chunk_x = UNIFORMS.current_chunk % (UNIFORMS.width / UNIFORMS.chunk_size);
    let chunk_y = UNIFORMS.current_chunk / (UNIFORMS.width / UNIFORMS.chunk_size);
    let pixel_x = chunk_x * UNIFORMS.chunk_size + gid.x; 
    let pixel_y = chunk_y * UNIFORMS.chunk_size + gid.y;
    let pixel = vec2u(pixel_x, pixel_y);

    if pixel.x > UNIFORMS.width || pixel.y > UNIFORMS.height {
        return;
    }

    var color = vec4f();

    for (var i = 0u; i < UNIFORMS.samples_per_pass; i++) {
        color += pixel_color(vec2f(pixel) + vec2f(rand(), rand()));
    }

    let sample = textureLoad(SAMPLES, pixel);    

    textureStore(SAMPLES, pixel, sample + color);
}
