const F32_MAX: f32 = 3.40282347e+38f;
const F32_EPSILON: f32 = 1.1920929e-7;
const PI: f32 = 3.1415926;
const INV_PI: f32 = 0.3183098;

@group(0) @binding(0)
var<uniform> UNIFORMS: Uniforms;

@group(0) @binding(1)
var<storage, read> OBJECT_BUFFER: array<Object>;

@group(0) @binding(2)
var<storage, read> MESH_BUFFER: array<Mesh>;

@group(0) @binding(3)
var<storage, read> PRIMITIVE_BUFFER: array<Primitive>;

@group(0) @binding(4)
var<storage, read> VERTEX_BUFFER: array<Vertex>;

@group(0) @binding(5)
var<storage, read> INDEX_BUFFER: array<u32>;

@group(0) @binding(6)
var<storage, read> MATERIAL_BUFFER: array<Material>;

@group(0) @binding(7)
var<storage, read> LIGHT_BUFFER: array<Light>;

@group(0) @binding(8)
var<storage, read> TEXTURE_DESCRIPTOR_BUFFER: array<TextureDesc>;

@group(0) @binding(9)
var TLAS: acceleration_structure;

@group(0) @binding(10)
var SAMPLES: texture_storage_2d<rgba32float, read_write>;

@group(0) @binding(11)
var TEXTURES: binding_array<texture_2d<f32>>;

@group(0) @binding(12)
var SAMPLER: sampler;

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
    perspective: mat4x4f,
    width: u32,
    height: u32,
    objects: u32,
    lights: u32,
    chunk_size: u32,
    bounces: u32,
    seed: u32,
    current_chunk: u32,
    samples: u32
}

struct Object {
    transform: mat4x4f,
    mesh: u32
}

struct Mesh {
    primitive_start: u32,
    primitive_count: u32,    
}

struct Primitive {
    vertex_start: u32,
    vertex_count: u32,
    index_start: u32,
    index_count: u32,
    material: u32
}

struct Vertex {
    pos: vec3f,
    normal: vec3f,
    uv: vec2f,
}

struct Material {
    metallic: f32,
    roughness: f32,
    emission: f32,
    ior: f32,
    texture: u32,
    has_texture: u32,
    color: vec4f
}

struct Light {
    transform: mat4x4f,
    color: vec4f,
    power: f32
}

struct TextureDesc {
    width: u32,
    height: u32
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

struct Hit {
    material: u32,
    normal: vec3f,
    pos: vec3f,
    uv: vec2f,
}

struct LightSample {
    radiance: vec4f,
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

fn square(x: f32) -> f32 {
    return x * x;
}

fn random_light() -> Light {
    let i = u32(rand() * f32(UNIFORMS.lights));

    return LIGHT_BUFFER[i];
}

fn light_is_blocked(p: vec3f, light: Light) -> bool {
    let light_position = (light.transform * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;

    let ray = Ray(p, normalize(light_position - p));

    let intersection = ray_query(ray, 0.0, distance(light_position, p));

    if intersection.kind == RAY_QUERY_INTERSECTION_NONE {
        return false;
    } else {
        return true;
    }
}

fn sample_light(p: vec3f, light: Light) -> vec4f {
    let light_position = (light.transform * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;

    return light.color / sqrt(distance(light_position, p));
}

fn sample_unit_disk(u: vec2f) -> vec2f {
    let r = sqrt(u.x);
    let theta = 2 * PI * u.y;
    let x = r * cos(theta);
    let y = r * sin(theta);
    return vec2f(x, y);
}

fn sample_cosine_hemisphere(u: vec2f) -> vec3f {
    let d = sample_unit_disk(u);
    let z = sqrt(1 - square(d.x) - square(d.y));
    return vec3f(d.x, d.y, z);
}

fn cosine_hemisphere_pdf(cos_theta: f32) -> f32 {
    return cos_theta * INV_PI;
}

fn diffuse_brdf(normal: vec3f,
                direction: vec3f,
                in_color: vec4f,
                scattered: ptr<function, vec3f>,
                out_color: ptr<function, vec4f>,
                pdf: ptr<function, f32>)
{
    (*scattered) = sample_cosine_hemisphere(vec2f(rand(), rand()));
    (*out_color) = in_color / PI;
    (*pdf) = cosine_hemisphere_pdf(abs(direction.z));
    
    if direction.z < 0.0 {
        (*scattered).z *= -1.0;
    }
}

fn metal_brdf(normal: vec3f,
              direction: vec3f,
              in_color: vec4f,
              roughness: f32,
              scattered: ptr<function, vec3f>,
              out_color: ptr<function, vec4f>,
              pdf: ptr<function, f32>)
{
    (*scattered) = reflect(direction, normal);
    (*out_color) = in_color;
    (*pdf) = 1.0;
}

fn glass_brdf(normal: vec3f,
              direction: vec3f,
              in_color: vec4f,
              ior: f32,
              scattered: ptr<function, vec3f>,
              color: ptr<function, vec4f>,
              pdf: ptr<function, f32>)
{
    let uv = normalize(direction);
    let cos_theta = min(-dot(uv, normal), 1.0);
    let out_perp = ior * (uv + cos_theta * normal);
    let out_parallel = -(1.0 - sqrt(abs(dot(out_perp, out_perp))) * normal);

    (*scattered) = out_perp + out_parallel;   
    (*color) = in_color;
    (*pdf) = 1.0;
}

fn get_intersection_data(intersection: RayIntersection) -> Hit {
    let object = OBJECT_BUFFER[intersection.instance_custom_index];
    let mesh = MESH_BUFFER[object.mesh];
    let primitive = PRIMITIVE_BUFFER[mesh.primitive_start + intersection.geometry_index];

    // offset into index buffer relative to the intersected primitive 
    let x = intersection.primitive_index * 3;

    // offset into the index buffer
    let first_index_index = x + primitive.index_start;

    // indices from the index buffer
    let i0 = INDEX_BUFFER[first_index_index];
    let i1 = INDEX_BUFFER[first_index_index + 1u];
    let i2 = INDEX_BUFFER[first_index_index + 2u];

    // vertex indices
    let vi0 = i0 + primitive.vertex_start;
    let vi1 = i1 + primitive.vertex_start;
    let vi2 = i2 + primitive.vertex_start;

    // vertices
    let v0 = VERTEX_BUFFER[vi0];
    let v1 = VERTEX_BUFFER[vi1];
    let v2 = VERTEX_BUFFER[vi2];

    let bary = vec3f(1.0 - intersection.barycentrics.x - intersection.barycentrics.y,
                     intersection.barycentrics);

    let normal = v0.normal * bary.x + v1.normal * bary.y + v2.normal * bary.z;
    let pos = v0.pos * bary.x + v1.pos * bary.y + v2.pos * bary.z;
    var uv = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;

    return Hit(primitive.material, normal, pos, uv);
}

fn ray_at(ray: Ray, t: f32) -> vec3f {
    return ray.origin + t * ray.direction;
}

fn cast_ray(pixel: vec2f) -> Ray {
    let clip = pixel
        / vec2f(f32(UNIFORMS.width), f32(UNIFORMS.height))
        * 2.0
        - 1.0;
    let camera = UNIFORMS.perspective * vec4f(clip.x, -clip.y, 0.0, 1.0);
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
    var intersection = ray_query(ray, 0.001, F32_MAX);
    var radiance = vec4f();
    var attenuation = vec4f(1.0, 1.0, 1.0, 0.0);
    var bounces = UNIFORMS.bounces;
    var scattered = vec3f();
    var pdf = 1.0;
    var out_color = vec4f(1.0);
    
    while intersection.kind != RAY_QUERY_INTERSECTION_NONE && bounces > 0u {
        bounces -= 1u;
        
        let hit = get_intersection_data(intersection);
        let material = MATERIAL_BUFFER[hit.material];
        
        var normal = vec3f();
        
        if dot(ray.direction, hit.normal) < 0.0 {
            normal = hit.normal;
        } else {
            normal = -hit.normal;
        }

        let p = (intersection.object_to_world * vec4f(hit.pos, 0.0)).xyz + normal * F32_EPSILON;

        var in_color = vec4f();
        
        if material.has_texture == 1 {
            in_color = textureSampleLevel(TEXTURES[material.texture], SAMPLER, hit.uv, 0.0);
        } else {
            in_color = material.color;
        }       
               
        if material.emission > 0.0 {
            radiance += material.color * material.emission;
            break;
        } else if material.metallic > 0.0 {
            metal_brdf(normal, ray.direction, in_color, material.roughness, &scattered, &out_color, &pdf);
            attenuation *= out_color / pdf;            
        } else {
            if rand() > 0.5 {
                diffuse_brdf(normal, ray.direction, in_color, &scattered, &out_color, &pdf);
            } else {
                glass_brdf(normal, ray.direction, in_color, material.ior, &scattered, &out_color, &pdf);
            }
            attenuation *= (out_color / pdf) * 0.5;
        }

        let light = random_light();

        if !light_is_blocked(p, light) {
            radiance += sample_light(p, light) / (1.0 / f32(UNIFORMS.lights));
        }

        ray = Ray(p, scattered);
        intersection = ray_query(ray, 0.001, F32_MAX);
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
    return textureLoad(SAMPLES, vec2u(pos.xy));   
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    RNG = (gid.x + 1) * (gid.y + 1) * (UNIFORMS.current_chunk + 1) * UNIFORMS.seed;

    var chunk_x = UNIFORMS.current_chunk % (UNIFORMS.width / UNIFORMS.chunk_size);
    var chunk_y = UNIFORMS.current_chunk / (UNIFORMS.width / UNIFORMS.chunk_size);
    let pixel_x = (chunk_x * UNIFORMS.chunk_size) + gid.x; 
    let pixel_y = (chunk_y * UNIFORMS.chunk_size) + gid.y;
    let pixel = vec2u(pixel_x, pixel_y);

    if pixel.x > UNIFORMS.width || pixel.y > UNIFORMS.height {
        return;
    }
    
    var color = vec4f();

    for (var i = 0u; i < UNIFORMS.samples; i++) {
        color += pixel_color(vec2f(pixel) + vec2f(rand(), rand()));
    }

    let sample = color / f32(UNIFORMS.samples);

    textureStore(SAMPLES, pixel, sample);
}
