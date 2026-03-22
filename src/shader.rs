pub const WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct Uniforms { yaw: f32, pitch: f32, fov: f32, aspect: f32, exposure: f32, gamma: f32, shift: f32, two_point_mode: u32 };
@group(0) @binding(0) var panorama_texture: texture_2d<f32>;
@group(0) @binding(1) var panorama_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

fn rotate_y(v: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(c*v.x+s*v.z, v.y, -s*v.x+c*v.z);
}
fn rotate_x(v: vec3<f32>, a: f32) -> vec3<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec3<f32>(v.x, c*v.y-s*v.z, s*v.y+c*v.z);
}
fn to_spherical(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.x,d.z), acos(clamp(d.y,-1.0,1.0)));
}

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var p = array<vec2<f32>,6>(vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),vec2<f32>(-1.0,1.0),vec2<f32>(-1.0,1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0));
    var o: VertexOutput;
    o.position = vec4<f32>(p[i],0.0,1.0);
    o.uv = p[i];
    return o;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let half_fov = uniforms.fov * 0.5 * 0.01745;
    let dir = normalize(vec3<f32>(in.uv.x*tan(half_fov)*uniforms.aspect, (-in.uv.y + uniforms.shift)*tan(half_fov), 1.0));
    let sph = to_spherical(rotate_y(rotate_x(dir, -uniforms.pitch), -uniforms.yaw));
    let uv = vec2<f32>(sph.x / 6.28318 + 0.5, 1.0 - sph.y / 3.14159);
    var color = textureSample(panorama_texture, panorama_sampler, uv);

    // Apply exposure (2^exposure because exposure is in stops; e.g., 1 = 2x, 2 = 4x, -1 = 0.5x, etc.)
    color = vec4<f32>(color.rgb * pow(2.0, uniforms.exposure), color.a);
    // Apply gamma correction
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / uniforms.gamma)), color.a);
    return color;
}
"#;