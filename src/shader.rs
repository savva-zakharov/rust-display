pub const WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct Uniforms { 
    yaw: f32, 
    pitch: f32, 
    fov: f32, 
    aspect: f32, 
    exposure: f32, 
    gamma: f32, 
    shift: f32, 
    two_point_mode: u32, 
    use_lut: u32, 
    lut_type: u32,
    _padding1: u32,
    _padding2: u32,
};
@group(0) @binding(0) var panorama_texture: texture_2d<f32>;
@group(0) @binding(1) var panorama_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(0) @binding(3) var lut_texture: texture_3d<f32>;
@group(0) @binding(4) var lut_sampler: sampler;

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

fn agx_log(col: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        0.8424202358, 0.0441506126, 0.0215263202,
        0.0784335493, 0.8782670234, 0.0432976603,
        0.0792225102, 0.0772714571, 0.9354002705
    );
    let c = max(col * m, vec3<f32>(1e-10));
    return clamp((log2(c) + 10.0) / 25.0, vec3<f32>(0.0), vec3<f32>(1.0));
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

    if (uniforms.use_lut == 1u) {
        var lut_in = color.rgb;
        if (uniforms.lut_type == 1u) {
            lut_in = agx_log(lut_in);
        } else {
            lut_in = clamp(lut_in, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        color = vec4<f32>(textureSample(lut_texture, lut_sampler, lut_in).rgb, color.a);
    }

    // Apply gamma correction
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / uniforms.gamma)), color.a);
    return color;
}
"#;