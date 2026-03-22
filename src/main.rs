use bytemuck::{Pod, Zeroable};
use half::f16;
use image::GenericImageView;
use std::path::PathBuf;
use std::sync::Arc;
use wgpu::TextureFormat;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

mod shader {
    pub const WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
struct Uniforms { yaw: f32, pitch: f32, fov: f32, aspect: f32, exposure: f32, gamma: f32 };
@group(0) @binding(0) var panorama_texture: texture_2d<f32>;
@group(0) @binding(1) var panorama_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var panorama_texture2: texture_2d<f32>;
@group(1) @binding(1) var panorama_sampler2: sampler;

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
    let dir = normalize(vec3<f32>(in.uv.x*tan(half_fov)*uniforms.aspect, -in.uv.y*tan(half_fov), 1.0));
    let sph = to_spherical(rotate_y(rotate_x(dir, -uniforms.pitch), -uniforms.yaw));
    var theta = sph.x + 3.14159;
    if theta < 0.0 { theta = theta + 6.28318; }
    if theta >= 6.28318 { theta = theta - 6.28318; }
    var color: vec4<f32>;
    if theta < 3.14159 {
        color = textureSample(panorama_texture, panorama_sampler, vec2<f32>(theta/3.14159, 1.0-sph.y/3.14159));
    } else {
        color = textureSample(panorama_texture2, panorama_sampler2, vec2<f32>((theta-3.14159)/3.14159, 1.0-sph.y/3.14159));
    }
    // Apply exposure (logarithmic: 0 = 1x, 1 = 2x, 2 = 4x, -1 = 0.5x, etc.)
    color = vec4<f32>(color.rgb * pow(2.0, uniforms.exposure), color.a);
    // Apply gamma correction
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / uniforms.gamma)), color.a);
    return color;
}
"#;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    yaw: f32,
    pitch: f32,
    fov: f32,
    aspect: f32,
    exposure: f32,
    gamma: f32,
}
impl Default for Uniforms {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fov: 90.0,
            aspect: 1.0,
            exposure: 0.0,
            gamma: 1.0,
        }
    }
}

struct App {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    texture1: Option<wgpu::Texture>,
    texture1_view: Option<wgpu::TextureView>,
    texture2: Option<wgpu::Texture>,
    texture2_view: Option<wgpu::TextureView>,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    bind_group0: Option<wgpu::BindGroup>,
    bind_group1: Option<wgpu::BindGroup>,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    surface_format: TextureFormat,
    uniforms: Uniforms,
    is_dragging: bool,
    drag_start: (f64, f64),
    yaw_start: f32,
    pitch_start: f32,
    image_loaded: bool,
    should_close: bool,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    show_controls: bool,
    egui_shapes: Vec<egui::epaint::ClippedShape>,
    egui_pixels_per_point: f32,
}

impl App {
    async fn new(event_loop: &EventLoop<()>) -> Self {
        let window = Arc::new(Window::new(event_loop).unwrap());
        window.set_title("360° Panorama Viewer - Press 'O' to open image");

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
        );

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("No adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("No device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader::WGSL.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UB"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pv = placeholder.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BGL1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ub.as_entire_binding(),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG1"),
            layout: &bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&pv),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PL"),
            bind_group_layouts: &[&bgl0, &bgl1],
            push_constant_ranges: &[],
        });
        let sf = surface
            .get_capabilities(&adapter)
            .formats
            .into_iter()
            .find(|f| f.is_srgb())
            .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: sf,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
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
        });

        let egui_renderer = egui_wgpu::Renderer::new(&device, sf, None, 1);

        Self {
            window,
            surface,
            device,
            queue,
            pipeline,
            texture1: None,
            texture1_view: None,
            texture2: None,
            texture2_view: None,
            sampler,
            uniform_buffer: ub,
            bind_group0: Some(bg0),
            bind_group1: Some(bg1),
            bgl0,
            bgl1,
            surface_format: sf,
            uniforms: Uniforms::default(),
            is_dragging: false,
            drag_start: (0.0, 0.0),
            yaw_start: 0.0,
            pitch_start: 0.0,
            image_loaded: false,
            should_close: false,
            egui_ctx,
            egui_state,
            egui_renderer,
            show_controls: true,
            egui_shapes: Vec::new(),
            egui_pixels_per_point: 1.0,
        }
    }

    fn load_image(&mut self, path: &PathBuf) {
        // Use ImageReader for better control over large images
        let img = match image::ImageReader::open(path) {
            Ok(reader) => {
                let mut reader = reader.with_guessed_format().unwrap();
                // Disable memory limit for large HDR images
                reader.no_limits();
                match reader.decode() {
                    Ok(img) => img,
                    Err(e) => {
                        eprintln!("Failed to decode image: {}", e);
                        return;
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to open image file: {}", e);
                return;
            }
        };

        let (orig_w, orig_h) = img.dimensions();
        println!("Original image size: {}x{}", orig_w, orig_h);

        const HM: u32 = 8192;
        const MAX_WIDTH: u32 = HM * 2;

        // Resize before conversion if needed to save memory
        let img = if orig_w > MAX_WIDTH {
            let s = MAX_WIDTH as f32 / orig_w as f32;
            let (nw, nh) = ((orig_w as f32 * s) as u32, (orig_h as f32 * s) as u32);
            println!("Resizing to {}x{} before conversion", nw, nh);
            img.resize(nw, nh, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };

        let (w, h) = img.dimensions();

        // Convert to RGBA32F for HDR support (handles f16, f32, and LDR formats)
        let rgba32f = img.to_rgba32f();

        let hw = w / 2;

        let left = image::imageops::crop_imm(&rgba32f, 0, 0, hw, h).to_image();
        let right = image::imageops::crop_imm(&rgba32f, hw, 0, hw, h).to_image();
        println!("Loaded: {}x{} → two {}x{} textures", w, h, hw, h);
        self.upload_hdr(&left, &right, hw, h);

        // Update window title with filename
        if let Some(file_name) = path.file_name() {
            self.window.set_title(&file_name.to_string_lossy());
        }
    }

    fn upload_hdr(
        &mut self,
        left: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>,
        right: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>,
        w: u32,
        h: u32,
    ) {
        // Use Rgba16Float for HDR textures (good balance of precision and memory)
        let texture_format = wgpu::TextureFormat::Rgba16Float;
        let bytes_per_pixel = 8; // 4 channels * 16 bits / 8 = 8 bytes

        self.texture1 = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T1"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let t1 = self.texture1.as_ref().unwrap();
        self.texture1_view = Some(t1.create_view(&wgpu::TextureViewDescriptor::default()));

        // Convert f32 to f16 for upload
        let left_f16: Vec<u16> = left
            .pixels()
            .flat_map(|p| {
                p.0.iter().map(|&v| {
                    let f16_val = f16::from_f32(v);
                    f16_val.to_bits()
                })
            })
            .collect();
        let left_bytes = bytemuck::cast_slice(&left_f16);

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: t1,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            left_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_pixel * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.texture2 = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T2"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let t2 = self.texture2.as_ref().unwrap();
        self.texture2_view = Some(t2.create_view(&wgpu::TextureViewDescriptor::default()));

        // Convert f32 to f16 for upload
        let right_f16: Vec<u16> = right
            .pixels()
            .flat_map(|p| {
                p.0.iter().map(|&v| {
                    let f16_val = f16::from_f32(v);
                    f16_val.to_bits()
                })
            })
            .collect();
        let right_bytes = bytemuck::cast_slice(&right_f16);

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: t2,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            right_bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_pixel * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.recreate_bind_groups();
        self.image_loaded = true;
    }

    fn recreate_bind_groups(&mut self) {
        let Some(t1v) = &self.texture1_view else {
            return;
        };
        let Some(t2v) = &self.texture2_view else {
            return;
        };

        self.bind_group0 = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG0"),
            layout: &self.bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(t1v),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        }));
        self.bind_group1 = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG1"),
            layout: &self.bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(t2v),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }

    fn update_uniforms(&mut self) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    fn render(&mut self) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        // Update aspect ratio
        self.uniforms.aspect = size.width as f32 / size.height as f32;
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Enc") });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            if self.image_loaded {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, self.bind_group0.as_ref().unwrap(), &[]);
                pass.set_bind_group(1, self.bind_group1.as_ref().unwrap(), &[]);
                pass.draw(0..6, 0..1);
            }
        }

        // Render egui on top
        let clipped_primitives = self
            .egui_ctx
            .tessellate(self.egui_shapes.clone(), self.egui_pixels_per_point);
        
        // Update egui vertex/index buffers
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut enc,
            &clipped_primitives,
            &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: self.egui_pixels_per_point,
            },
        );
        
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.egui_renderer.render(
                &mut rpass,
                &clipped_primitives,
                &egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [size.width, size.height],
                    pixels_per_point: self.egui_pixels_per_point,
                },
            );
        }

        self.queue.submit(std::iter::once(enc.finish()));
        frame.present();
    }

    fn draw_egui(&mut self) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut load_path: Option<PathBuf> = None;
        let mut reset = false;
        let mut new_fov: Option<f32> = None;
        let mut new_exposure: Option<f32> = None;
        let mut new_gamma: Option<f32> = None;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if self.show_controls {
                egui::Area::new(egui::Id::new("controls"))
                    .anchor(egui::Align2::CENTER_TOP, [0.0, 20.0])
                    .show(ctx, |ui| {
                        egui::Frame::window(ui.style())
                            .rounding(20.0)
                            .shadow(egui::epaint::Shadow {
                                blur: 10.0,
                                ..Default::default()
                            })
                            .fill(egui::Color32::from_black_alpha(180))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 15.0;
                                    if ui.button("📁 Open").clicked() {
                                        if let Some(p) = rfd::FileDialog::new()
                                            .add_filter(
                                                "Images",
                                                &[
                                                    "png", "jpg", "jpeg", "bmp", "webp", "tif",
                                                    "tiff", "exr",
                                                ],
                                            )
                                            .pick_file()
                                        {
                                            load_path = Some(p);
                                        }
                                    }
                                    if ui.button("⟲ Reset").clicked() {
                                        reset = true;
                                    }
                                    ui.separator();
                                    ui.label("FOV:");
                                    let mut fov = self.uniforms.fov;
                                    if ui
                                        .add(egui::Slider::new(&mut fov, 5.0..=150.0).suffix("°"))
                                        .changed()
                                    {
                                        new_fov = Some(fov);
                                    }
                                    ui.separator();
                                    ui.label("Exposure:");
                                    let mut exposure = self.uniforms.exposure;
                                    if ui
                                        .add(egui::Slider::new(&mut exposure, -30.0..=10.0).suffix(" EV"))
                                        .changed()
                                    {
                                        new_exposure = Some(exposure);
                                    }
                                    ui.separator();
                                    ui.label("Gamma:");
                                    let mut gamma = self.uniforms.gamma;
                                    if ui
                                        .add(egui::Slider::new(&mut gamma, 0.5..=3.0))
                                        .changed()
                                    {
                                        new_gamma = Some(gamma);
                                    }
                                    if self.image_loaded {
                                        let mut yaw = String::from( self.uniforms.yaw.to_degrees().round().to_string());
                                        while yaw.len() < 6 {
                                            yaw =  " ".to_string() + &yaw;
                                        }
                                        let mut pitch = String::from(self.uniforms.pitch.to_degrees().round().to_string());
                                        while pitch.len() < 6 {
                                            pitch =  " ".to_string() + &pitch;
                                        }

                                        ui.separator();
                                        ui.label(format!(
                                            "Yaw: {}° | Pitch: {}°",
                                            yaw,
                                            pitch
                                        ));
                                    }
                                });
                            });
                    });
            }
        });

        // Apply changes after egui context is released
        if let Some(p) = load_path {
            self.load_image(&p);
        }
        if reset {
            self.uniforms = Uniforms::default();
            self.update_uniforms();
        }
        if let Some(fov) = new_fov {
            self.uniforms.fov = fov;
            self.update_uniforms();
        }
        if let Some(exposure) = new_exposure {
            self.uniforms.exposure = exposure;
            self.update_uniforms();
        }
        if let Some(gamma) = new_gamma {
            self.uniforms.gamma = gamma;
            self.update_uniforms();
        }

        // Store shapes and pixels_per_point for tessellation in render()
        self.egui_shapes = full_output.shapes;
        self.egui_pixels_per_point = full_output.pixels_per_point;

        // Update egui textures (fonts, icons, etc.)
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);
    }

    fn on_event(&mut self, event: &WindowEvent) {
        let _ = self.egui_state.on_window_event(&self.window, event);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut app = rt.block_on(App::new(&event_loop));

    let _ = event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                app.on_event(&event);
                match event {
                    WindowEvent::CloseRequested => app.should_close = true,
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::KeyO) => {
                                    if let Some(p) = rfd::FileDialog::new()
                                        .add_filter(
                                            "Images",
                                            &[
                                                "png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff",
                                                "exr",
                                            ],
                                        )
                                        .pick_file()
                                    {
                                        app.load_image(&p);
                                    }
                                }
                                PhysicalKey::Code(KeyCode::Escape) => app.should_close = true,
                                PhysicalKey::Code(KeyCode::Space) => {
                                    app.show_controls = !app.show_controls
                                }
                                // Exposure controls: , to decrease, . to increase
                                //Comma, Period,  Semicolon, Quote
                                PhysicalKey::Code(KeyCode::Comma) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        app.uniforms.exposure = (app.uniforms.exposure - 1.0).max(-30.0);
                                        app.update_uniforms();
                                    }
                                }
                                PhysicalKey::Code(KeyCode::Period) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        app.uniforms.exposure = (app.uniforms.exposure + 1.0).min(10.0);
                                        app.update_uniforms();
                                    }
                                }
                                // Gamma controls: ; to decrease, ' to increase
                                PhysicalKey::Code(KeyCode::Semicolon) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        app.uniforms.gamma = (app.uniforms.gamma - 0.1).max(0.5);
                                        app.update_uniforms();
                                    }
                                }
                                PhysicalKey::Code(KeyCode::Quote) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        app.uniforms.gamma = (app.uniforms.gamma + 0.1).min(3.0);
                                        app.update_uniforms();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        use winit::event::MouseButton;
                        if button == MouseButton::Left && !app.egui_ctx.wants_pointer_input() {
                            app.is_dragging = state == ElementState::Pressed;
                            if app.is_dragging {
                                app.yaw_start = app.uniforms.yaw;
                                app.pitch_start = app.uniforms.pitch;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if app.is_dragging && !app.egui_ctx.wants_pointer_input() {
                            let (dx, dy) =
                                (position.x - app.drag_start.0, position.y - app.drag_start.1);
                            let sens = 0.0025 * (app.uniforms.fov / 90.0);
                            app.uniforms.yaw = app.yaw_start + dx as f32 * sens;
                            app.uniforms.pitch =
                                (app.pitch_start - dy as f32 * sens).clamp(-1.55, 1.55);
                            app.update_uniforms();
                        } else {
                            app.drag_start = (position.x, position.y);
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        if !app.egui_ctx.wants_pointer_input() {
                            let s = match delta {
                                MouseScrollDelta::LineDelta(_, y) => -y * 5.0,
                                MouseScrollDelta::PixelDelta(p) => -p.y as f32 * 0.05,
                            };
                            app.uniforms.fov = (app.uniforms.fov + s).clamp(5.0, 150.0);
                            app.update_uniforms();
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        app.draw_egui();
                        app.render();
                    }
                    _ => {}
                }
            }
            winit::event::Event::AboutToWait => {
                if app.should_close {
                    elwt.exit();
                }
                app.window.request_redraw();
            }
            _ => {}
        }
    });
}
