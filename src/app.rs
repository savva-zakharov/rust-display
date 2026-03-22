use crate::shader;
use crate::uniforms::Uniforms;
use half::f16;
use image::GenericImageView;
use std::path::PathBuf;
use std::sync::Arc;
use wgpu::TextureFormat;
use winit::event::{WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;


pub struct App {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    panorama_texture: Option<wgpu::Texture>,
    panorama_view: Option<wgpu::TextureView>,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bgl: wgpu::BindGroupLayout,
    surface_format: TextureFormat,
    pub uniforms: Uniforms,
    pub is_dragging: bool,
    pub drag_start: (f64, f64),
    pub yaw_start: f32,
    pub pitch_start: f32,
    pub shift_start: f32,
    image_loaded: bool,
    pub should_close: bool,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    pub show_controls: bool,
    egui_shapes: Vec<egui::epaint::ClippedShape>,
    egui_pixels_per_point: f32,
    pub is_loading: bool,
    image_sender: Sender<Result<(Vec<u16>, u32, u32, PathBuf), String>>,
    image_receiver: Receiver<Result<(Vec<u16>, u32, u32, PathBuf), String>>,
    pub capture_path: Option<PathBuf>,
}

impl App {
    pub async fn new(event_loop: &EventLoop<()>) -> Self {
        let (tx, rx) = mpsc::channel();
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

        // Request high limits
        let mut limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();
        limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        println!("Adapter max texture size: {}", limits.max_texture_dimension_2d);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
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
            address_mode_u: wgpu::AddressMode::Repeat, // Panoramic repeat
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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BGL"),
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

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG"),
            layout: &bgl,
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

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PL"),
            bind_group_layouts: &[&bgl],
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
            panorama_texture: None,
            panorama_view: None,
            sampler,
            uniform_buffer: ub,
            bind_group: Some(bg),
            bgl,
            surface_format: sf,
            uniforms: Uniforms::default(),
            is_dragging: false,
            drag_start: (0.0, 0.0),
            yaw_start: 0.0,
            pitch_start: 0.0,
            shift_start: 0.0,
            image_loaded: false,
            should_close: false,
            egui_ctx,
            egui_state,
            egui_renderer,
            show_controls: true,
            egui_shapes: Vec::new(),
            egui_pixels_per_point: 1.0,
            is_loading: false,
            image_sender: tx,
            image_receiver: rx,
            capture_path: None,
        }
    }

    pub fn start_loading_image(&mut self, path: PathBuf) {
        self.is_loading = true;
        let sender = self.image_sender.clone();
        let max_size = self.device.limits().max_texture_dimension_2d;
        thread::spawn(move || {
            let result = App::load_image_data(&path, max_size);
            sender.send(result).unwrap();
        });
    }

    pub fn check_for_loaded_image(&mut self) {
        if self.is_loading {
            if let Ok(result) = self.image_receiver.try_recv() {
                self.is_loading = false;
                match result {
                    Ok((f16_data, w, h, path)) => {
                        self.upload_hdr_from_data(&f16_data, w, h);
                        if let Some(file_name) = path.file_name() {
                            self.window.set_title(&file_name.to_string_lossy());
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to load image: {}", e);
                    }
                }
            }
        }
    }

    fn load_image_data(path: &PathBuf, max_size: u32) -> Result<(Vec<u16>, u32, u32, PathBuf), String> {
        let img = match image::ImageReader::open(path) {
            Ok(reader) => {
                let mut reader = reader.with_guessed_format().unwrap();
                reader.no_limits();
                match reader.decode() {
                    Ok(img) => img,
                    Err(e) => return Err(e.to_string()),
                }
            }
            Err(e) => return Err(e.to_string()),
        };

        let (orig_w, orig_h) = img.dimensions();
        println!("Original image size: {}x{}", orig_w, orig_h);

        let img = if orig_w > max_size || orig_h > max_size {
            let s_w = max_size as f32 / orig_w as f32;
            let s_h = max_size as f32 / orig_h as f32;
            let s = s_w.min(s_h);
            let (nw, nh) = ((orig_w as f32 * s) as u32, (orig_h as f32 * s) as u32);
            println!("Resizing to {}x{} (HW limit: {})", nw, nh, max_size);
            img.resize(nw, nh, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };

        let (w, h) = img.dimensions();
        let rgba32f = img.to_rgba32f();

        println!("Loaded: {}x{}", w, h);

        let f16_data: Vec<u16> = rgba32f
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| f16::from_f32(v).to_bits()))
            .collect();

        Ok((f16_data, w, h, path.clone()))
    }

    fn upload_hdr_from_data(
        &mut self,
        f16_data: &[u16],
        w: u32,
        h: u32,
    ) {
        let texture_format = wgpu::TextureFormat::Rgba16Float;
        let bytes_per_pixel = 8;

        self.panorama_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Panorama Texture"),
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
        let t = self.panorama_texture.as_ref().unwrap();
        self.panorama_view = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));

        let bytes = bytemuck::cast_slice(f16_data);

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: t,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
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
        let Some(tv) = &self.panorama_view else {
            return;
        };

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tv),
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
    }

    pub fn update_uniforms(&mut self) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    pub fn render(&mut self) {
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

        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if self.capture_path.is_some() {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }

        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage,
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
                pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                pass.draw(0..6, 0..1);
            }
        }

        let capture_info = if let Some(path) = self.capture_path.take() {
            let u32_size = std::mem::size_of::<u32>() as u32;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let unpadded_bytes_per_row = u32_size * size.width;
            let padding = (align - unpadded_bytes_per_row % align) % align;
            let padded_bytes_per_row = unpadded_bytes_per_row + padding;

            let capture_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Capture Buffer"),
                size: (padded_bytes_per_row * size.height) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            enc.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &frame.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &capture_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row),
                        rows_per_image: Some(size.height),
                    },
                },
                wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
            );
            Some((path, capture_buffer, padded_bytes_per_row))
        } else {
            None
        };

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

        if let Some((path, capture_buffer, padded_bytes_per_row)) = capture_info {
            let buffer_slice = capture_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = buffer_slice.get_mapped_range().to_vec();
            let width = size.width;
            let height = size.height;
            let format = self.surface_format;
            let u32_size = std::mem::size_of::<u32>() as usize;

            thread::spawn(move || {
                let mut png_data = Vec::with_capacity((width * height * 3) as usize);
                for y in 0..height {
                    for x in 0..width {
                        let i = (y as usize * padded_bytes_per_row as usize + x as usize * u32_size);
                        let r; let g; let b;
                        match format {
                            wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm => {
                                r = data[i + 2];
                                g = data[i + 1];
                                b = data[i];
                            }
                            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Rgba8Unorm => {
                                r = data[i];
                                g = data[i + 1];
                                b = data[i + 2];
                            }
                            _ => {
                                r = data[i];
                                g = data[i + 1];
                                b = data[i + 2];
                            }
                        }
                        png_data.push(r);
                        png_data.push(g);
                        png_data.push(b);
                    }
                }
                image::save_buffer(
                    &path,
                    &png_data,
                    width,
                    height,
                    image::ColorType::Rgb8,
                ).unwrap();
                println!("View saved to {:?}", path);
            });
        }

        frame.present();
    }

    pub fn draw_egui(&mut self) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut load_path: Option<PathBuf> = None;
        let mut reset = false;
        let mut new_fov: Option<f32> = None;
        let mut new_exposure: Option<f32> = None;
        let mut new_gamma: Option<f32> = None;

        let mut update_uniforms = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if self.is_loading {
                egui::Area::new(egui::Id::new("loading"))
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        ui.add(egui::Spinner::new());
                    });
            }
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
                                    if ui.button("💾 Save").clicked() {
                                        if let Some(p) = rfd::FileDialog::new()
                                            .add_filter("Images", &["png", "jpg", "jpeg", "bmp"])
                                            .set_file_name("view.png")
                                            .save_file()
                                        {
                                            self.capture_path = Some(p);
                                        }
                                    }
                                    ui.separator();
                                    let mut two_point = self.uniforms.two_point_mode == 1;
                                    if ui.checkbox(&mut two_point, "2-Point").changed() {
                                        self.uniforms.two_point_mode = if two_point { 1 } else { 0 };
                                        if two_point {
                                            self.uniforms.pitch = 0.0;
                                        } else {
                                            self.uniforms.shift = 0.0;
                                        }
                                        update_uniforms = true;
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
                                        
                                        ui.separator();
                                        if self.uniforms.two_point_mode == 1 {
                                            ui.label(format!(
                                                "Yaw: {}° | Shift: {:.2}",
                                                yaw,
                                                self.uniforms.shift
                                            ));
                                        } else {
                                            let mut pitch = String::from(self.uniforms.pitch.to_degrees().round().to_string());
                                            while pitch.len() < 6 {
                                                pitch =  " ".to_string() + &pitch;
                                            }
                                            ui.label(format!(
                                                "Yaw: {}° | Pitch: {}°",
                                                yaw,
                                                pitch
                                            ));
                                        }
                                    }
                                });
                            });
                    });
            }
        });

        // Apply changes after egui context is released
        if let Some(p) = load_path {
            self.start_loading_image(p);
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
        if update_uniforms {
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

    pub fn on_event(&mut self, event: &WindowEvent) {
        let _ = self.egui_state.on_window_event(&self.window, event);
    }
}
