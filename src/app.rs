use crate::shader;
use crate::uniforms::Uniforms;
use crate::luts::{self, Lut};
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
    panorama_textures: [Option<wgpu::Texture>; 2],
    panorama_views: [Option<wgpu::TextureView>; 2],
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    bind_groups: [Option<wgpu::BindGroup>; 2],
    bgl: wgpu::BindGroupLayout,
    surface_format: TextureFormat,
    pub uniforms: Uniforms,
    pub is_dragging: bool,
    pub drag_start: (f64, f64),
    pub yaw_start: f32,
    pub pitch_start: f32,
    pub shift_start: f32,
    pub current_panorama: usize,
    image_loaded: [bool; 2],
    pub should_close: bool,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    pub show_controls: bool,
    egui_shapes: Vec<egui::epaint::ClippedShape>,
    egui_pixels_per_point: f32,
    pub is_loading: bool,
    image_sender: Sender<Result<(Vec<u16>, u32, u32, PathBuf, usize), String>>,
    image_receiver: Receiver<Result<(Vec<u16>, u32, u32, PathBuf, usize), String>>,
    pub capture_path: Option<PathBuf>,
    pub luts: Vec<PathBuf>,
    pub current_lut: Option<usize>,
    lut_texture: Option<wgpu::Texture>,
    lut_view: Option<wgpu::TextureView>,
    lut_sampler: wgpu::Sampler,
}

impl App {
    pub async fn new(event_loop: &EventLoop<()>) -> Self {
        let (tx, rx) = mpsc::channel();
        let window = Arc::new(Window::new(event_loop).unwrap());
        window.set_title("360° Panorama Viewer");

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
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("LUT Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
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

        let lut_placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LUT Placeholder"),
            size: wgpu::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lv = lut_placeholder.create_view(&wgpu::TextureViewDescriptor::default());

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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG 1"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lv),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
                },
            ],
        });
        let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG 2"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&lv),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&lut_sampler),
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
        let available_luts = luts::list_luts();

        Self {
            window,
            surface,
            device,
            queue,
            pipeline,
            panorama_textures: [None, None],
            panorama_views: [None, None],
            sampler,
            uniform_buffer: ub,
            bind_groups: [Some(bg1), Some(bg2)],
            bgl,
            surface_format: sf,
            uniforms: Uniforms::default(),
            is_dragging: false,
            drag_start: (0.0, 0.0),
            yaw_start: 0.0,
            pitch_start: 0.0,
            shift_start: 0.0,
            current_panorama: 0,
            image_loaded: [false, false],
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
            luts: available_luts,
            current_lut: None,
            lut_texture: Some(lut_placeholder),
            lut_view: Some(lv),
            lut_sampler,
        }
    }

    pub fn start_loading_image(&mut self, path: PathBuf, slot: usize) {
        self.is_loading = true;
        let sender = self.image_sender.clone();
        let max_size = self.device.limits().max_texture_dimension_2d;
        thread::spawn(move || {
            let result = App::load_image_data(&path, max_size, slot);
            sender.send(result).unwrap();
        });
    }

    pub fn check_for_loaded_image(&mut self) {
        if self.is_loading {
            if let Ok(result) = self.image_receiver.try_recv() {
                self.is_loading = false;
                match result {
                    Ok((f16_data, w, h, path, slot)) => {
                        self.upload_hdr_from_data(&f16_data, w, h, slot);
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

    fn load_image_data(path: &PathBuf, max_size: u32, slot: usize) -> Result<(Vec<u16>, u32, u32, PathBuf, usize), String> {
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
        let img = if orig_w > max_size || orig_h > max_size {
            let s = max_size as f32 / orig_w.max(orig_h) as f32;
            let (nw, nh) = ((orig_w as f32 * s) as u32, (orig_h as f32 * s) as u32);
            img.resize(nw, nh, image::imageops::FilterType::Lanczos3)
        } else {
            img
        };

        let (w, h) = img.dimensions();
        let rgba32f = img.to_rgba32f();
        let f16_data: Vec<u16> = rgba32f
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| f16::from_f32(v).to_bits()))
            .collect();

        Ok((f16_data, w, h, path.clone(), slot))
    }

    fn upload_hdr_from_data(&mut self, f16_data: &[u16], w: u32, h: u32, slot: usize) {
        let texture_format = wgpu::TextureFormat::Rgba16Float;
        let bytes_per_pixel = 8;

        self.panorama_textures[slot] = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Panorama Texture {}", slot)),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let t = self.panorama_textures[slot].as_ref().unwrap();
        self.panorama_views[slot] = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));

        let bytes = bytemuck::cast_slice(f16_data);
        self.queue.write_texture(
            wgpu::ImageCopyTexture { texture: t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytes,
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(bytes_per_pixel * w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );

        self.recreate_bind_groups();
        self.image_loaded[slot] = true;
        self.current_panorama = slot;
    }

    pub fn set_lut(&mut self, index: Option<usize>) {
        self.current_lut = index;
        if let Some(idx) = index {
            let path = &self.luts[idx];
            let file_name = path.file_name().unwrap().to_string_lossy();
            if file_name.contains("AgX") {
                self.uniforms.lut_type = 1;
            } else {
                self.uniforms.lut_type = 0;
            }
            match Lut::load_cube(path) {
                Ok(lut) => {
                    self.upload_lut_texture(&lut);
                    self.uniforms.use_lut = 1;
                }
                Err(e) => {
                    eprintln!("Failed to load LUT: {}", e);
                    self.uniforms.use_lut = 0;
                }
            }
        } else {
            self.uniforms.use_lut = 0;
            self.uniforms.lut_type = 0;
        }
        self.update_uniforms();
        self.recreate_bind_groups();
    }

    fn upload_lut_texture(&mut self, lut: &Lut) {
        let size = lut.size;
        let mut rgba_data = Vec::with_capacity((size * size * size * 4) as usize);
        for i in 0..(size * size * size) as usize {
            rgba_data.push(f16::from_f32(lut.data[i * 3]).to_bits());
            rgba_data.push(f16::from_f32(lut.data[i * 3 + 1]).to_bits());
            rgba_data.push(f16::from_f32(lut.data[i * 3 + 2]).to_bits());
            rgba_data.push(f16::from_f32(1.0).to_bits());
        }

        self.lut_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LUT Texture"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        }));
        let t = self.lut_texture.as_ref().unwrap();
        self.lut_view = Some(t.create_view(&wgpu::TextureViewDescriptor::default()));

        self.queue.write_texture(
            wgpu::ImageCopyTexture { texture: t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&rgba_data),
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(8 * size), rows_per_image: Some(size) },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
        );
    }

    fn recreate_bind_groups(&mut self) {
        for i in 0..2 {
            let panorama_view = if let Some(v) = &self.panorama_views[i] { v } else {
                // If we don't have a view yet, we should use the placeholder view.
                // But wait, where is the placeholder view? It was local to App::new.
                // I should probably store it or just skip if None, 
                // but since App::new initializes them, it's better to ensure we always have something.
                continue;
            };
            let lut_view = self.lut_view.as_ref().expect("LUT view must be initialized");

            self.bind_groups[i] = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("BG {}", i)),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(panorama_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.uniform_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(lut_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.lut_sampler) },
                ],
            }));
        }
    }

    pub fn update_uniforms(&mut self) {
        self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    pub fn render(&mut self) {
        let size = self.window.inner_size();
        if size.width == 0 || size.height == 0 { return; }

        self.uniforms.aspect = size.width as f32 / size.height as f32;
        self.update_uniforms();

        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if self.capture_path.is_some() { usage |= wgpu::TextureUsages::COPY_SRC; }

        self.surface.configure(&self.device, &wgpu::SurfaceConfiguration {
            usage, format: self.surface_format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo, alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Enc") });

        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            if self.image_loaded[self.current_panorama] {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, self.bind_groups[self.current_panorama].as_ref().unwrap(), &[]);
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
                wgpu::ImageCopyTexture { texture: &frame.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::ImageCopyBuffer { buffer: &capture_buffer, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(padded_bytes_per_row), rows_per_image: Some(size.height) } },
                wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            );
            Some((path, capture_buffer, padded_bytes_per_row))
        } else { None };

        let clipped_primitives = self.egui_ctx.tessellate(self.egui_shapes.clone(), self.egui_pixels_per_point);
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut enc, &clipped_primitives, &egui_wgpu::ScreenDescriptor { size_in_pixels: [size.width, size.height], pixels_per_point: self.egui_pixels_per_point });
        
        {
            let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            self.egui_renderer.render(&mut rpass, &clipped_primitives, &egui_wgpu::ScreenDescriptor { size_in_pixels: [size.width, size.height], pixels_per_point: self.egui_pixels_per_point });
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
                        let i = y as usize * padded_bytes_per_row as usize + x as usize * u32_size;
                        let r; let g; let b;
                        match format {
                            wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm => { r = data[i + 2]; g = data[i + 1]; b = data[i]; }
                            _ => { r = data[i]; g = data[i + 1]; b = data[i + 2]; }
                        }
                        png_data.push(r); png_data.push(g); png_data.push(b);
                    }
                }
                image::save_buffer(&path, &png_data, width, height, image::ColorType::Rgb8).unwrap();
            });
        }
        frame.present();
    }

    pub fn draw_egui(&mut self) {
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let mut load_path: Option<(PathBuf, usize)> = None;
        let mut reset = false;
        let mut new_fov: Option<f32> = None;
        let mut new_exposure: Option<f32> = None;
        let mut new_gamma: Option<f32> = None;
        let mut update_uniforms = false;
        let mut selected_lut = self.current_lut;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            if self.is_loading {
                egui::Area::new(egui::Id::new("loading")).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| { ui.add(egui::Spinner::new()); });
            }
            if self.show_controls {
                egui::Area::new(egui::Id::new("controls")).anchor(egui::Align2::CENTER_TOP, [0.0, 20.0]).show(ctx, |ui| {
                    egui::Frame::window(ui.style()).rounding(20.0).fill(egui::Color32::from_black_alpha(180)).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 10.0;
                            if ui.button("📁 1").clicked() {
                                if let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "exr"]).pick_file() {
                                    load_path = Some((p, 0));
                                }
                            }
                            if ui.button("📁 2").clicked() {
                                if let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "exr"]).pick_file() {
                                    load_path = Some((p, 1));
                                }
                            }
                            ui.separator();
                            let mut current = self.current_panorama;
                            if ui.selectable_value(&mut current, 0, "Image 1").clicked() { self.current_panorama = 0; }
                            if ui.selectable_value(&mut current, 1, "Image 2").clicked() { self.current_panorama = 1; }
                            ui.separator();
                            if ui.button("⟲ Reset").clicked() { reset = true; }
                            if ui.button("💾 Save").clicked() {
                                if let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp"]).set_file_name("view.png").save_file() {
                                    self.capture_path = Some(p);
                                }
                            }
                            ui.separator();
                            let mut two_point = self.uniforms.two_point_mode == 1;
                            if ui.checkbox(&mut two_point, "2-Point").changed() {
                                self.uniforms.two_point_mode = if two_point { 1 } else { 0 };
                                if two_point { self.uniforms.pitch = 0.0; } else { self.uniforms.shift = 0.0; }
                                update_uniforms = true;
                            }
                            ui.separator();
                            ui.label("FOV:");
                            let mut fov = self.uniforms.fov;
                            if ui.add(egui::Slider::new(&mut fov, 5.0..=150.0).suffix("°")).changed() { new_fov = Some(fov); }
                            ui.separator();
                            ui.label("Exposure:");
                            let mut exposure = self.uniforms.exposure;
                            if ui.add(egui::Slider::new(&mut exposure, -30.0..=10.0).suffix(" EV")).changed() { new_exposure = Some(exposure); }
                            ui.separator();
                            ui.label("Gamma:");
                            let mut gamma = self.uniforms.gamma;
                            if ui.add(egui::Slider::new(&mut gamma, 0.5..=3.0)).changed() { new_gamma = Some(gamma); }
                            ui.separator();
                            ui.label("LUT:");
                            let lut_name = if let Some(idx) = selected_lut {
                                self.luts[idx].file_name().unwrap().to_string_lossy().to_string()
                            } else {
                                "None".to_string()
                            };
                            egui::ComboBox::from_id_source("lut_select")
                                .selected_text(lut_name)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut selected_lut, None, "None");
                                    for (i, path) in self.luts.iter().enumerate() {
                                        let name = path.file_name().unwrap().to_string_lossy();
                                        ui.selectable_value(&mut selected_lut, Some(i), name);
                                    }
                                });
                        });
                    });
                });
            }
        });

        if selected_lut != self.current_lut {
            self.set_lut(selected_lut);
        }

        if let Some((p, slot)) = load_path { self.start_loading_image(p, slot); }
        if reset { self.uniforms = Uniforms::default(); self.update_uniforms(); }
        if let Some(fov) = new_fov { self.uniforms.fov = fov; self.update_uniforms(); }
        if let Some(exposure) = new_exposure { self.uniforms.exposure = exposure; self.update_uniforms(); }
        if let Some(gamma) = new_gamma { self.uniforms.gamma = gamma; self.update_uniforms(); }
        if update_uniforms { self.update_uniforms(); }

        self.egui_shapes = full_output.shapes;
        self.egui_pixels_per_point = full_output.pixels_per_point;
        for (id, image_delta) in &full_output.textures_delta.set { self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta); }
        for id in &full_output.textures_delta.free { self.egui_renderer.free_texture(id); }
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
    }

    pub fn on_event(&mut self, event: &WindowEvent) {
        let _ = self.egui_state.on_window_event(&self.window, event);
    }
}
