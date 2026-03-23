mod app;
mod shader;
mod uniforms;
mod luts;

use app::App;
use winit::event::{ElementState, Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Debug).expect("Could not initialize logger");
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let app = rt.block_on(App::new(&event_loop));
        run_app(event_loop, app);
    }

    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            let mut app = App::new(&event_loop).await;
            
            // On wasm, we need to append the canvas to the document body
            use winit::platform::web::WindowExtWebSys;
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .and_then(|body| {
                    let canvas = web_sys::Element::from(app.window.canvas().unwrap());
                    body.append_child(&canvas).ok()
                })
                .expect("Couldn't append canvas to document body.");

            // Check for URL parameters
            if let Some(window) = web_sys::window() {
                if let Ok(search) = window.location().search() {
                    let params = web_sys::UrlSearchParams::new_with_str(&search).unwrap();
                    if let Some(img1) = params.get("img1") {
                        app.load_image_from_url(img1, 0);
                    }
                    if let Some(img2) = params.get("img2") {
                        app.load_image_from_url(img2, 1);
                    }
                }
            }

            run_app(event_loop, app);
        });
    }
}

fn run_app(event_loop: EventLoop<()>, mut app: App) {
    let _ = event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => {
                app.on_event(&event);
                match event {
                    WindowEvent::CloseRequested => app.should_close = true,
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed {
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::KeyO) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        #[cfg(not(target_arch = "wasm32"))]
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
                                            app.start_loading_image(p, 0);
                                        }

                                        #[cfg(target_arch = "wasm32")]
                                        {
                                            let sender = app.image_sender.clone();
                                            wasm_bindgen_futures::spawn_local(async move {
                                                if let Some(file) = rfd::AsyncFileDialog::new()
                                                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "exr"])
                                                    .pick_file()
                                                    .await
                                                {
                                                    let data = file.read().await;
                                                    let path = std::path::PathBuf::from(file.file_name());
                                                    let max_size = 4096; // Adjust if needed
                                                    let result = App::load_image_data_from_memory(&data, path, max_size, 0)
                                                        .map(|(data, w, h, path, slot)| crate::app::LoadResult::Image(data, w, h, path, slot));
                                                    sender.send(result).unwrap();
                                                }
                                            });
                                        }
                                    }
                                }
                                PhysicalKey::Code(KeyCode::KeyP) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        #[cfg(not(target_arch = "wasm32"))]
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
                                            app.start_loading_image(p, 1);
                                        }

                                        #[cfg(target_arch = "wasm32")]
                                        {
                                            let sender = app.image_sender.clone();
                                            wasm_bindgen_futures::spawn_local(async move {
                                                if let Some(file) = rfd::AsyncFileDialog::new()
                                                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "exr"])
                                                    .pick_file()
                                                    .await
                                                {
                                                    let data = file.read().await;
                                                    let path = std::path::PathBuf::from(file.file_name());
                                                    let max_size = 4096; // Adjust if needed
                                                    let result = App::load_image_data_from_memory(&data, path, max_size, 1)
                                                        .map(|(data, w, h, path, slot)| crate::app::LoadResult::Image(data, w, h, path, slot));
                                                    sender.send(result).unwrap();
                                                }
                                            });
                                        }
                                    }
                                }
                                PhysicalKey::Code(KeyCode::KeyS) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        app.current_panorama = 1 - app.current_panorama;
                                    }
                                }
                                PhysicalKey::Code(KeyCode::BracketLeft) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        let len = app.luts.len();
                                        let next = match app.current_lut {
                                            None => if len > 0 { Some(len - 1) } else { None },
                                            Some(0) => None,
                                            Some(i) => Some(i - 1),
                                        };
                                        app.set_lut(next);
                                    }
                                }
                                PhysicalKey::Code(KeyCode::BracketRight) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
                                        let len = app.luts.len();
                                        let next = match app.current_lut {
                                            None => if len > 0 { Some(0) } else { None },
                                            Some(i) if i + 1 < len => Some(i + 1),
                                            _ => None,
                                        };
                                        app.set_lut(next);
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
                                app.shift_start = app.uniforms.shift;
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if app.is_dragging && !app.egui_ctx.wants_pointer_input() {
                            let (dx, dy) =
                                (position.x - app.drag_start.0, position.y - app.drag_start.1);
                            let sens = 0.00125 * (app.uniforms.fov / 90.0);
                            app.uniforms.yaw = app.yaw_start + dx as f32 * sens;
                            if app.uniforms.two_point_mode == 1 {
                                app.uniforms.shift = app.shift_start - dy as f32 * sens;
                            } else {
                                app.uniforms.pitch =
                                    (app.pitch_start - dy as f32 * sens).clamp(-1.55, 1.55);
                            }
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
                    WindowEvent::Resized(size) => {
                        app.uniforms.aspect = size.width as f32 / size.height as f32;
                    }
                    WindowEvent::RedrawRequested => {
                        app.draw_egui();
                        app.render();
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                app.check_for_loaded_image();
                if app.should_close {
                    elwt.exit();
                }
                app.window.request_redraw();
            }
            _ => {}
        }
    });
}
