mod app;
mod shader;
mod uniforms;
mod luts;

use app::App;
use winit::event::{ElementState, Event, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut app = rt.block_on(App::new(&event_loop));

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
                                    }
                                }
                                PhysicalKey::Code(KeyCode::KeyP) => {
                                    if !app.egui_ctx.wants_keyboard_input() {
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
