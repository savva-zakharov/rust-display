use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub aspect: f32,
    pub exposure: f32,
    pub gamma: f32,
    pub shift: f32,
    pub two_point_mode: u32,
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
            shift: 0.0,
            two_point_mode: 0,
        }
    }
}
