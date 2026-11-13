use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::picking::hover::Hovered;
use bevy::prelude::*;

use crate::ui::UiPanelRoot;

#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::new(0.0, 0.85, 0.0),
            radius: 4.3,
            yaw: 0.62,
            pitch: 0.34,
        }
    }
}

impl OrbitCamera {
    pub fn apply_to_transform(&self, transform: &mut Transform) {
        let cp = self.pitch.cos();
        let offset = Vec3::new(
            self.radius * cp * self.yaw.sin(),
            self.radius * self.pitch.sin(),
            self.radius * cp * self.yaw.cos(),
        );
        transform.translation = self.focus + offset;
        transform.look_at(self.focus, Vec3::Y);
    }
}

pub fn orbit_camera_system(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_msgs: MessageReader<MouseMotion>,
    mut scroll_msgs: MessageReader<MouseWheel>,
    ui_panel: Query<&Hovered, With<UiPanelRoot>>,
    mut cameras: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let pointer_on_ui = ui_panel.iter().any(|h| h.get());

    let mut delta = Vec2::ZERO;
    for ev in motion_msgs.read() {
        delta += ev.delta;
    }

    let mut zoom_delta = 0.0f32;
    for ev in scroll_msgs.read() {
        let scale = match ev.unit {
            MouseScrollUnit::Line => 0.35,
            MouseScrollUnit::Pixel => 0.003,
        };
        zoom_delta += ev.y * scale;
    }

    if pointer_on_ui {
        return;
    }

    for (mut orbit, mut transform) in cameras.iter_mut() {
        let mut changed = false;
        if mouse_buttons.pressed(MouseButton::Left) && delta != Vec2::ZERO {
            orbit.yaw -= delta.x * 0.0065;
            orbit.pitch = (orbit.pitch + delta.y * 0.0065).clamp(-0.05, 1.42);
            changed = true;
        }
        if zoom_delta.abs() > 1e-5 {
            orbit.radius = (orbit.radius - zoom_delta).clamp(1.8, 12.0);
            changed = true;
        }
        if changed {
            orbit.apply_to_transform(&mut transform);
        }
    }
}
