use std::f32::consts::PI;

use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;
use bevy::window::WindowResolution;

use wgsl_compute_shader::bin_asset_loader::BinAssetPlugin;
use wgsl_compute_shader::camera::{OrbitCamera, orbit_camera_system};
use wgsl_compute_shader::material::SoftBodyMaterialExt;
use wgsl_compute_shader::phys_data::{
    GenerationCounter, Preset, RequestLoadPreset, SimConfig, finalize_pending_model_load_system,
    trigger_preset_load_system,
};
use wgsl_compute_shader::sim_pipeline::SimPlugin;
use wgsl_compute_shader::ui::{spawn_ui, ui_interaction_system, update_stats_text_system};

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.07, 0.08, 0.11)))
        .init_resource::<SimConfig>()
        .init_resource::<GenerationCounter>()
        .add_message::<RequestLoadPreset>()
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "WGSL XPBD Soft-Body Simulation".into(),
                        resolution: WindowResolution::new(1120, 720),
                        canvas: Some("#bevy-canvas".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            BinAssetPlugin,
            MaterialPlugin::<ExtendedMaterial<StandardMaterial, SoftBodyMaterialExt>>::default(),
            SimPlugin,
        ))
        .add_systems(Startup, (setup_scene, spawn_ui))
        .add_systems(
            Update,
            (
                ui_interaction_system,
                trigger_preset_load_system,
                finalize_pending_model_load_system,
                orbit_camera_system,
                update_stats_text_system,
            )
                .chain(),
        )
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut load_writer: MessageWriter<RequestLoadPreset>,
) {
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.82, 0.96),
        brightness: 260.0,
        ..default()
    });

    // Key directional light with dynamic shadows
    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.97, 0.92),
            illuminance: 9_500.0,
            shadow_maps_enabled: true,
            ..default()
        },
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 6.0,
            maximum_distance: 18.0,
            ..default()
        }
        .build(),
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.95, 0.55, 0.0)),
    ));

    // Rim point light for depth separation
    commands.spawn((
        PointLight {
            color: Color::srgb(0.45, 0.65, 1.0),
            intensity: 180_000.0,
            range: 20.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(-3.5, 3.8, -3.5),
    ));

    // Ground floor at y = 0.0
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(14.0, 14.0))),
        MeshMaterial3d(std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.18, 0.22),
            perceptual_roughness: 0.88,
            metallic: 0.05,
            ..default()
        })),
        Transform::IDENTITY,
    ));

    // Subtle floor reference ring
    commands.spawn((
        Mesh3d(meshes.add(Annulus::new(1.55, 1.58))),
        MeshMaterial3d(std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.26, 0.31, 0.40),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.002, 0.0).with_rotation(Quat::from_rotation_x(-PI * 0.5)),
    ));

    // Orbit camera
    let orbit = OrbitCamera::default();
    let mut cam_transform = Transform::IDENTITY;
    orbit.apply_to_transform(&mut cam_transform);
    commands.spawn((Camera3d::default(), orbit, cam_transform));

    // Trigger initial default model (Bunny)
    load_writer.write(RequestLoadPreset(Preset::Bunny));
}
