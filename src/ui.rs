use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::input_focus::InputFocus;
use bevy::picking::Pickable;
use bevy::picking::hover::Hovered;
use bevy::prelude::*;
use bevy::ui::Pressed;
use bevy::ui_widgets::{Activate, Button, observe};

use crate::phys_data::{PhysData, Preset, RequestLoadPreset, SimConfig, StiffnessPreset};

#[derive(Component)]
pub struct UiPanelRoot;

#[derive(Component, Copy, Clone)]
pub enum UiAction {
    SelectPreset(Preset),
    SelectStiffness(StiffnessPreset),
    TogglePause,
    Reset,
    ToggleGravity,
}

#[derive(Component)]
pub struct StatsText;

#[derive(Component)]
pub struct PauseButtonLabel;

#[derive(Component)]
pub struct GravityButtonLabel;

const BG_PANEL: Color = Color::srgba(0.09, 0.11, 0.15, 0.86);
const BORDER_PANEL: Color = Color::srgba(1.0, 1.0, 1.0, 0.10);
const BTN_IDLE: Color = Color::srgba(0.18, 0.21, 0.28, 0.95);
const BTN_HOVER: Color = Color::srgba(0.26, 0.30, 0.39, 0.98);
const BTN_ACTIVE: Color = Color::srgb(0.24, 0.54, 0.88);
const TEXT_PRIMARY: Color = Color::srgb(0.94, 0.96, 0.99);
const TEXT_MUTED: Color = Color::srgb(0.62, 0.67, 0.76);

pub fn spawn_ui(mut commands: Commands) {
    commands
        .spawn((
            UiPanelRoot,
            Hovered::default(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(14.0),
                left: Val::Px(14.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(8.0),
                padding: UiRect::all(Val::Px(12.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(BG_PANEL),
            BorderColor::all(BORDER_PANEL),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("WGSL XPBD Soft-Body"),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(TEXT_PRIMARY),
            ));

            panel.spawn((
                StatsText,
                Text::new("Loading model..."),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(TEXT_MUTED),
            ));

            // Row 1: 4 Model Presets
            panel
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(5.0),
                    ..default()
                })
                .with_children(|row| {
                    for preset in Preset::ALL {
                        spawn_btn(row, preset.label(), UiAction::SelectPreset(preset));
                    }
                });

            // Row 2: Stiffness Presets
            panel
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(5.0),
                    align_items: AlignItems::Center,
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Text::new("Stiffness:"),
                        TextFont {
                            font_size: FontSize::Px(12.0),
                            ..default()
                        },
                        TextColor(TEXT_MUTED),
                    ));
                    for stiff in StiffnessPreset::ALL {
                        spawn_btn(row, stiff.label(), UiAction::SelectStiffness(stiff));
                    }
                });

            // Row 3: Simulation Controls
            panel
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(5.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        Button,
                        Hovered::default(),
                        UiAction::TogglePause,
                        button_node(),
                        BackgroundColor(BTN_IDLE),
                        observe(on_ui_button_activate),
                    ))
                    .with_children(|b| {
                        b.spawn((
                            PauseButtonLabel,
                            Pickable::IGNORE,
                            Text::new("Pause"),
                            TextFont {
                                font_size: FontSize::Px(12.0),
                                ..default()
                            },
                            TextColor(TEXT_PRIMARY),
                        ));
                    });

                    spawn_btn(row, "Reset (R)", UiAction::Reset);

                    row.spawn((
                        Button,
                        Hovered::default(),
                        UiAction::ToggleGravity,
                        button_node(),
                        BackgroundColor(BTN_ACTIVE),
                        observe(on_ui_button_activate),
                    ))
                    .with_children(|b| {
                        b.spawn((
                            GravityButtonLabel,
                            Pickable::IGNORE,
                            Text::new("Gravity: ON"),
                            TextFont {
                                font_size: FontSize::Px(12.0),
                                ..default()
                            },
                            TextColor(TEXT_PRIMARY),
                        ));
                    });
                });

            panel.spawn((
                Text::new("Drag: Orbit | Scroll: Zoom | Space: Pause"),
                TextFont {
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(TEXT_MUTED),
            ));
        });
}

fn button_node() -> Node {
    Node {
        padding: UiRect::axes(Val::Px(9.0), Val::Px(5.0)),
        justify_content: JustifyContent::Center,
        align_items: AlignItems::Center,
        border_radius: BorderRadius::all(Val::Px(6.0)),
        ..default()
    }
}

fn spawn_btn(parent: &mut ChildSpawnerCommands, text: &str, action: UiAction) {
    parent
        .spawn((
            Button,
            Hovered::default(),
            action,
            button_node(),
            BackgroundColor(BTN_IDLE),
            observe(on_ui_button_activate),
        ))
        .with_children(|b| {
            b.spawn((
                Pickable::IGNORE,
                Text::new(text),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(TEXT_PRIMARY),
            ));
        });
}

fn on_ui_button_activate(
    activate: On<Activate>,
    actions: Query<&UiAction>,
    mut sim_config: ResMut<SimConfig>,
    mut load_writer: MessageWriter<RequestLoadPreset>,
    mut input_focus: ResMut<InputFocus>,
) {
    input_focus.clear();
    let Ok(&action) = actions.get(activate.entity) else {
        return;
    };
    match action {
        UiAction::SelectPreset(preset) => {
            sim_config.preset = preset;
            sim_config.paused = false;
            load_writer.write(RequestLoadPreset(preset));
        }
        UiAction::SelectStiffness(stiff) => {
            sim_config.stiffness = stiff;
        }
        UiAction::TogglePause => {
            sim_config.paused = !sim_config.paused;
        }
        UiAction::Reset => {
            sim_config.paused = false;
            load_writer.write(RequestLoadPreset(sim_config.preset));
        }
        UiAction::ToggleGravity => {
            sim_config.gravity_enabled = !sim_config.gravity_enabled;
        }
    }
}

pub fn ui_interaction_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut sim_config: ResMut<SimConfig>,
    mut load_writer: MessageWriter<RequestLoadPreset>,
    mut buttons: Query<(Has<Pressed>, &Hovered, &UiAction, &mut BackgroundColor), With<Button>>,
    mut pause_text: Query<&mut Text, (With<PauseButtonLabel>, Without<GravityButtonLabel>)>,
    mut grav_text: Query<&mut Text, (With<GravityButtonLabel>, Without<PauseButtonLabel>)>,
) {
    if keys.just_pressed(KeyCode::Space) {
        sim_config.paused = !sim_config.paused;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        sim_config.paused = false;
        load_writer.write(RequestLoadPreset(sim_config.preset));
    }

    for (pressed, hovered, action, mut bg) in buttons.iter_mut() {
        let is_selected = match *action {
            UiAction::SelectPreset(p) => sim_config.preset == p,
            UiAction::SelectStiffness(s) => sim_config.stiffness == s,
            UiAction::TogglePause => sim_config.paused,
            UiAction::ToggleGravity => sim_config.gravity_enabled,
            UiAction::Reset => false,
        };

        *bg = BackgroundColor(if is_selected || pressed {
            BTN_ACTIVE
        } else if hovered.get() {
            BTN_HOVER
        } else {
            BTN_IDLE
        });
    }

    for mut t in pause_text.iter_mut() {
        t.0 = if sim_config.paused {
            "Resume".to_string()
        } else {
            "Pause".to_string()
        };
    }
    for mut t in grav_text.iter_mut() {
        t.0 = if sim_config.gravity_enabled {
            "Gravity: ON".to_string()
        } else {
            "Gravity: OFF".to_string()
        };
    }
}

pub fn update_stats_text_system(
    diagnostics: Res<DiagnosticsStore>,
    phys_data: Option<Res<PhysData>>,
    mut query: Query<&mut Text, With<StatsText>>,
) {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .unwrap_or(0.0);

    let Ok(mut text) = query.single_mut() else {
        return;
    };

    if let Some(pd) = phys_data {
        text.0 = format!("Verts: {}  |  Tets: {}  |  FPS: {:.0}", pd.nv, pd.nt, fps);
    }
}
