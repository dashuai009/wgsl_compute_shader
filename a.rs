use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
use bevy::render::render_resource::{PrimitiveTopology, VertexFormat};
use std::f32::consts::PI;

#[derive(Component)]
struct MyCube;


fn animate_mesh(
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<&Mesh3d, With<MyCube>>, // 只查询带有 MyCube 标记的实体
) {
    if let Some(mesh_handle) = query.iter().next() {
        if let Some(mesh) = meshes.get_mut(mesh_handle) {
            if let Some(VertexAttributeValues::Float32x3(positions)) =
                mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
            {
                // 计算一个基于时间的缩放值
                let scale = 1.0 + (time.elapsed_secs() * 2.0).sin() * 0.2;

                // 遍历并修改所有顶点
                for pos in positions.iter_mut() {
                    // 重置到原始位置 (这里假设原始立方体顶点在 [-0.5, 0.5] 范围内)
                    let original_x = pos[0].clamp(-0.5, 0.5);
                    let original_y = pos[1].clamp(-0.5, 0.5);
                    let original_z = pos[2].clamp(-0.5, 0.5);

                    // 应用缩放
                    pos[0] = original_x * scale;
                    pos[1] = original_y * scale;
                    pos[2] = original_z * scale;
                }
            }
        }
    }
}

/// 随便生成一个简单三角形，方便观察
fn spawn_demo(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    // 创建一个立方体 Mesh
    let mesh = meshes.add(Sphere::new(0.9).mesh());

    commands.spawn((
        Mesh3d(mesh),
        MeshMaterial3d(materials.add(StandardMaterial::from(Color::srgb(0.8, 0.7, 0.6)))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        MyCube, // 添加一个标记组件
    ));

    // 光源
    commands.spawn((
        PointLight {
            intensity: 1500.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    commands.spawn((
        Camera3d::default(),
       Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::new(0.0, -0.5, 0.0), Vec3::Y),
        PointLight { color: Color::linear_rgb(0.5, 0., 0.), ..default() },
    ));
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // .add_plugins( ScheduleRunnerPlugin::run_loop(std::time::Duration::from_secs_f64(
        //         1.0 / 60.0, // 设置刷新率为 60Hz
        //     ) ))
        .add_systems(Startup, spawn_demo)
        .add_systems(Update, animate_mesh) // ← 每帧改网格
        .run();
}
