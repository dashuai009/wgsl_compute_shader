use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;

mod utils;


async fn run() {
    let (device, queue) = utils::get_wgpu_device_and_queue().await;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.run();
    let buffers = app.world_mut().resource_mut::<Assets<ShaderStorageBuffer>>();
}



#[cfg(test)]
mod test {
    use super::run;

    #[test]
    fn test_arap() {
        pollster::block_on(run());
    }
}
