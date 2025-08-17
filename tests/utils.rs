use slint::wgpu_24::wgpu;
use slint::wgpu_24::wgpu::Device;

pub async fn get_wgpu_device_and_queue() -> (Device, wgpu::Queue) {

    // ---------- wgpu 初始化 ----------
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No adapter");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        )
        .await
        .expect("request_device failed");

    return (device, queue);
}