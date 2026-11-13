#[allow(dead_code)]
pub async fn get_wgpu_device_and_queue() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        })
        .await
    {
        Ok(a) => a,
        Err(_) => instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
                ..Default::default()
            })
            .await
            .expect("No hardware or software wgpu adapter available"),
    };

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("headless_test_device"),
            required_features: wgpu::Features::empty(),
            // Use standard WebGPU limits (maxStorageBuffersPerShaderStage = 8)
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("request_device failed")
}
