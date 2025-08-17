mod utils;


async fn run() {
    let (device, queue) = utils::get_wgpu_device_and_queue().await;

}



#[cfg(test)]
mod test {
    use super::run;

    #[test]
    fn test_arap() {
        pollster::block_on(run());
    }
}
