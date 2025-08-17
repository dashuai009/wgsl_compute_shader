use super::utils;


async fn run() {
    let device = utils::get_wgpu_device().await;
    
}



#[cfg(test)]
mod test {
    use super::run;

    #[test]
    fn test_arap() {
        pollster::block_on(run());
    }
}
