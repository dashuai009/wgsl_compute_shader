use std::io::Read;
use anyhow::{anyhow, bail, Context, Result};
use bevy::asset::{io::Reader, AssetLoader, LoadContext};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};


/// 可选：一个通用的“POD 数组资产”，方便以后扩展到 u32 / i32 / 自定义结构体等
#[derive(Asset, TypePath, Debug, Clone)]
pub struct PodArrayAsset<T: Pod + Zeroable + std::fmt::Debug + Send + Sync + 'static + bevy::prelude::TypePath> {
    pub data: Vec<T>,
}

/// 用这个宏快速生成不同元素类型的 Loader（裸数据版）
macro_rules! make_pod_raw_loader {
    ($name:ident, $ty:ty, [$($ext:literal),+ $(,)?]) => {
        pub struct $name;
        // #[async_trait::async_trait]
        impl AssetLoader for $name {
            type Asset = PodArrayAsset<$ty>;
            type Settings = ();
            type Error = anyhow::Error;

            async fn load(
                & self,
                reader: & mut dyn Reader,
                _settings: & Self::Settings,
                _ctx: & mut LoadContext<'_>,
            ) -> Result<Self::Asset, Self::Error> {
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes).await?;
                if bytes.len() % std::mem::size_of::<$ty>() != 0 {
                    bail!("byte length {} is not a multiple of {}", bytes.len(), std::mem::size_of::<$ty>());
                }
                let slice = bytemuck::try_cast_slice::<u8, $ty>(&bytes)
                    .map_err(|_| anyhow!("alignment/size mismatch when casting to slice"))?;
                Ok(PodArrayAsset::<$ty> { data: slice.to_vec() })
            }

            fn extensions(&self) -> &[&str] {
                &[$($ext),+]
            }
        }
    };
}

// 举例：u32 裸数据 loader（把 .u32bin 读成 Vec<u32>）
make_pod_raw_loader!(U32BinLoaderRaw, u32, ["u32bin"]);
make_pod_raw_loader!(Vec4U32BinLoaderRaw, [u32; 4], ["vec4u32bin"]);
make_pod_raw_loader!(Vec3F32BinLoaderRaw, [f32;3], ["vec3f32bin"]);


pub struct BinAssetPlugin;

impl Plugin for BinAssetPlugin {
    fn build(&self, app: &mut App) {
        app
            // 注册资产类型
            .init_asset::<PodArrayAsset<u32>>()
            .init_asset::<PodArrayAsset<[u32;4]>>()
            .init_asset::<PodArrayAsset<[f32;3]>>()
            // 注册不同 loader
            .register_asset_loader(U32BinLoaderRaw)
            .register_asset_loader(Vec4U32BinLoaderRaw)
            .register_asset_loader(Vec3F32BinLoaderRaw);
    }
}
