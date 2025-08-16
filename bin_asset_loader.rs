use std::io::Read;
use anyhow::{anyhow, bail, Context, Result};
use bevy::asset::{io::Reader, AssetLoader, LoadContext};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};


// 
// /// 你的目标资产：一串 f32
// #[derive(Asset, TypePath, Debug, Clone)]
// pub struct F32BinAsset {
//     pub data: Vec<f32>,
// }
// 
// /// 裸数据版：整个文件是连续的 f32（little-endian，IEEE 754）
// pub struct F32BinLoaderRaw;
// 
// /// 有头版：前 4 字节是元素个数（u32，小端），后面跟连续 f32
// pub struct F32BinLoaderWithHeader;
// 
// #[async_trait::async_trait]
// impl AssetLoader for F32BinLoaderRaw {
//     type Asset = F32BinAsset;
//     type Settings = ();
//     type Error = anyhow::Error;
// 
//     async fn load<'a>(
//         &'a self,
//         reader: &'a mut Reader,
//         _settings: &'a Self::Settings,
//         _ctx: &'a mut LoadContext<'_>,
//     ) -> Result<Self::Asset> {
//         let mut bytes = Vec::new();
//         reader.read_to_end(&mut bytes).await?;
//         if bytes.len() % 4 != 0 {
//             bail!("byte length {} is not a multiple of 4 for f32", bytes.len());
//         }
//         // bytemuck 不做字节序转换；我们默认文件是小端（几乎所有桌面/移动平台都是小端）
//         let slice = bytemuck::try_cast_slice::<u8, f32>(&bytes)
//             .map_err(|_| anyhow!("alignment/size mismatch when casting to f32 slice"))?;
//         Ok(F32BinAsset { data: slice.to_vec() })
//     }
// 
//     fn extensions(&self) -> &[&str] {
//         // 你可以用更专用的后缀以避免和别的 .bin 冲突，例如 "f32bin"
//         &["bin", "f32bin"]
//     }
// }
// 
// #[async_trait::async_trait]
// impl AssetLoader for F32BinLoaderWithHeader {
//     type Asset = F32BinAsset;
//     type Settings = ();
//     type Error = anyhow::Error;
// 
//     async fn load<'a>(
//         &'a self,
//         reader: &'a mut Reader,
//         _settings: &'a Self::Settings,
//         _ctx: &'a mut LoadContext<'_>,
//     ) -> Result<Self::Asset> {
//         use std::io::Cursor;
//         use byteorder::{LittleEndian, ReadBytesExt};
// 
//         let mut bytes = Vec::new();
//         reader.read_to_end(&mut bytes).await?;
// 
//         let mut cur = Cursor::new(&bytes);
//         let count: u32 = cur
//             .read_u32::<LittleEndian>()
//             .context("failed to read count header (u32)")?;
// 
//         let data_bytes = &bytes[4..];
//         if data_bytes.len() != (count as usize) * 4 {
//             bail!(
//                 "payload size mismatch: header count={} but payload_bytes={} (expected {})",
//                 count,
//                 data_bytes.len(),
//                 count as usize * 4
//             );
//         }
// 
//         // 仍然用 bytemuck 做 reinterpret，再 to_vec 复制一份
//         let slice = bytemuck::try_cast_slice::<u8, f32>(data_bytes)
//             .map_err(|_| anyhow!("alignment/size mismatch in payload"))?;
// 
//         Ok(F32BinAsset { data: slice.to_vec() })
//     }
// 
//     fn extensions(&self) -> &[&str] {
//         &["f32binh"] // 给“有头版”单独后缀，更清晰
//     }
// }

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
