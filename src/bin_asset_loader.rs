use anyhow::{Result, bail};
use bevy::asset::{AssetLoader, LoadContext, io::Reader};
use bevy::prelude::*;
use bytemuck::{Pod, Zeroable};

#[derive(Asset, TypePath, Debug, Clone)]
pub struct PodArrayAsset<
    T: Pod + Zeroable + std::fmt::Debug + Send + Sync + 'static + bevy::prelude::TypePath,
> {
    pub data: Vec<T>,
}

macro_rules! make_pod_raw_loader {
    ($name:ident, $ty:ty, [$($ext:literal),+ $(,)?]) => {
        #[derive(Default, TypePath)]
        pub struct $name;

        impl AssetLoader for $name {
            type Asset = PodArrayAsset<$ty>;
            type Settings = ();
            type Error = anyhow::Error;

            async fn load(
                &self,
                reader: &mut dyn Reader,
                _settings: &Self::Settings,
                _ctx: &mut LoadContext<'_>,
            ) -> Result<Self::Asset, Self::Error> {
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes).await?;
                let elem_size = std::mem::size_of::<$ty>();
                if !bytes.len().is_multiple_of(elem_size) {
                    bail!(
                        "byte length {} is not a multiple of {}",
                        bytes.len(),
                        elem_size
                    );
                }
                let data: Vec<$ty> = bytemuck::pod_collect_to_vec(&bytes);
                Ok(PodArrayAsset::<$ty> { data })
            }

            fn extensions(&self) -> &[&str] {
                &[$($ext),+]
            }
        }
    };
}

make_pod_raw_loader!(U32BinLoaderRaw, u32, ["u32bin"]);
make_pod_raw_loader!(Vec4U32BinLoaderRaw, [u32; 4], ["vec4u32bin"]);
make_pod_raw_loader!(Vec3F32BinLoaderRaw, [f32; 3], ["vec3f32bin"]);

pub struct BinAssetPlugin;

impl Plugin for BinAssetPlugin {
    fn build(&self, app: &mut App) {
        app.init_asset::<PodArrayAsset<u32>>()
            .init_asset::<PodArrayAsset<[u32; 4]>>()
            .init_asset::<PodArrayAsset<[f32; 3]>>()
            .init_asset_loader::<U32BinLoaderRaw>()
            .init_asset_loader::<Vec4U32BinLoaderRaw>()
            .init_asset_loader::<Vec3F32BinLoaderRaw>();
    }
}
