# WGSL XPBD Soft-Body Simulation

基于 **Rust + Bevy 0.20 + WGSL / WESL Shader** 的实时四面体有限元 / XPBD（Extended Position-Based Dynamics）软体物理实验项目，支持原生桌面端（Metal / Vulkan / DX12）与浏览器端 **WebAssembly + WebGPU** 同源运行。

- **在线体验 (WebGPU)**：<https://dashuai009.github.io/wgsl_compute_shader/>
- **从零实现技术博客**：<https://dashuai009.github.io/blog/058/>

## 核心特性

1. **完整 GPU 子步 XPBD 求解管线 (`assets/shaders/arap.wgsl`)**
   - 每帧拆分为 $N_{\text{sub}} = 10$ 个物理子步（$\Delta t = 1/600\,\text{s}$），每子步执行半隐式 Euler 重力预测、拉格朗日乘子清零、$K = 8$ 次无数据竞争并行 Jacobi 约束投影与 XPBD 速度摩擦更新。
   - **McAdams 3×3 Signed SVD 翻转安全应变能**：保证 $\det(U)=\det(V)=+1$ 且 $\sigma_3$ 携带 $\det(F)$ 符号，支持四面体极端挤压甚至发生内外翻转（$\det(F) < 0$）时平滑弹回静止体积。
   - **双约束耦合（ARAP + 静水体积保持）**：耦合共旋应变约束 $C_{\text{arap}} = \|\Sigma - \mathbf{1}\|_2$ 与体积保持约束 $C_{\text{vol}} = \det(F) - 1$，并将按顶点邻接度计算的保动量松弛权重 $\omega_t$ 同步作用于 $\delta\lambda$ 与 $\Delta x$，消除 Jacobi 松弛引入的人为刚度漂移。
   - **GPU 动态平滑法线与实时阴影**：子步完成后在 GPU 上对外表面三角形做面积加权法线原子累加与归一化，顶点着色器（含阴影预通道 `PREPASS_PIPELINE`）直接读取只读 Storage Buffer 渲染变形网格与动态阴影。
2. **开箱即用的 4 套轻量四面体资产 (`assets/models/`)**
   - **Bunny**：基于 BCC Isosurface Stuffing 生成的流形斯坦福兔子四面体网格（1,173 顶点 / 4,572 四面体），自由跌落回弹。
   - **Cube**：共形 6-四面体剖分弹性方块（343 顶点 / 1,296 四面体），带初始倾角跌落翻滚。
   - **Torus**：环向剖分弹性圆环（600 顶点 / 2,304 四面体），展示大旋转几何非线性形变。
   - **Beam**：左端面固定（`inv_mass = 0`）的悬臂梁基准（325 顶点 / 1,152 四面体），直观验证不同杨氏模量档位下的弯曲挠度与振荡。

## 本地运行与测试

```bash
# 运行原生桌面端应用
cargo run --release

# 运行无头 GPU 单元测试（Signed SVD 精度、零漂移静止姿态、翻转恢复、悬臂梁与地面碰撞）
cargo test --release

# 重新生成默认四面体资产（可选）
cargo run --release --example gen_assets
```

## 构建 WebGPU (WASM) 版本

需要支持 WebGPU 的现代浏览器（Chrome / Edge 113+、Safari 18+）以及与 `Cargo.lock` 匹配的 `wasm-bindgen-cli`：

```bash
cargo install wasm-bindgen-cli --version 0.2.130 --locked
./scripts/build-web.sh
python3 -m http.server 8080 --directory dist
```

随后在浏览器访问 `http://localhost:8080`。
