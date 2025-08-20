

== Todo

权衡workgroup size的取值，，如果需要动态取值，需要设置shaderdefs

合并inv mass到pos的第四维上，是否性能更好
（需要同时修改arap、sim_pipeline两个storage buffer）

分别处理三角形数据和四面体数据，尝试优化加载第一帧的性能。

svd  `assert!(s_err <= tol_s, "singular values too far");`暂时被取消掉了

功能：
1. ui
2. 参数更新的逻辑
3. 模型切换
4. slint & bevy bug: windows最小化的时候崩溃
5. wasm for web