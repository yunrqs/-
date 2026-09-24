# FocusLens React 前端

唯一前端，保留摄像头预览、七类情绪概率、人脸框、EAR 曲线、眨眼统计、暂停、重置和截图。
后端位于项目根目录的 backend。走神分类仍由离线实验处理。

从项目根目录执行 `npm --prefix web-react ci`、`npm --prefix web-react run dev`。
构建：`npm --prefix web-react run build`；浏览器测试：`npm --prefix web-react run test:e2e`；算法单元测试：`npm --prefix web-react run test:unit`。

`src/pages/Monitor` 管理页面状态与交互；`src/features/monitoring` 包含请求、摄像头工具、类型和业务组件；
`src/components` 是通用组件，`src/styles` 为全局样式。App 仅组装页面。
`src/features/gaze` 是独立视线预测模块：`components` 提供面板、校准和落点显示，
`hooks` 管理状态与生命周期，`services` 调度模型与预测，`core` 保存特征和回归算法，`types.ts` 定义业务类型。
它与 monitoring 共用页面持有的摄像头，在浏览器内独立计算；模型与 WASM 位于 `public/mediapipe/face_mesh`。
仍使用 React 内置状态、ref、effect 和 fetch，没有新增运行时依赖。

详见 [开发与测试](../docs/development.md)、[部署](../docs/deployment.md)、[架构](../docs/architecture.md)。
