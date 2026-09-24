# 架构与迁移映射

```text
.
├── web-react/                       # React + TypeScript 前端
│   ├── src/
│   │   ├── pages/Monitor/           # 监测页面和交互流程
│   │   ├── features/monitoring/     # 请求、摄像头、类型、业务组件
│   │   ├── features/gaze/           # 浏览器内视线预测
│   │   │   ├── components/         # 面板、校准验证、落点标记
│   │   │   ├── hooks/              # useGazePrediction.ts
│   │   │   ├── services/           # gazeService.ts：推理调度和资源释放
│   │   │   ├── core/               # 特征、回归算法及来源许可证
│   │   │   └── types.ts            # 视线业务类型
│   │   ├── components/             # 通用图标
│   │   ├── styles/                 # 全局样式
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── tests/e2e/                  # Playwright 浏览器测试
│   ├── tests/unit/                 # 视线回归算法测试
│   ├── public/mediapipe/face_mesh/  # 浏览器模型与 WASM 运行资源
│   └── package.json                # Vite / TypeScript 构建配置同级
├── backend/
│   ├── src/
│   │   ├── config/paths.py          # 统一资源路径
│   │   ├── modules/monitoring/      # routes.py 接口、service.py 会话
│   │   ├── algorithms/
│   │   │   ├── emotion/            # 情绪模型训练与推理
│   │   │   └── blink_detection/    # EAR 和眨眼检测
│   │   ├── mind_wandering/         # 离线数据、特征、训练、推理及测试
│   │   ├── routes/index.py         # API 路由注册
│   │   ├── app.py                  # 生命周期与静态页面托管
│   │   └── server.py               # 服务入口
│   ├── models/emotion/             # 模型权重
│   ├── outputs/                    # 实验结果、模型和训练图
│   ├── tests/unit/                 # 眨眼单元测试
│   ├── tests/integration/          # HTTP 契约测试
│   ├── requirements.txt            # 在线识别依赖
│   └── requirements-dev.txt        # 接口测试依赖
├── data/
│   ├── raw/                        # fer2013、mwdet、eegmeditation
│   └── processed/                  # 特征 CSV
├── docs/                           # 架构、接口、开发、部署说明
│   ├── papers/                     # 论文
│   └── specifications/             # 眨眼与视线模块规格说明
├── .gitignore
└── README.md

```

依赖方向：前端页面 → monitoring 功能；HTTP 路由 → 识别服务 → algorithms。
离线 mind_wandering 也调用 algorithms；算法不依赖 FastAPI 或前端。
config/paths.py 统一定位数据、模型、输出和前端构建目录。

视线预测依赖方向：Monitor 页面 → gaze/components → gaze/hooks → gaze/services → gaze/core。
Monitor 持有摄像头，视线功能与 monitoring 共用 video，分别控制自己的分析流程。
gaze 在浏览器内完成眼部特征提取、校准和回归预测，不调用 Python 推理接口，也未接入 MWDET 走神分类器。
浏览器模型和 WASM 保留在 public/mediapipe/face_mesh，Vite 将它们复制到 dist；FastAPI 仅在构建部署时托管这些静态资源。
backend/src/algorithms 和 backend/models 继续用于 Python 算法及其权重。

| 原位置 | 新位置 |
| --- | --- |
| web-react/backend/main.py | backend/src/app.py、routes/index.py、modules/monitoring/routes.py |
| web-react/backend/analysis.py | backend/src/modules/monitoring/service.py |
| emo/emotion | backend/src/algorithms/emotion |
| zhayan/blink_detection | backend/src/algorithms/blink_detection |
| mind_wandering | backend/src/mind_wandering |
| emo/models | backend/models/emotion |
| outputs | backend/outputs |
| emo/emotion/FER-2013 | data/raw/fer2013 |
| web-react/src/App.tsx 的页面实现 | web-react/src/pages/Monitor/index.tsx |
| web | 删除，唯一前端为 web-react |
| web-react/src/features/gaze/useGazePrediction.ts | web-react/src/features/gaze/hooks/useGazePrediction.ts |
| web-react/src/features/gaze/service.ts | web-react/src/features/gaze/services/gazeService.ts |
| web-react/tests/gaze-regression.test.mjs | web-react/tests/unit/gaze-regression.test.mjs |

算法计算和训练参数保持原样；调整仅涉及模块组织、包导入、资源路径与入口配置。
Python 采用从根目录运行 `python -m backend.src...` 的包入口，避免 sys.path 注入和同名 config 冲突。
源码、数据、权重和生成产物分开管理。保留已有本地虚拟环境和前端依赖目录，不将缓存视为源码。
