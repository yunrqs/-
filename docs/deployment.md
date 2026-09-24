# 构建与本地部署

从项目根目录运行 `npm --prefix web-react run build`，然后运行
`python -m backend.src.server`，访问 http://127.0.0.1:8001 。
也可使用 `python -m uvicorn backend.src.app:app --host 127.0.0.1 --port 8001`。
首次构建后应重启后端，以注册 assets 静态目录。

部署需要 backend 源码、Python 依赖、backend/models/emotion 权重和 web-react/dist。
离线训练数据与 outputs 不属于在线识别的必要资源。保持项目的相对目录布局。
当前识别会话为进程内单用户会话，只启动一个 worker；会话语义未在此次重构中改变。
