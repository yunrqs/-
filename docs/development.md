# 开发与测试

以下命令均从项目根目录执行。Python 推荐 3.10 或 3.11，Node.js 要求见 web-react/package.json。

```powershell
python -m venv --system-site-packages backend/.venv
./backend/.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
./backend/.venv/Scripts/python.exe -m pip install -r backend/src/mind_wandering/requirements.txt
npm --prefix web-react ci
```

若需要隔离环境，创建虚拟环境时去掉 `--system-site-packages`。
本地已有 `web-react/.venv` 可继续使用，只需在根目录调用其 Python；虚拟环境不直接移动，避免内置绝对路径失效。

Windows 新电脑可双击根目录 `setup.cmd`：自动创建隔离的 `backend/.venv`，安装
`backend/requirements.txt` 和前端依赖。需预先安装 64 位 Python 3.10+（建议 3.11）及
Node.js 22.12+，完整解压项目并包含情绪模型权重。环境不能跨电脑复制，若已有环境不可用，
重命名 `backend/.venv` 后重新运行脚本即可，脚本不会删除旧环境。

双击 `start.cmd` 也会自动检查和安装依赖，在后台启动后端并等待健康检查成功，
再启动前端并打开浏览器。依赖安装成功后记录清单指纹，后续启动复用；清单变化或安装中断会重试。
`setup.cmd -Reinstall` 强制重新执行安装（不删除 Python 环境，也不强制升级已满足要求的包）。
开发测试与离线训练依赖仍需按本页命令另行安装。旧 `web-react/.venv` 仍可手动使用，
一键脚本统一使用 `backend/.venv`。

已有后端会直接复用，后端日志保存在 `backend/outputs/dev/`。
正常按 `Ctrl+C` 退出前端时，脚本会关闭本次启动的后端；不会关闭复用的后端。
直接关闭终端窗口可能留下后台进程，建议使用 `Ctrl+C`。
若前端已经运行，只需执行 `./start.cmd -BackendOnly` 补启动后端并保留在后台，
此模式只安装后端依赖，无需 Node.js。

也可以分别在两个终端运行（使用已有环境时将下面的 `backend/.venv` 替换为 `web-react/.venv`）：

```powershell
./backend/.venv/Scripts/python.exe -m backend.src.server
npm --prefix web-react run dev
```

访问 http://127.0.0.1:5173 ，Vite 代理 `/api` 到 8001 端口。
模型权重默认读取 `backend/models/emotion/fer2013_best_model.pth`。
API 文档位于 http://127.0.0.1:8001/docs 。

```powershell
npm --prefix web-react run build
./backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests -v
./backend/.venv/Scripts/python.exe -m unittest discover -s backend/src/mind_wandering/tests -v
npm --prefix web-react run test:e2e
```

浏览器测试使用 Edge、Canvas 合成视频流和模拟接口；视线测试另包含真实本地
MediaPipe 加载与空白帧推理。合成流避免本机虚拟摄像头驱动间歇性不出帧。
运行前关闭占用 5173 端口的开发服务。回归算法测试：

```powershell
npm --prefix web-react run test:unit
```
独立算法也从项目根目录通过包入口运行：

```powershell
python -m backend.src.algorithms.blink_detection.main --help
python -m backend.src.algorithms.emotion.use_model
python -m backend.src.mind_wandering.training.train --help
```
