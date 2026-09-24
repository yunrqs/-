"""FastAPI application lifecycle and built frontend hosting."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config.paths import DIST
from .modules.monitoring import routes as monitoring
from .routes.index import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    monitoring.session.close()

app = FastAPI(title="FocusLens 本地识别接口", lifespan=lifespan)
app.include_router(api_router)

# 先 npm run build，再启动服务，即可只用 FastAPI 提供前端页面和接口。
# API 路由放在静态文件之前，避免静态文件处理器拦截 API 请求。
@app.get("/", include_in_schema=False)
def index():
    if not (DIST / "index.html").is_file():
        raise HTTPException(status_code=404, detail="请先执行 npm run build，或通过 Vite 的 5173 端口开发")
    return FileResponse(DIST / "index.html")


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

if (DIST / "mediapipe").is_dir():
    app.mount("/mediapipe", StaticFiles(directory=DIST / "mediapipe"), name="mediapipe")
