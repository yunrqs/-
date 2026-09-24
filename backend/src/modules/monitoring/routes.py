"""Monitoring HTTP endpoints; request and response contracts are unchanged."""
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from .service import AnalysisSession

router = APIRouter()
session = AnalysisSession()
MAX_FRAME_BYTES = 5 * 1024 * 1024

@router.get("/api/health")
def health():
    return {"ok": True, "ready": session.ready}


@router.post("/api/session/start")
def start_session():
    try:
        session.start()
        return {"ok": True, "ready": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"模型加载失败：{exc}") from exc


@router.post("/api/session/reset")
def reset_session():
    try:
        session.reset()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"统计重置失败：{exc}") from exc


# 普通 def 接口由 FastAPI 放在线程池执行，模型计算不会阻塞健康检查。
@router.post("/api/analyze")
def analyze(frame: UploadFile | None = File(default=None)):
    if frame is None:
        raise HTTPException(status_code=400, detail="请求中缺少 frame 图像")
    try:
        data = frame.file.read(MAX_FRAME_BYTES + 1)
    finally:
        frame.file.close()
    if not data:
        raise HTTPException(status_code=400, detail="上传的图像为空")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="图像不能超过 5 MB")
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise HTTPException(status_code=400, detail="无法解码摄像头帧") from exc
    if image is None:
        raise HTTPException(status_code=400, detail="无法解码摄像头帧")
    if not session.ready:
        raise HTTPException(status_code=409, detail="请先点击开始识别，加载模型")
    try:
        return session.analyze(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"分析失败：{exc}") from exc


