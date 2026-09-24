"""测试 HTTP 契约与错误处理，不加载大型模型、不访问摄像头。"""
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.src.modules.monitoring import routes as main
from backend.src.app import app
from backend.src.config.paths import DIST


class FakeSession:
    def __init__(self):
        self.ready = False
        self.count = 0

    def start(self):
        self.ready = True

    def reset(self):
        self.count = 0

    def analyze(self, frame):
        self.count += 1
        return {"frame_width": frame.shape[1], "frame_height": frame.shape[0], "blink_count": self.count}

    def close(self):
        self.ready = False


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.patch = patch.object(main, "session", self.session)
        self.patch.start()
        self.client = TestClient(app)
        _, encoded = cv2.imencode('.jpg', np.zeros((48, 64, 3), dtype=np.uint8))
        self.jpeg = encoded.tobytes()

    def tearDown(self):
        self.client.close()
        self.patch.stop()

    def upload(self, data):
        return self.client.post('/api/analyze', files={"frame": ("frame.jpg", data, "image/jpeg")})

    def test_start_analyze_reset(self):
        self.assertEqual(self.client.get('/api/health').json(), {"ok": True, "ready": False})
        self.assertEqual(self.upload(self.jpeg).status_code, 409)
        self.assertEqual(self.client.post('/api/session/start').status_code, 200)
        result = self.upload(self.jpeg)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), {"frame_width": 64, "frame_height": 48, "blink_count": 1})
        self.assertEqual(self.client.post('/api/session/reset').status_code, 200)
        self.assertEqual(self.session.count, 0)

    def test_reject_invalid_uploads(self):
        self.assertEqual(self.client.post('/api/analyze').status_code, 400)
        for data in [b'', b'not an image']:
            self.assertEqual(self.upload(data).status_code, 400)
        self.assertEqual(self.upload(b'x' * (main.MAX_FRAME_BYTES + 1)).status_code, 413)

    def test_model_failure_is_readable(self):
        with patch.object(self.session, 'start', side_effect=RuntimeError('模型文件不存在')):
            response = self.client.post('/api/session/start')
        self.assertEqual(response.status_code, 500)
        self.assertIn('模型文件不存在', response.json()['detail'])

    def test_reset_failure_is_not_reported_as_success(self):
        with patch.object(self.session, 'reset', side_effect=RuntimeError('重置失败')):
            response = self.client.post('/api/session/reset')
        self.assertEqual(response.status_code, 500)
        self.assertIn('重置失败', response.json()['detail'])

    def test_built_page_and_api_docs(self):
        if (DIST / 'index.html').exists():
            response = self.client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('/assets/', response.text)
        self.assertEqual(self.client.get('/openapi.json').status_code, 200)
        self.assertEqual(self.client.get('/api/not-found').status_code, 404)

    def test_gaze_runtime_is_served_after_build(self):
        if not (DIST / 'mediapipe').exists():
            self.skipTest('Frontend has not been built with gaze resources')
        for name in ['face_mesh.js', 'face_mesh.binarypb', 'face_mesh_solution_wasm_bin.wasm']:
            response = self.client.get('/mediapipe/face_mesh/' + name)
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.content), 100)


if __name__ == '__main__':
    unittest.main()
