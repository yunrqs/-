# FocusLens 单页监测界面

该界面使用原生 HTML、CSS、JavaScript，并通过本机 Flask 服务调用项目现有的
FER-2013 情绪识别和 MediaPipe + EAR 眨眼检测模块。

在项目根目录运行：

```powershell
python -m pip install -r web\requirements.txt
python web\server.py
```

浏览器打开 `http://127.0.0.1:5000`，依次点击左侧“开启摄像头”和“开始识别”。

浏览器视频帧仅发送给当前计算机上的 `127.0.0.1` 服务，不会上传到外部服务器，后端也不
保存画面。截图功能只在浏览器端按用户操作保存当前帧。

当前页面展示七类情绪概率、情绪置信度、EAR、动态阈值、眨眼次数和眨眼频率。走神分类
没有接入，因为当前 MWDET 模型要求 WebGazer 屏幕注视特征，与摄像头模块输出不兼容。
