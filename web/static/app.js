const emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
const emotionMeta = {
  angry: ["愤怒", "!"], disgust: ["厌恶", "×"], fear: ["恐惧", "△"],
  happy: ["愉悦", "+"], sad: ["悲伤", "–"], surprise: ["惊讶", "○"], neutral: ["中性", "·"]
};
const $ = (id) => document.getElementById(id);
const els = Object.fromEntries([
  "cameraBtn", "analysisBtn", "pauseBtn", "resetBtn", "captureBtn", "camera", "overlay",
  "placeholder", "serviceStatus", "clock", "liveBadge", "resolution", "analysisRate",
  "faceState", "confidenceLabel", "emotionGlyph", "emotionName", "emotionCn", "earValue",
  "eyeState", "blinkTotal", "blinkRate", "thresholdValue", "validRatio", "earChart", "toast",
  "captureCanvas", "emotionBars"
].map(id => [id, $(id)]));

let stream = null;
let analyzing = false;
let requestPending = false;
let analysisTimer = null;
let toastTimer = null;
let frameTimes = [];
let earHistory = [];
let thresholdHistory = [];

function buildEmotionBars() {
  els.emotionBars.innerHTML = emotions.map(name => `
    <div class="emotion-row" data-emotion="${name}">
      <label>${name}</label><div class="bar-track"><i></i></div><output>0.0%</output>
    </div>`).join("");
}

function notify(message, error = false) {
  clearTimeout(toastTimer);
  els.toast.textContent = message;
  els.toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => els.toast.className = "toast", 3200);
}

async function checkService() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    els.serviceStatus.className = "status-pill online";
    els.serviceStatus.innerHTML = `<i></i>${data.ready ? "识别服务就绪" : "服务在线 · 模型待加载"}`;
  } catch {
    els.serviceStatus.className = "status-pill offline";
    els.serviceStatus.innerHTML = "<i></i>识别服务离线";
  }
}

async function startCamera() {
  if (stream) {
    stopCamera();
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
    els.camera.srcObject = stream;
    await els.camera.play();
    const settings = stream.getVideoTracks()[0].getSettings();
    els.resolution.textContent = `${settings.width || els.camera.videoWidth}×${settings.height || els.camera.videoHeight}`;
    els.placeholder.classList.add("hidden");
    els.cameraBtn.querySelector("span:last-child").textContent = "关闭摄像头";
    els.cameraBtn.classList.remove("primary");
    els.analysisBtn.disabled = false;
    els.captureBtn.disabled = false;
    resizeCanvases();
    notify("摄像头已开启，画面只在本机处理。 ");
  } catch (error) {
    notify(`无法开启摄像头：${error.message}`, true);
  }
}

function stopCamera() {
  stopAnalysis();
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null;
  els.camera.srcObject = null;
  els.placeholder.classList.remove("hidden");
  els.cameraBtn.querySelector("span:last-child").textContent = "开启摄像头";
  els.cameraBtn.classList.add("primary");
  els.analysisBtn.disabled = true;
  els.captureBtn.disabled = true;
  els.resolution.textContent = "—";
  clearOverlay();
}

async function startAnalysis() {
  if (!stream || analyzing) return;
  els.analysisBtn.disabled = true;
  els.analysisBtn.querySelector("span:last-child").textContent = "正在加载模型";
  try {
    const response = await fetch("/api/session/start", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "模型加载失败");
    analyzing = true;
    els.analysisBtn.classList.add("active");
    els.analysisBtn.querySelector("span:last-child").textContent = "识别运行中";
    els.pauseBtn.disabled = false;
    els.liveBadge.className = "live-badge live";
    els.liveBadge.innerHTML = "<i></i><span>实时分析</span>";
    els.serviceStatus.className = "status-pill online";
    els.serviceStatus.innerHTML = "<i></i>识别服务就绪";
    analysisTimer = setInterval(analyzeFrame, 220);
    notify("情绪识别与眨眼检测已启动。 ");
  } catch (error) {
    els.analysisBtn.disabled = false;
    els.analysisBtn.querySelector("span:last-child").textContent = "开始识别";
    notify(error.message, true);
  }
}

function stopAnalysis() {
  analyzing = false;
  clearInterval(analysisTimer);
  analysisTimer = null;
  els.analysisBtn.disabled = !stream;
  els.analysisBtn.classList.remove("active");
  els.analysisBtn.querySelector("span:last-child").textContent = "开始识别";
  els.pauseBtn.disabled = true;
  els.liveBadge.className = "live-badge";
  els.liveBadge.innerHTML = "<i></i><span>已暂停</span>";
  els.analysisRate.textContent = "0.0";
}

function resizeCanvases() {
  const width = 640;
  const ratio = els.camera.videoHeight && els.camera.videoWidth ? els.camera.videoHeight / els.camera.videoWidth : 9 / 16;
  els.overlay.width = width;
  els.overlay.height = Math.round(width * ratio);
}

async function analyzeFrame() {
  if (!analyzing || requestPending || !els.camera.videoWidth) return;
  requestPending = true;
  const canvas = els.captureCanvas;
  canvas.width = 640;
  canvas.height = Math.round(640 * els.camera.videoHeight / els.camera.videoWidth);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(els.camera, 0, 0, canvas.width, canvas.height);
  ctx.restore();
  try {
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", .78));
    const form = new FormData();
    form.append("frame", blob, "frame.jpg");
    const started = performance.now();
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "分析失败");
    frameTimes.push(performance.now() - started);
    if (frameTimes.length > 20) frameTimes.shift();
    const meanMs = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    els.analysisRate.textContent = (1000 / meanMs).toFixed(1);
    updateResult(result);
  } catch (error) {
    notify(error.message, true);
    stopAnalysis();
  } finally {
    requestPending = false;
  }
}

function updateResult(result) {
  els.faceState.textContent = result.face_detected ? "已锁定" : "未检测";
  els.faceState.style.color = result.face_detected ? "var(--cyan)" : "";
  els.earValue.textContent = number(result.ear, 3);
  els.eyeState.textContent = result.eye_state || "—";
  els.blinkTotal.textContent = result.blink_count ?? 0;
  els.blinkRate.textContent = number(result.blink_rate_per_min, 1, "0.0");
  els.thresholdValue.textContent = number(result.threshold, 3);
  els.validRatio.textContent = Math.round((result.valid_face_ratio || 0) * 100);
  updateEmotion(result.emotion, result.emotion_confidence, result.emotion_probabilities || {});
  drawFace(result.face_box, result.frame_width, result.frame_height, result.blink_detected);
  earHistory.push(result.ear ?? null);
  thresholdHistory.push(result.threshold ?? null);
  if (earHistory.length > 80) { earHistory.shift(); thresholdHistory.shift(); }
  drawChart();
}

function updateEmotion(name, confidence, probabilities) {
  const active = emotions.includes(name) ? name : null;
  els.emotionName.textContent = active || "未识别";
  els.emotionCn.textContent = active ? emotionMeta[active][0] : "请保持面部位于画面中央";
  els.emotionGlyph.textContent = active ? emotionMeta[active][1] : "—";
  els.confidenceLabel.textContent = `置信度 ${active ? (confidence * 100).toFixed(1) + "%" : "—"}`;
  document.querySelectorAll(".emotion-row").forEach(row => {
    const value = probabilities[row.dataset.emotion] || 0;
    row.querySelector("i").style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
    row.querySelector("output").textContent = `${(value * 100).toFixed(1)}%`;
    row.classList.toggle("dominant", row.dataset.emotion === active);
  });
}

function drawFace(box, frameWidth, frameHeight, blink) {
  const ctx = els.overlay.getContext("2d");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  if (!box || !frameWidth || !frameHeight) return;
  const sx = els.overlay.width / frameWidth, sy = els.overlay.height / frameHeight;
  const [x, y, w, h] = box;
  ctx.strokeStyle = blink ? "#ffbd68" : "#42e8d1";
  ctx.lineWidth = 2;
  ctx.setLineDash([12, 7]);
  ctx.strokeRect(x * sx, y * sy, w * sx, h * sy);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(7,16,29,.82)";
  ctx.fillRect(x * sx, Math.max(0, y * sy - 22), 105, 19);
  ctx.fillStyle = blink ? "#ffbd68" : "#42e8d1";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(blink ? "BLINK DETECTED" : "FACE TRACKED", x * sx + 7, Math.max(13, y * sy - 9));
}

function clearOverlay() { els.overlay.getContext("2d").clearRect(0, 0, els.overlay.width, els.overlay.height); }

function drawChart() {
  const canvas = els.earChart;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.strokeStyle = "rgba(151,175,204,.09)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { ctx.beginPath(); ctx.moveTo(0, i * h / 4); ctx.lineTo(w, i * h / 4); ctx.stroke(); }
  drawSeries(ctx, thresholdHistory, w, h, "#ffbd68", 1);
  drawSeries(ctx, earHistory, w, h, "#42e8d1", 2);
}

function drawSeries(ctx, values, w, h, color, width) {
  if (values.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round";
  values.forEach((value, index) => {
    if (value == null) return;
    const x = index * w / Math.max(1, values.length - 1);
    const y = h - Math.max(0, Math.min(.5, value)) / .5 * h;
    index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

async function resetStats() {
  try { await fetch("/api/session/reset", { method: "POST" }); } catch {}
  earHistory = []; thresholdHistory = []; frameTimes = [];
  els.blinkTotal.textContent = "0"; els.blinkRate.textContent = "0.0"; els.validRatio.textContent = "0";
  updateEmotion(null, 0, {}); drawChart(); notify("本次会话统计已重置。 ");
}

function saveCapture() {
  if (!stream) return;
  const canvas = els.captureCanvas;
  canvas.width = els.camera.videoWidth; canvas.height = els.camera.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(els.camera, 0, 0);
  const link = document.createElement("a");
  link.download = `focuslens-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`;
  link.href = canvas.toDataURL("image/jpeg", .92); link.click();
  notify("当前画面已保存。 ");
}

function number(value, digits, fallback = "—") { return Number.isFinite(value) ? value.toFixed(digits) : fallback; }

els.cameraBtn.addEventListener("click", startCamera);
els.analysisBtn.addEventListener("click", startAnalysis);
els.pauseBtn.addEventListener("click", stopAnalysis);
els.resetBtn.addEventListener("click", resetStats);
els.captureBtn.addEventListener("click", saveCapture);
window.addEventListener("resize", () => { resizeCanvases(); drawChart(); });
window.addEventListener("beforeunload", stopCamera);

buildEmotionBars();
drawChart();
checkService();
setInterval(() => els.clock.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false }), 1000);
