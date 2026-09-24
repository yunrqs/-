// 预览已经通过 CSS 镜像；上传和截图也镜像，让人脸框与画面一致。
export async function playCamera(video: HTMLVideoElement): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([video.play(), new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error('摄像头未返回画面，请重新开启或检查是否被其他程序占用')), 15000);
    })]);
  } finally { clearTimeout(timer); }
}

export function captureVideo(video: HTMLVideoElement, width: number): HTMLCanvasElement {
  if (!video.videoWidth || !video.videoHeight) throw new Error('摄像头画面尚未就绪');
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = Math.round(width * video.videoHeight / video.videoWidth);
  const context = canvas.getContext('2d');
  if (!context) throw new Error('浏览器不支持 Canvas');
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

export function toJpeg(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('无法生成摄像头图像'));
    }, 'image/jpeg', 0.78);
  });
}
