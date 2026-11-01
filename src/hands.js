
// Webcam + MediaPipe HandLandmarker
// GitHub Pages compatible version

const MEDIAPIPE_VERSION = '1.0.1';

const MEDIAPIPE_CDN =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;

const VISION = `${MEDIAPIPE_CDN}/vision_bundle.mjs`;
const WASM = `${MEDIAPIPE_CDN}/wasm`;

const MODEL = './models/hand_landmarker.task';

export class HandCamera {
  constructor(video) {
    this.video = video;
    this.landmarker = null;
    this.stream = null;
    this.running = false;
    this.delegate = null;
    this.lastVideoTime = -1;
    this.lastTs = 0;
    this.frames = 0;
    this.fps = 0;
    this.detectMs = 0;
    this._n = 0;
    this._t = performance.now();
    this.newFrame = false;
  }

  async start(onStatus = () => {}) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Camera requires HTTPS or localhost.');
    }

    onStatus('Requesting camera…');

    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
        frameRate: { ideal: 30 }
      },
      audio: false
    });

    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;

    await this.video.play();

    onStatus('Loading hand tracker…');

    const { FilesetResolver, HandLandmarker } =
      await import(VISION);

    const fileset =
      await FilesetResolver.forVisionTasks(WASM);

    const make = (delegate) =>
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: {
          modelAssetPath: MODEL,
          delegate
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

    try {
      this.landmarker = await make('GPU');
      this.delegate = 'GPU';
    } catch (error) {
      console.warn('GPU failed. Trying CPU.', error);
      this.landmarker = await make('CPU');
      this.delegate = 'CPU';
    }

    this.running = true;
    onStatus('');
  }

  poll() {
    const v = this.video;

    if (
      !this.running ||
      v.readyState < 2 ||
      v.currentTime === this.lastVideoTime
    ) {
      return undefined;
    }

    this.lastVideoTime = v.currentTime;
    this.newFrame = true;

    const ts = Math.max(
      Math.round(performance.now()),
      this.lastTs + 1
    );

    this.lastTs = ts;

    const t0 = performance.now();

    const res = this.landmarker.detectForVideo(v, ts);

    this.detectMs = performance.now() - t0;

    this.frames++;
    this._n++;

    const now = performance.now();

    if (now - this._t > 1000) {
      this.fps = (this._n * 1000) / (now - this._t);
      this._n = 0;
      this._t = now;
    }

    return (res.landmarks || []).map(
      lm => lm.map(p => [1 - p.x, p.y])
    );
  }

  stop() {
    this.running = false;

    this.stream?.getTracks().forEach(
      track => track.stop()
    );

    this.stream = null;

    this.landmarker?.close?.();
    this.landmarker = null;
  }
}
