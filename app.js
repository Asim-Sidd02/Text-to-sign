import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";
import { FilesetResolver, PoseLandmarker, HandLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODELS = {
  pose: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
  hand: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  face: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
};
const DEFAULT_AVATAR = "models/avatar.vrm";
const FORMAT = 3;
const FACE_KEYS = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel", "mouthStretchLeft", "mouthStretchRight", "browInnerUp", "browDownLeft", "browDownRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight", "cheekPuff"];
const FINGERS = [
  ["Thumb", ["Metacarpal", "Proximal", "Distal"], [1, 2, 3, 4]],
  ["Index", ["Proximal", "Intermediate", "Distal"], [5, 6, 7, 8]],
  ["Middle", ["Proximal", "Intermediate", "Distal"], [9, 10, 11, 12]],
  ["Ring", ["Proximal", "Intermediate", "Distal"], [13, 14, 15, 16]],
  ["Little", ["Proximal", "Intermediate", "Distal"], [17, 18, 19, 20]]
];
const SIDES = {
  left: { shoulder: 12, elbow: 14, wrist: 16, hand: "left", s: "ls", e: "le", w: "lw", sign: 1 },
  right: { shoulder: 11, elbow: 13, wrist: 15, hand: "right", s: "rs", e: "re", w: "rw", sign: -1 }
};
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const POSE_LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [7, 8]];
const GAP_MS = 320;
const HOLD_FRAMES = 8;
const LIVE_LERP = 0.55;
const PLAY_LERP = 0.8;
const VIS_MIN = 0.35;

const $ = id => document.getElementById(id);
const el = {
  video: $("video"), overlay: $("overlay"), avatar: $("avatar"), status: $("status"), countdown: $("countdown"),
  recBadge: $("recBadge"), trackInfo: $("trackInfo"), camBtn: $("camBtn"), recBtn: $("recBtn"), stopBtn: $("stopBtn"),
  wordInput: $("wordInput"), liveChk: $("liveChk"), textInput: $("textInput"), playBtn: $("playBtn"), haltBtn: $("haltBtn"),
  speed: $("speed"), speedVal: $("speedVal"), loopChk: $("loopChk"), modelInput: $("modelInput"), avatarLoading: $("avatarLoading"),
  resetViewBtn: $("resetViewBtn"), sequence: $("sequence"), caption: $("caption"), library: $("library"), preview: $("preview"),
  exportBtn: $("exportBtn"), importInput: $("importInput")
};
const octx = el.overlay.getContext("2d");

const state = {
  models: null, stream: null, lastVT: -1, lastTs: 0, liveFrame: null, miss: { left: 0, right: 0 }, filters: {},
  recording: false, rec: [], recStart: 0, recWord: "", recorder: null, chunks: [],
  playing: false, timeline: null, playT: 0, lastNow: 0, activeSeg: -1, library: new Map(), vrm: null, rest: null, view: null
};

const setStatus = (msg, kind = "") => {
  el.status.textContent = msg;
  el.status.className = `status ${kind}`;
};
const clamp01 = v => Math.min(1, Math.max(0, v));
const r4 = v => Math.round(v * 10000) / 10000;
const sleep = ms => new Promise(res => setTimeout(res, ms));
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
const normalizeKey = s => s.toLowerCase().replace(/[^a-z0-9\s']/g, " ").replace(/\s+/g, " ").trim();
const dist2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
const add = (a, b) => a.map((v, i) => r4(v + b[i]));
const mirror = p => [r4(-p.x), r4(-p.y), r4(-p.z)];
const V = a => new THREE.Vector3(a[0], a[1], a[2]);
const I = () => new THREE.Quaternion();

const IDLE_BODY = (() => {
  const ls = [0.18, 0.45, 0];
  const rs = [-0.18, 0.45, 0];
  const le = add(ls, [0.04, -0.27, -0.01]);
  const re = add(rs, [-0.04, -0.27, -0.01]);
  return { ls, rs, le, re, lw: add(le, [0.01, -0.25, 0.04]), rw: add(re, [-0.01, -0.25, 0.04]), lh: [0.1, 0, 0], rh: [-0.1, 0, 0] };
})();
const IDLE = { t: 0, body: IDLE_BODY, left: null, right: null, head: [0, 0, 0, 1], face: Object.fromEntries(FACE_KEYS.map(k => [k, 0])) };

const oneEuro = (minCutoff, beta) => {
  const s = { x: null, dx: 0, t: 0 };
  const alpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
  return (x, t) => {
    const dt = Math.max((t - s.t) / 1000, 1e-3);
    const dx = s.x === null ? 0 : (x - s.x) / dt;
    const edx = s.dx + alpha(1, dt) * (dx - s.dx);
    const out = s.x === null ? x : s.x + alpha(minCutoff + beta * Math.abs(edx), dt) * (x - s.x);
    Object.assign(s, { x: out, dx: edx, t });
    return out;
  };
};
const filterLandmarks = (name, lms, t, minCutoff, beta) => {
  const bank = (state.filters[name] = state.filters[name] ?? []);
  return lms.map((p, i) => {
    const f = (bank[i] = bank[i] ?? { x: oneEuro(minCutoff, beta), y: oneEuro(minCutoff, beta), z: oneEuro(minCutoff, beta) });
    return { ...p, x: f.x(p.x, t), y: f.y(p.y, t), z: f.z(p.z, t) };
  });
};
const resetFilter = name => { state.filters[name] = []; };

const db = await new Promise((res, rej) => {
  const req = indexedDB.open("texttosign-3d", 1);
  req.onupgradeneeded = () => req.result.createObjectStore("signs", { keyPath: "word" });
  req.onsuccess = () => res(req.result);
  req.onerror = () => rej(req.error);
});
const tx = (mode, fn) => new Promise((res, rej) => {
  const t = db.transaction("signs", mode);
  const req = fn(t.objectStore("signs"));
  t.oncomplete = () => res(req?.result);
  t.onerror = () => rej(t.error);
});
const dbPut = sign => tx("readwrite", s => s.put(sign));
const dbAll = () => tx("readonly", s => s.getAll());
const dbDelete = word => tx("readwrite", s => s.delete(word));

const loadLibrary = async () => {
  const all = await dbAll();
  state.library = new Map(all.filter(s => s.v === FORMAT).map(s => [s.word, s]));
  renderLibrary();
};

const renderLibrary = () => {
  const items = [...state.library.values()].sort((a, b) => (a.word.length === 1) - (b.word.length === 1) || a.word.localeCompare(b.word));
  el.library.innerHTML = items.length
    ? items.map(s => `<li>
        <span class="word">${esc(s.word)}</span>
        <span class="meta">${s.word.length === 1 ? "letter" : "sign"} · ${(s.duration / 1000).toFixed(1)}s · ${s.frames.length}f</span>
        <button data-action="play" data-word="${esc(s.word)}">Play</button>
        ${s.video ? `<button data-action="video" data-word="${esc(s.word)}">Video</button>` : ""}
        <button data-action="delete" data-word="${esc(s.word)}">Delete</button>
      </li>`).join("")
    : `<li class="empty">No signs yet. Start the camera, type a word and press "Record sign".</li>`;
};

const libraryActions = {
  play: word => playItems([{ label: word, sign: state.library.get(word) }]),
  video: word => {
    URL.revokeObjectURL(el.preview.src);
    el.preview.src = URL.createObjectURL(state.library.get(word).video);
    el.preview.play();
  },
  delete: async word => {
    await dbDelete(word);
    state.library.delete(word);
    renderLibrary();
    setStatus(`Deleted "${word}"`);
  }
};
el.library.addEventListener("click", e => libraryActions[e.target.dataset.action]?.(e.target.dataset.word));

const renderer = new THREE.WebGLRenderer({ canvas: el.avatar, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);
camera.position.set(0, 1.2, 2.4);
const controls = new OrbitControls(camera, el.avatar);
controls.enableDamping = true;
controls.minDistance = 0.4;
controls.maxDistance = 6;
scene.add(new THREE.HemisphereLight(0xffffff, 0x404050, 1.4));
const key = new THREE.DirectionalLight(0xffffff, 1.8);
key.position.set(0.6, 2, 2.5);
scene.add(key);
const rim = new THREE.DirectionalLight(0xbfd0ff, 0.9);
rim.position.set(-1.5, 2, -1.5);
scene.add(rim);
const clock = new THREE.Clock();
const loader = new GLTFLoader();
loader.register(parser => new VRMLoaderPlugin(parser));

const resizeAvatar = () => {
  const w = el.avatar.clientWidth;
  const h = el.avatar.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
};
window.addEventListener("resize", resizeAvatar);
resizeAvatar();

const node = name => state.vrm?.humanoid.getNormalizedBoneNode(name) ?? null;
const worldPos = name => node(name)?.getWorldPosition(new THREE.Vector3()) ?? null;
const restDir = (a, b) => {
  const pa = worldPos(a);
  const pb = worldPos(b);
  return pa && pb ? pb.sub(pa).normalize() : null;
};

const captureRest = () => {
  state.vrm.scene.updateMatrixWorld(true);
  const side = s => {
    const fingers = FINGERS.map(([f, segs]) => {
      const names = segs.map(seg => `${s}${f}${seg}`);
      const d0 = restDir(names[0], names[1]);
      const d1 = restDir(names[1], names[2]);
      return { names, dirs: [d0, d1, d1] };
    });
    const span = (worldPos(`${s}IndexProximal`) ?? new THREE.Vector3(0, 0, 0.02)).sub(worldPos(`${s}LittleProximal`) ?? new THREE.Vector3());
    return {
      upper: restDir(`${s}UpperArm`, `${s}LowerArm`),
      lower: restDir(`${s}LowerArm`, `${s}Hand`),
      handF: restDir(`${s}Hand`, `${s}MiddleProximal`),
      handS: span.normalize(),
      fingers
    };
  };
  state.rest = {
    sceneQ: state.vrm.scene.getWorldQuaternion(new THREE.Quaternion()),
    hasChest: !!node("chest"),
    left: side("left"),
    right: side("right")
  };
};

const frameCamera = () => {
  const head = worldPos("head");
  const hips = worldPos("hips");
  const torso = head.y - hips.y;
  const top = head.y + torso * 0.75;
  const bottom = hips.y - torso * 0.15;
  const centerY = (top + bottom) / 2;
  const fitH = (top - bottom) / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  const fitW = fitH / Math.min(camera.aspect, 1);
  const distance = Math.max(fitH, fitW * 0.85);
  state.view = { pos: new THREE.Vector3(0, centerY, distance), target: new THREE.Vector3(0, centerY, 0) };
  resetView();
};
const resetView = () => {
  state.view && camera.position.copy(state.view.pos);
  state.view && controls.target.copy(state.view.target);
  controls.update();
};

const loadAvatar = async url => {
  el.avatarLoading.textContent = "Loading avatar…";
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.removeUnnecessaryJoints(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse(o => {
    o.frustumCulled = false;
  });
  const old = state.vrm;
  old && scene.remove(old.scene);
  old && VRMUtils.deepDispose(old.scene);
  scene.add(vrm.scene);
  state.vrm = vrm;
  vrm.update(0);
  captureRest();
  frameCamera();
  el.avatarLoading.textContent = "";
};

const basisQuat = (a, b) => {
  const x = a.clone().normalize();
  const z = new THREE.Vector3().crossVectors(x, b).normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
};
const alignFrame = (restA, restB, a, b) => {
  const ok = new THREE.Vector3().crossVectors(a.clone().normalize(), b.clone().normalize()).length() > 0.08;
  return ok
    ? basisQuat(a, b).multiply(basisQuat(restA, restB).invert())
    : new THREE.Quaternion().setFromUnitVectors(restA, a.clone().normalize());
};
const swingTo = (parent, rest, target) => new THREE.Quaternion()
  .setFromUnitVectors(rest.clone().applyQuaternion(parent).normalize(), target.clone().normalize())
  .multiply(parent);
const relative = (parent, child) => parent.clone().invert().multiply(child);
const twistAbout = (q, axis) => {
  const p = axis.clone().multiplyScalar(axis.dot(new THREE.Vector3(q.x, q.y, q.z)));
  return new THREE.Quaternion(p.x, p.y, p.z, q.w).normalize();
};

const solveArm = (out, side, frame, chestQ) => {
  const cfg = SIDES[side];
  const rest = state.rest[side];
  const b = frame.body;
  const upperDir = V(b[cfg.e]).sub(V(b[cfg.s]));
  const foreDir = V(b[cfg.w]).sub(V(b[cfg.e]));
  const upperQ = alignFrame(rest.upper, new THREE.Vector3(0, 0, 1), upperDir, foreDir);
  const lowerQ = swingTo(upperQ, rest.lower, foreDir);
  const hand = frame[cfg.hand];
  const handQ = hand
    ? alignFrame(rest.handF, rest.handS, V(hand[9]).sub(V(hand[0])), V(hand[5]).sub(V(hand[17])))
    : lowerQ.clone();
  const handRel = relative(lowerQ, handQ);
  const halfTwist = I().slerp(twistAbout(handRel, rest.lower), 0.5);
  const lowerFinal = lowerQ.clone().multiply(halfTwist);
  out[`${side}UpperArm`] = relative(chestQ, upperQ);
  out[`${side}LowerArm`] = relative(upperQ, lowerFinal);
  out[`${side}Hand`] = relative(lowerFinal, handQ);
  rest.fingers.forEach((finger, fi) => {
    const idx = FINGERS[fi][2];
    const chain = { parent: handQ };
    finger.names.forEach((name, si) => {
      const segQ = hand && finger.dirs[si]
        ? swingTo(chain.parent, finger.dirs[si], V(hand[idx[si + 1]]).sub(V(hand[idx[si]])))
        : chain.parent.clone();
      out[name] = relative(chain.parent, segQ);
      chain.parent = segQ;
    });
  });
};

const solvePose = frame => {
  const b = frame.body;
  const across = V(b.ls).sub(V(b.rs));
  const up = V(b.ls).add(V(b.rs)).multiplyScalar(0.5).sub(V(b.lh).add(V(b.rh)).multiplyScalar(0.5));
  const torso = I().slerp(alignFrame(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), across, up), 0.85);
  const spineQ = state.rest.hasChest ? I().slerp(torso, 0.5) : torso.clone();
  const chestQ = torso;
  const headQ = new THREE.Quaternion(...frame.head).normalize();
  const neckQ = chestQ.clone().slerp(headQ, 0.4);
  const out = {
    hips: I(),
    spine: spineQ,
    chest: relative(spineQ, chestQ),
    upperChest: I(),
    neck: relative(chestQ, neckQ),
    head: relative(neckQ, headQ),
    leftShoulder: I(),
    rightShoulder: I()
  };
  solveArm(out, "left", frame, chestQ);
  solveArm(out, "right", frame, chestQ);
  return out;
};

const applyPose = (frame, lerp) => {
  const rel = solvePose(frame);
  const S = state.rest.sceneQ;
  const Si = S.clone().invert();
  Object.entries(rel).forEach(([name, q]) => {
    const n = node(name);
    n && n.quaternion.slerp(Si.clone().multiply(q).multiply(S), lerp);
  });
};

const applyFace = (face, lerp) => {
  const em = state.vrm?.expressionManager;
  const avg = (a, b) => ((face[a] ?? 0) + (face[b] ?? 0)) / 2;
  const target = {
    aa: clamp01((face.jawOpen ?? 0) * 1.7),
    oh: clamp01((face.mouthFunnel ?? 0) * 1.6),
    ou: clamp01((face.mouthPucker ?? 0) * 1.4),
    ee: clamp01(avg("mouthStretchLeft", "mouthStretchRight") * 2.2),
    happy: clamp01(avg("mouthSmileLeft", "mouthSmileRight") * 1.2),
    surprised: clamp01((face.browInnerUp ?? 0) * 0.9 + avg("eyeWideLeft", "eyeWideRight") * 0.8),
    angry: clamp01(avg("browDownLeft", "browDownRight") * 1.6),
    blinkLeft: clamp01((face.eyeBlinkRight ?? 0) * 1.35 - 0.1),
    blinkRight: clamp01((face.eyeBlinkLeft ?? 0) * 1.35 - 0.1)
  };
  em && Object.entries(target).forEach(([name, v]) => {
    const cur = em.getValue(name) ?? 0;
    em.getExpression(name) && em.setValue(name, cur + (v - cur) * lerp);
  });
};

const applyFrame = (frame, lerp) => {
  state.rest && applyPose(frame, lerp);
  applyFace(frame.face ?? IDLE.face, lerp);
};

const createModel = (fs, Cls, path, extra) => {
  const opts = delegate => ({ baseOptions: { modelAssetPath: path, delegate }, runningMode: "VIDEO", ...extra });
  return Cls.createFromOptions(fs, opts("GPU")).catch(() => Cls.createFromOptions(fs, opts("CPU")));
};

const loadModels = async () => {
  const fs = await FilesetResolver.forVisionTasks(WASM);
  const [pose, hand, face] = await Promise.all([
    createModel(fs, PoseLandmarker, MODELS.pose, { numPoses: 1, minPoseDetectionConfidence: 0.6, minPosePresenceConfidence: 0.6, minTrackingConfidence: 0.6 }),
    createModel(fs, HandLandmarker, MODELS.hand, { numHands: 2, minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.6 }),
    createModel(fs, FaceLandmarker, MODELS.face, { numFaces: 1, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true })
  ]);
  state.models = { pose, hand, face };
};

const startCamera = async () => {
  state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" }, audio: false });
  el.video.srcObject = state.stream;
  await el.video.play();
  el.overlay.width = el.video.videoWidth;
  el.overlay.height = el.video.videoHeight;
  el.camBtn.disabled = true;
  el.camBtn.textContent = "Camera on";
  el.recBtn.disabled = false;
  setStatus("Camera ready. Type a word and record it.", "ok");
};

const hold = (key, val) => {
  state.miss[key] = val ? 0 : state.miss[key] + 1;
  return val ?? (state.miss[key] < HOLD_FRAMES ? state.liveFrame?.[key] ?? null : null);
};

const splitHands = (hands2d, pose2d) => {
  const scored = hands2d
    .map((h, i) => ({ i, s: dist2(h[0], pose2d[16]) - dist2(h[0], pose2d[15]) }))
    .sort((a, b) => a.s - b.s);
  return scored.length > 1
    ? { userRight: scored[0].i, userLeft: scored[1].i }
    : { userRight: scored[0]?.s < 0 ? scored[0].i : -1, userLeft: scored[0]?.s >= 0 ? scored[0].i : -1 };
};

const headQuat = (fr, t) => {
  const data = fr.facialTransformationMatrixes?.[0]?.data;
  const q = data ? new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().fromArray(data)) : null;
  const sign = q && q.w < 0 ? -1 : 1;
  const f = q ? filterLandmarks("head", [{ x: q.x * sign, y: -q.y * sign, z: -q.z * sign }, { x: q.w * sign, y: 0, z: 0 }], t, 1.5, 0.3) : null;
  return f ? new THREE.Quaternion(f[0].x, f[0].y, f[0].z, f[1].x).normalize().toArray().map(r4) : state.liveFrame?.head ?? [0, 0, 0, 1];
};

const buildFrame = (pr, hr, fr, t) => {
  const pose2d = pr.landmarks[0];
  const pw = filterLandmarks("pose", pr.worldLandmarks[0], t, 1.2, 0.5);
  const vis = i => pose2d[i].visibility ?? 1;
  const armPts = side => {
    const c = SIDES[side];
    const ok = Math.min(vis(c.elbow), vis(c.wrist)) > VIS_MIN;
    const s = mirror(pw[c.shoulder]);
    const e = ok ? mirror(pw[c.elbow]) : add(s, [c.sign * 0.04, -0.27, -0.01]);
    const w = ok ? mirror(pw[c.wrist]) : add(e, [c.sign * 0.01, -0.25, 0.04]);
    return { [c.s]: s, [c.e]: e, [c.w]: w };
  };
  const body = { ...armPts("left"), ...armPts("right"), lh: mirror(pw[24]), rh: mirror(pw[23]) };
  const { userRight, userLeft } = splitHands(hr.landmarks ?? [], pose2d);
  const world = hr.worldLandmarks ?? [];
  userRight < 0 && resetFilter("handA");
  userLeft < 0 && resetFilter("handB");
  const left = userRight >= 0 ? filterLandmarks("handA", world[userRight], t, 1.6, 1.0).map(mirror) : null;
  const right = userLeft >= 0 ? filterLandmarks("handB", world[userLeft], t, 1.6, 1.0).map(mirror) : null;
  const cats = fr.faceBlendshapes?.[0]?.categories ?? [];
  const face = Object.fromEntries(FACE_KEYS.map(k => [k, r4(cats.find(c => c.categoryName === k)?.score ?? 0)]));
  return { body, left: hold("left", left), right: hold("right", right), head: headQuat(fr, t), face };
};

const drawOverlay = (pr, hr, fr) => {
  const w = el.overlay.width;
  const h = el.overlay.height;
  octx.clearRect(0, 0, w, h);
  const line = (pts, links, color, width) => {
    octx.strokeStyle = color;
    octx.lineWidth = width;
    links.forEach(([a, b]) => {
      octx.beginPath();
      octx.moveTo(pts[a].x * w, pts[a].y * h);
      octx.lineTo(pts[b].x * w, pts[b].y * h);
      octx.stroke();
    });
  };
  const dots = (pts, color, r) => {
    octx.fillStyle = color;
    pts.forEach(p => {
      octx.beginPath();
      octx.arc(p.x * w, p.y * h, r, 0, Math.PI * 2);
      octx.fill();
    });
  };
  const lm = pr.landmarks?.[0];
  lm && line(lm, POSE_LINKS, "rgba(79,124,255,.9)", 4);
  (hr.landmarks ?? []).forEach(hand => {
    line(hand, HAND_LINKS, "rgba(60,207,145,.95)", 3);
    dots(hand, "#fff", 3.5);
  });
  const face = fr.faceLandmarks?.[0];
  face && dots(face.filter((_, i) => i % 6 === 0), "rgba(255,200,120,.7)", 1.4);
  const split = lm ? splitHands(hr.landmarks ?? [], lm) : { userLeft: -1, userRight: -1 };
  const flags = { Body: !!lm, "L hand": split.userLeft >= 0, "R hand": split.userRight >= 0, Face: !!face };
  el.trackInfo.innerHTML = Object.entries(flags).map(([k, v]) => `<span class="${v ? "on" : ""}">${k}</span>`).join("");
};

const detect = now => {
  state.lastVT = el.video.currentTime;
  const ts = Math.max(Math.round(now), state.lastTs + 1);
  state.lastTs = ts;
  const pr = state.models.pose.detectForVideo(el.video, ts);
  const hr = state.models.hand.detectForVideo(el.video, ts);
  const fr = state.models.face.detectForVideo(el.video, ts);
  drawOverlay(pr, hr, fr);
  const has = pr.landmarks?.length > 0 && pr.worldLandmarks?.length > 0;
  const frame = has ? buildFrame(pr, hr, fr, now) : null;
  state.liveFrame = frame ?? state.liveFrame;
  state.recording && frame && state.rec.push({ ...frame, t: Math.round(now - state.recStart) });
};

const lerpAny = (a, b, t) => a && b
  ? (typeof b === "number"
    ? a + (b - a) * t
    : Array.isArray(b)
      ? b.map((v, i) => lerpAny(a[i], v, t))
      : Object.fromEntries(Object.keys(b).map(k => [k, lerpAny(a[k], b[k], t)])))
  : (typeof b === "number" ? b : (t < 0.5 ? a : b));
const lerpFrame = (a, b, t) => {
  const head = new THREE.Quaternion(...a.head).slerp(new THREE.Quaternion(...b.head), t).toArray();
  return { ...lerpAny({ body: a.body, left: a.left, right: a.right, face: a.face }, { body: b.body, left: b.left, right: b.right, face: b.face }, t), head };
};

const findIndex = (frames, t) => {
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const m = (lo + hi + 1) >> 1;
    frames[m].t <= t ? (lo = m) : (hi = m - 1);
  }
  return lo;
};

const sampleTimeline = (tl, t) => {
  const i = findIndex(tl.frames, t);
  const a = tl.frames[i];
  const b = tl.frames[Math.min(i + 1, tl.frames.length - 1)];
  const span = b.t - a.t;
  const k = span > 0 ? clamp01((t - a.t) / span) : 0;
  return lerpFrame(a, b, span > GAP_MS * 0.5 ? k * k * (3 - 2 * k) : k);
};

const buildTimeline = items => {
  const frames = [{ ...IDLE, t: 0 }];
  const segments = [];
  let offset = GAP_MS;
  items.forEach(item => {
    const src = item.sign.frames;
    const t0 = src[0].t;
    const dur = src.at(-1).t - t0;
    src.forEach(f => frames.push({ ...f, t: offset + f.t - t0 }));
    segments.push({ start: offset, end: offset + dur, label: item.label });
    offset += dur + GAP_MS;
  });
  frames.push({ ...IDLE, t: offset });
  return { frames, segments, duration: offset };
};

const renderSequence = (items, missing) => {
  el.sequence.innerHTML = items.map((it, i) => `<span class="chip ${it.spelled ? "spelled" : ""}" data-seg="${i}">${esc(it.label)}</span>`).join("")
    + missing.map(w => `<span class="chip missing" title="Not in library">${esc(w)}</span>`).join("");
};

const highlightSegment = idx => {
  state.activeSeg = idx;
  [...el.sequence.querySelectorAll("[data-seg]")].forEach(c => c.classList.toggle("active", Number(c.dataset.seg) === idx));
  el.caption.textContent = state.timeline?.segments[idx]?.label ?? "";
};

const playItems = items => {
  const valid = items.filter(i => i.sign?.frames?.length > 1);
  return valid.length ? beginPlayback(valid) : setStatus("Nothing to play. Record signs first.", "err");
};

const beginPlayback = items => {
  state.timeline = buildTimeline(items);
  state.playT = 0;
  state.lastNow = performance.now();
  state.playing = true;
  state.activeSeg = -1;
  setStatus(`Signing ${items.length} item(s)…`, "ok");
};

const stopPlayback = () => {
  state.playing = false;
  highlightSegment(-1);
};

const playbackFrame = now => {
  const dt = now - state.lastNow;
  state.lastNow = now;
  state.playT += dt * Number(el.speed.value);
  const tl = state.timeline;
  const t = Math.min(state.playT, tl.duration);
  const seg = tl.segments.findIndex(s => t >= s.start && t <= s.end);
  seg !== state.activeSeg && seg >= 0 && highlightSegment(seg);
  const frame = sampleTimeline(tl, t);
  state.playT >= tl.duration && (el.loopChk.checked ? (state.playT = 0) : stopPlayback());
  return frame;
};

const planText = text => {
  const tokens = normalizeKey(text).split(" ").filter(Boolean);
  const items = [];
  const missing = [];
  let i = 0;
  while (i < tokens.length) {
    const n = [4, 3, 2, 1].find(k => i + k <= tokens.length && state.library.has(tokens.slice(i, i + k).join(" ")));
    const key = tokens.slice(i, i + (n ?? 1)).join(" ");
    const letters = [...key.replace(/'/g, "")];
    const spelled = letters.filter(ch => state.library.has(ch)).map(ch => ({ label: ch.toUpperCase(), sign: state.library.get(ch), spelled: true }));
    const lacking = letters.filter(ch => !state.library.has(ch));
    items.push(...(n ? [{ label: key, sign: state.library.get(key) }] : spelled));
    missing.push(...(n ? [] : lacking.length ? [`${key} (${[...new Set(lacking)].join(",")})`] : []));
    i += n ?? 1;
  }
  return { items, missing };
};

const translate = () => {
  const { items, missing } = planText(el.textInput.value);
  renderSequence(items, missing);
  playItems(items);
  missing.length && setStatus(`Missing signs: ${missing.join("; ")}. Record them (or their letters) in the library.`, "err");
};

const countdown = async () => {
  for (const n of [3, 2, 1]) {
    el.countdown.textContent = n;
    await sleep(1000);
  }
  el.countdown.textContent = "";
};

const startRecording = async () => {
  const word = normalizeKey(el.wordInput.value);
  return word ? beginRecording(word) : setStatus("Type the word or letter you are about to sign.", "err");
};

const beginRecording = async word => {
  el.recBtn.disabled = true;
  stopPlayback();
  await countdown();
  const mime = ["video/webm;codecs=vp9", "video/webm", "video/mp4"].find(m => MediaRecorder.isTypeSupported(m)) ?? "";
  state.chunks = [];
  state.recorder = new MediaRecorder(state.stream, mime ? { mimeType: mime } : {});
  state.recorder.ondataavailable = e => e.data.size && state.chunks.push(e.data);
  state.recorder.start(200);
  state.rec = [];
  state.recWord = word;
  state.recStart = performance.now();
  state.recording = true;
  el.recBadge.classList.add("on");
  el.stopBtn.disabled = false;
  setStatus(`Recording "${word}"… press Stop & save when done.`);
};

const stopRecording = () => {
  state.recording = false;
  el.recBadge.classList.remove("on");
  el.stopBtn.disabled = true;
  el.recBtn.disabled = false;
  const frames = state.rec;
  const word = state.recWord;
  const recorder = state.recorder;
  recorder.onstop = () => finishRecording(word, frames, new Blob(state.chunks, { type: recorder.mimeType || "video/webm" }));
  recorder.stop();
};

const finishRecording = async (word, frames, video) => frames.length >= 5
  ? saveSign(word, frames, video)
  : setStatus("Recording too short or no body detected. Try again.", "err");

const saveSign = async (word, frames, video) => {
  const t0 = frames[0].t;
  const clean = frames.map(f => ({ ...f, t: f.t - t0 }));
  const sign = { v: FORMAT, word, frames: clean, duration: clean.at(-1).t, video, created: Date.now() };
  await dbPut(sign);
  state.library.set(word, sign);
  renderLibrary();
  el.wordInput.value = "";
  setStatus(`Saved "${word}" (${clean.length} frames).`, "ok");
  playItems([{ label: word, sign }]);
};

const exportLibrary = () => {
  const data = [...state.library.values()].map(({ word, frames, duration, created }) => ({ word, frames, duration, created }));
  const blob = new Blob([JSON.stringify({ version: FORMAT, signs: data })], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sign-library.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

const importLibrary = async file => {
  const parsed = JSON.parse(await file.text());
  const signs = (parsed.version === FORMAT ? parsed.signs ?? [] : []).filter(s => s.word && s.frames?.length > 1);
  await Promise.all(signs.map(s => dbPut({ ...s, v: FORMAT, word: normalizeKey(s.word), video: state.library.get(normalizeKey(s.word))?.video ?? null })));
  await loadLibrary();
  setStatus(`Imported ${signs.length} sign(s).`, signs.length ? "ok" : "err");
};

const loop = now => {
  const ready = state.models && state.stream && el.video.readyState >= 2 && el.video.currentTime !== state.lastVT;
  ready && detect(now);
  const frame = state.playing ? playbackFrame(now) : (el.liveChk.checked && state.liveFrame) || IDLE;
  state.vrm && applyFrame(frame, state.playing ? PLAY_LERP : LIVE_LERP);
  state.vrm?.update(clock.getDelta());
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
};

el.camBtn.addEventListener("click", () => startCamera().catch(e => setStatus(`Camera error: ${e.message}`, "err")));
el.recBtn.addEventListener("click", () => startRecording().catch(e => setStatus(`Record error: ${e.message}`, "err")));
el.stopBtn.addEventListener("click", stopRecording);
el.playBtn.addEventListener("click", translate);
el.textInput.addEventListener("keydown", e => e.key === "Enter" && translate());
el.wordInput.addEventListener("keydown", e => e.key === "Enter" && !el.recBtn.disabled && startRecording());
el.haltBtn.addEventListener("click", stopPlayback);
el.resetViewBtn.addEventListener("click", resetView);
el.speed.addEventListener("input", () => { el.speedVal.textContent = `${Number(el.speed.value).toFixed(2)}×`; });
el.exportBtn.addEventListener("click", exportLibrary);
el.importInput.addEventListener("change", e => {
  const file = e.target.files[0];
  file && importLibrary(file).catch(err => setStatus(`Import failed: ${err.message}`, "err"));
  e.target.value = "";
});
el.modelInput.addEventListener("change", e => {
  const file = e.target.files[0];
  file && loadAvatar(URL.createObjectURL(file)).then(() => setStatus(`Avatar "${file.name}" loaded.`, "ok")).catch(err => setStatus(`Avatar load failed: ${err.message}`, "err"));
  e.target.value = "";
});

await loadLibrary();
requestAnimationFrame(loop);
loadAvatar(DEFAULT_AVATAR).catch(e => {
  el.avatarLoading.textContent = "Avatar failed to load";
  setStatus(`Avatar load failed: ${e.message}`, "err");
});
loadModels()
  .then(() => {
    el.camBtn.disabled = false;
    setStatus("Models loaded. Start the camera.", "ok");
  })
  .catch(e => setStatus(`Model load failed: ${e.message}`, "err"));
