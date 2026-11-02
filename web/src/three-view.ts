import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Vec3 } from './mls';
import { onScreen } from './util';

/** A render-on-demand WebGL view of one canvas. Several views may share one camera. */
export class Viewer {
  readonly scene = new THREE.Scene();
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  readonly camera: THREE.PerspectiveCamera;
  private dirty = true;
  private visible = false;
  private raf = 0;
  private peers: Viewer[] = [];
  onFrame?: (time: number) => boolean | void; // return true to keep animating

  constructor(readonly canvas: HTMLCanvasElement, camera?: THREE.PerspectiveCamera, opts: { zoom?: boolean } = {}) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.camera = camera ?? new THREE.PerspectiveCamera(32, 1, 0.05, 50);
    if (!camera) this.camera.position.set(2.6, 1.7, 3.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.enablePan = false;
    this.controls.enableZoom = opts.zoom ?? true;
    this.controls.minDistance = 1.6;
    this.controls.maxDistance = 16;
    this.controls.addEventListener('change', () => this.invalidate());

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8090a0, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(3, 5, 4);
    this.scene.add(sun);

    new ResizeObserver(() => this.resize()).observe(canvas);
    onScreen(canvas, (v) => { this.visible = v; if (v) this.loop(); });
  }

  /** Views sharing a camera re-render together. */
  link(other: Viewer) { this.peers.push(other); other.peers.push(this); }

  invalidate() {
    this.dirty = true;
    for (const p of this.peers) p.dirty = true;
    if (this.visible) this.loop();
  }

  private resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  private loop() {
    if (this.raf) return;
    const tick = (time: number) => {
      this.raf = 0;
      if (!this.visible) return;
      const moved = this.controls.update();
      const animating = this.onFrame?.(time) === true;
      if (moved || this.dirty || animating) {
        this.dirty = false;
        this.renderer.render(this.scene, this.camera);
      }
      if (moved || animating || this.controlsActive) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private get controlsActive() {
    // keep ticking briefly so damping can settle
    return (this.controls as unknown as { state: number }).state !== -1;
  }
}

const UP = new THREE.Vector3(0, 1, 0);
const tmpM = new THREE.Matrix4();
const tmpQ = new THREE.Quaternion();
const tmpV = new THREE.Vector3();
const tmpP = new THREE.Vector3();
const tmpS = new THREE.Vector3();

/** Instanced spheres, one per point, coloured individually. */
export class PointSprites {
  readonly mesh: THREE.InstancedMesh;
  constructor(count: number, radius = 0.022) {
    const geo = new THREE.IcosahedronGeometry(radius, 1);
    const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
    this.mesh = new THREE.InstancedMesh(geo, mat, count);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  }
  setPositions(points: Vec3[]) {
    points.forEach((p, i) => { tmpM.makeTranslation(p[0], p[1], p[2]); this.mesh.setMatrixAt(i, tmpM); });
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.computeBoundingSphere();
  }
  setColors(colors: THREE.Color[]) {
    colors.forEach((c, i) => this.mesh.setColorAt(i, c));
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

/** Instanced arrows: a thin cylinder shaft and a cone head. */
export class Arrows {
  readonly group = new THREE.Group();
  private shaft: THREE.InstancedMesh;
  private head: THREE.InstancedMesh;
  readonly material: THREE.MeshLambertMaterial;
  constructor(count: number, color: THREE.ColorRepresentation, private radius = 0.0055) {
    this.material = new THREE.MeshLambertMaterial({ color });
    const shaftGeo = new THREE.CylinderGeometry(1, 1, 1, 6, 1, true).translate(0, 0.5, 0);
    const headGeo = new THREE.ConeGeometry(1, 1, 10).translate(0, 0.5, 0);
    this.shaft = new THREE.InstancedMesh(shaftGeo, this.material, count);
    this.head = new THREE.InstancedMesh(headGeo, this.material, count);
    this.shaft.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.head.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.shaft.frustumCulled = false;
    this.head.frustumCulled = false;
    this.group.add(this.shaft, this.head);
  }
  /** Arrow i starts at origins[i] (lifted along lift[i] if given) and has length |v|*scale. */
  set(origins: Vec3[], vectors: Vec3[], scale: number, lift?: Vec3[], liftBy = 0) {
    const r = this.radius;
    for (let i = 0; i < origins.length; i++) {
      const v = vectors[i];
      const len = Math.hypot(v[0], v[1], v[2]) * scale;
      const o = origins[i];
      tmpP.set(o[0], o[1], o[2]);
      if (lift) tmpP.addScaledVector(tmpV.set(lift[i][0], lift[i][1], lift[i][2]), liftBy);
      if (len < 1e-6) {
        tmpM.makeScale(0, 0, 0);
        this.shaft.setMatrixAt(i, tmpM); this.head.setMatrixAt(i, tmpM);
        continue;
      }
      tmpV.set(v[0], v[1], v[2]).normalize();
      tmpQ.setFromUnitVectors(UP, tmpV);
      const headLen = Math.min(len * 0.42, r * 7);
      const shaftLen = Math.max(len - headLen, 0);
      tmpM.compose(tmpP, tmpQ, tmpS.set(r, shaftLen, r));
      this.shaft.setMatrixAt(i, tmpM);
      tmpP.addScaledVector(tmpV, shaftLen);
      tmpM.compose(tmpP, tmpQ, tmpS.set(r * 2.6, headLen, r * 2.6));
      this.head.setMatrixAt(i, tmpM);
    }
    this.shaft.count = this.head.count = origins.length;
    this.shaft.instanceMatrix.needsUpdate = true;
    this.head.instanceMatrix.needsUpdate = true;
  }
}

export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

export const norm3 = (v: Vec3) => Math.hypot(v[0], v[1], v[2]);

/** Fit the camera so a cloud centred at the origin fills the view. */
export function frameCloud(camera: THREE.PerspectiveCamera, points: Vec3[], controls: OrbitControls, distance = 4.2) {
  let r = 0;
  for (const p of points) r = Math.max(r, Math.hypot(p[0], p[1], p[2]));
  const dir = camera.position.clone().normalize();
  const w = (controls.domElement as HTMLElement | null)?.clientWidth ?? 1;
  const h = (controls.domElement as HTMLElement | null)?.clientHeight ?? 1;
  const halfH = Math.atan(Math.tan((camera.fov * Math.PI) / 360) * (w / Math.max(h, 1)));
  camera.position.copy(dir.multiplyScalar(Math.max(distance * r, (1.12 * r) / Math.sin(halfH))));
  controls.target.set(0, 0, 0);
  controls.update();
}
