import * as THREE from 'three';
import type { CloudExample } from './data';
import { applyGradient, buildGradient, METHODS, relativeL2, tangentBasis, type MethodName, type Vec3 } from './mls';
import { Arrows, median, norm3, PointSprites, Viewer } from './three-view';
import { fieldInterp, fieldInterpDark, isDark, col } from './colors';
import { cssVar, fmtRel, onThemeChange, reducedMotion, segmented } from './util';

/**
 * Hero: every point's tangent frame spins at its own speed. The gradient arrows are
 * recomputed with the spinning frames using the real MLS operator; with the invariant
 * conventions they do not move, with the upstream ones they breathe slightly.
 */
export function initHero(ex: CloudExample) {
  const canvas = document.getElementById('hero-canvas') as HTMLCanvasElement;
  const view = new Viewer(canvas, undefined, { zoom: false });
  view.camera.up.set(0, 0, 1);
  view.controls.autoRotate = false;
  const n = ex.points.length;

  const group = new THREE.Group();
  view.scene.add(group);

  const dots = new PointSprites(n, 0.014);
  dots.setPositions(ex.points);
  group.add(dots.mesh);

  // tangent frames: two short segments per point
  const frameLen = 0.065;
  const framePos = new Float32Array(n * 4 * 3);
  const frameCol = new Float32Array(n * 4 * 3);
  const frameGeo = new THREE.BufferGeometry();
  frameGeo.setAttribute('position', new THREE.BufferAttribute(framePos, 3).setUsage(THREE.DynamicDrawUsage));
  frameGeo.setAttribute('color', new THREE.BufferAttribute(frameCol, 3));
  const frames = new THREE.LineSegments(frameGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }));
  frames.frustumCulled = false;
  group.add(frames);

  const arrows = new Arrows(n, 0x1c2330, 0.0038);
  group.add(arrows.group);

  let method: MethodName = 'C_invariant';
  const reference: Partial<Record<MethodName, Vec3[]>> = {};
  const refFor = (m: MethodName) =>
    (reference[m] ??= applyGradient(buildGradient(ex.points, ex.normals, ex.index, METHODS[m], ex.regularizer), ex.signal, true));
  // fixed display scale from the invariant reference, so a change of scale shows as a change of length
  const displayScale = 0.095 / median(refFor('C_invariant').map(norm3));
  const scaleFor = (m: MethodName) => displayScale * median(refFor('C_invariant').map(norm3)) / median(refFor(m).map(norm3));

  // per-point spin speeds and phases (deterministic)
  const speed = new Float64Array(n);
  const phase = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    speed[i] = (0.35 + 0.9 * frac(Math.sin(i * 12.9898) * 43758.5453)) * (i % 2 ? 1 : -1);
    phase[i] = ex.angles[i];
  }
  const angles = new Float64Array(n);

  const output = document.getElementById('hero-drift') as HTMLOutputElement;
  let lastCompute = 0;
  let current: Vec3[] = refFor(method);

  function paint() {
      const dark = isDark();
      const interp = dark ? fieldInterpDark : fieldInterp;
      const s = ex.signal;
      const lo = Math.min(...s), hi = Math.max(...s);
      dots.setColors(s.map((v) => col(interp((v - lo) / (hi - lo)))));
      arrows.material.color.set(cssVar('--arrow'));
      if (view.scene.fog instanceof THREE.Fog) view.scene.fog.color.set(cssVar('--hero-bg'));
      const u = col(cssVar('--u')), v = col(cssVar('--v'));
      for (let i = 0; i < n; i++) {
        frameCol.set([u.r, u.g, u.b, u.r, u.g, u.b, v.r, v.g, v.b, v.r, v.g, v.b], i * 12);
      }
      (frameGeo.attributes.color as THREE.BufferAttribute).needsUpdate = true;
      view.invalidate();
  }

  function drawFrames() {
    const basis = tangentBasis(ex.normals, angles);
    for (let i = 0; i < n; i++) {
      const p = ex.points[i], x = basis.x[i], y = basis.y[i];
      const o = i * 12;
      framePos[o] = p[0]; framePos[o + 1] = p[1]; framePos[o + 2] = p[2];
      framePos[o + 3] = p[0] + x[0] * frameLen; framePos[o + 4] = p[1] + x[1] * frameLen; framePos[o + 5] = p[2] + x[2] * frameLen;
      framePos[o + 6] = p[0]; framePos[o + 7] = p[1]; framePos[o + 8] = p[2];
      framePos[o + 9] = p[0] + y[0] * frameLen; framePos[o + 10] = p[1] + y[1] * frameLen; framePos[o + 11] = p[2] + y[2] * frameLen;
    }
    (frameGeo.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  }

  function compute() {
    const g = buildGradient(ex.points, ex.normals, ex.index, METHODS[method], ex.regularizer, angles);
    current = applyGradient(g, ex.signal, true);
    arrows.set(ex.points, current, scaleFor(method), ex.normals, 0.004);
    output.textContent = fmtRel(relativeL2(current, refFor(method)));
    output.style.color = method === 'C_invariant' ? cssVar('--stable') : cssVar('--drift');
  }

  let t0 = -1;
  view.onFrame = (time) => {
    if (reducedMotion()) return false;
    if (t0 < 0) t0 = time;
    const t = (time - t0) / 1000;
    for (let i = 0; i < n; i++) angles[i] = phase[i] + speed[i] * t;
    group.rotation.z = t * 0.04;
    drawFrames();
    if (time - lastCompute > 90) { lastCompute = time; compute(); }
    return true;
  };

  // initial state (and the only state under reduced motion): the audit's own random rotation
  for (let i = 0; i < n; i++) angles[i] = phase[i];
  drawFrames();
  compute();
  paint();
  onThemeChange(paint);

  segmented(document.getElementById('hero-toggle')!, (m) => { method = m as MethodName; compute(); view.invalidate(); });

  // place the sphere to the right on wide screens, behind the title on narrow ones
  const place = () => {
    const wide = canvas.clientWidth > 900;
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const halfH = Math.atan(Math.tan((view.camera.fov * Math.PI) / 360) * aspect);
    const dist = wide ? 5.3 : Math.max(5.3, 1.12 / Math.sin(halfH));
    view.controls.maxDistance = 20;
    view.camera.position.set(3.4, 2.4, 1.5).normalize().multiplyScalar(dist);
    view.scene.fog = new THREE.Fog(cssVar('--hero-bg'), dist - 0.7, dist + 1.3);
    view.controls.target.set(0, 0, 0);
    view.camera.setViewOffset(canvas.clientWidth, canvas.clientHeight, wide ? -canvas.clientWidth * 0.17 : 0, wide ? 0 : canvas.clientHeight * 0.07, canvas.clientWidth, canvas.clientHeight);
    view.controls.update();
    view.invalidate();
  };
  new ResizeObserver(place).observe(canvas);
}

const frac = (x: number) => x - Math.floor(x);
