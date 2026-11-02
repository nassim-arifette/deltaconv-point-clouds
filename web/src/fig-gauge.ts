import * as THREE from 'three';
import type { CloudExample } from './data';
import { exampleLabel, METHOD_LABEL, METHOD_LETTER } from './data';
import { applyGradient, buildGradient, METHOD_NAMES, METHODS, relativeL2, type MethodName, type Vec3 } from './mls';
import { Arrows, frameCloud, median, norm3, PointSprites, Viewer } from './three-view';
import { col, driftInterp, driftInterpDark, isDark } from './colors';
import { cssVar, fmtRel, onThemeChange, sweep } from './util';

const LOG_LO = -16, LOG_HI = -1;

/** Figure 4: the audit's gauge test, recomputed live for two methods side by side. */
export function initGauge(examples: CloudExample[]) {
  const fig = document.getElementById('fig-gauge')!;
  const canvases = fig.querySelectorAll<HTMLCanvasElement>('canvas');
  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 50);
  camera.up.set(0, 0, 1);
  camera.position.set(2.8, 2.2, 1.9);
  const views = [new Viewer(canvases[0], camera), new Viewer(canvases[1], camera)];
  views[0].link(views[1]);

  const selects = [fig.querySelector<HTMLSelectElement>('[data-ctl=left]')!, fig.querySelector<HTMLSelectElement>('[data-ctl=right]')!];
  const outs = [fig.querySelector<HTMLOutputElement>('[data-out=left]')!, fig.querySelector<HTMLOutputElement>('[data-out=right]')!];
  for (const s of selects) METHOD_NAMES.forEach((m) => s.add(new Option(METHOD_LABEL[m], m)));
  const methods: MethodName[] = ['A_upstream', 'C_invariant'];
  selects.forEach((s, i) => { s.value = methods[i]; });

  const exSelect = fig.querySelector<HTMLSelectElement>('[data-ctl=example]')!;
  examples.forEach((e, i) => exSelect.add(new Option(exampleLabel(e), String(i))));
  // start on the noisy biased sphere: the audit's harshest saved example
  let exIdx = Math.max(0, examples.findIndex((e) => e.shape === 'sphere' && e.density === 'biased'));
  exSelect.value = String(exIdx);

  const slider = fig.querySelector<HTMLInputElement>('[data-ctl=t]')!;
  const tOut = fig.querySelector<HTMLOutputElement>('[data-out=t]')!;
  const playBtn = fig.querySelector<HTMLButtonElement>('[data-ctl=play]')!;
  const match = fig.querySelector<HTMLElement>('[data-role=match]')!;
  const legend = fig.querySelector<HTMLElement>('[data-role=legend]')!;

  let t = 0;
  const scenes = views.map(() => ({ dots: null as PointSprites | null, arrows: null as Arrows | null }));
  const refCache = new Map<string, Vec3[]>();
  const reference = (ex: CloudExample, m: MethodName) => {
    const key = `${ex.case_id}/${m}`;
    if (!refCache.has(key)) refCache.set(key, applyGradient(buildGradient(ex.points, ex.normals, ex.index, METHODS[m], ex.regularizer), ex.signal, true));
    return refCache.get(key)!;
  };

  function rebuild() {
    const ex = examples[exIdx];
    views.forEach((v, i) => {
      const s = scenes[i];
      if (s.dots) v.scene.remove(s.dots.mesh);
      if (s.arrows) v.scene.remove(s.arrows.group);
      s.dots = new PointSprites(ex.points.length, 0.024);
      s.dots.setPositions(ex.points);
      s.arrows = new Arrows(ex.points.length, cssVar('--arrow'), 0.0042);
      v.scene.add(s.dots.mesh, s.arrows.group);
    });
    frameCloud(camera, ex.points, views[0].controls, ex.shape === 'sphere' ? 4.6 : 3.8);
    update();
  }

  function update() {
    const ex = examples[exIdx];
    tOut.textContent = `${Math.round(t * 100)} %`;
    const angles = ex.angles.map((a) => a * t);
    const interp = isDark() ? driftInterpDark : driftInterp;
    methods.forEach((m, vi) => {
      const ref = reference(ex, m);
      const now = t === 0 ? ref : applyGradient(buildGradient(ex.points, ex.normals, ex.index, METHODS[m], ex.regularizer, angles), ex.signal, true);
      const rms = Math.sqrt(ref.reduce((s, g) => s + norm3(g) ** 2, 0) / ref.length);
      const colors = now.map((g, i) => {
        const d = norm3([g[0] - ref[i][0], g[1] - ref[i][1], g[2] - ref[i][2]]) / rms;
        const l = d > 0 ? Math.log10(d) : LOG_LO;
        return col(interp(Math.min(1, Math.max(0, (l - LOG_LO) / (LOG_HI - LOG_LO)))));
      });
      const s = scenes[vi];
      s.dots!.setColors(colors);
      s.arrows!.set(ex.points, now, 0.1 / median(ref.map(norm3)));
      s.arrows!.material.color.set(cssVar('--arrow'));
      const drift = t === 0 ? 0 : relativeL2(now, ref);
      outs[vi].textContent = t === 0 ? '0' : fmtRel(drift);
      outs[vi].style.color = drift > 1e-10 ? cssVar('--drift') : cssVar('--stable');
      views[vi].invalidate();
    });
    if (t >= 0.999) {
      const rec = methods.map((m) => `${METHOD_LETTER[m]} ${fmtRel(ex.measured[m].gauge_normalized_relative_l2)}`).join(', ');
      const rounding = methods.some((m) => ex.measured[m].gauge_normalized_relative_l2 < 1e-10);
      match.textContent = `The Python audit recorded ${rec} for this cloud.` +
        (rounding ? ' Values at the 10⁻¹⁵ level are rounding noise, so they differ in the last digit between NumPy’s solver and this one.' : '');
    } else {
      match.textContent = 'At full rotation the numbers match the values recorded by the Python audit.';
    }
  }

  function paintLegend() {
    const interp = isDark() ? driftInterpDark : driftInterp;
    const stops = Array.from({ length: 9 }, (_, i) => `${interp(i / 8)} ${(i / 8) * 100}%`).join(', ');
    legend.innerHTML = '';
    const lo = document.createElement('span'); lo.textContent = 'unchanged (≤10⁻¹⁶)';
    const bar = document.createElement('span'); bar.className = 'bar'; bar.style.background = `linear-gradient(90deg, ${stops})`;
    const hi = document.createElement('span'); hi.textContent = '10 %';
    legend.append(lo, bar, hi);
  }

  selects.forEach((s, i) => s.addEventListener('change', () => { methods[i] = s.value as MethodName; update(); }));
  exSelect.addEventListener('change', () => { exIdx = +exSelect.value; rebuild(); });
  let stop = () => {};
  slider.addEventListener('input', () => { stop(); t = +slider.value; update(); });
  playBtn.addEventListener('click', () => {
    stop();
    const from = t >= 0.99 ? 0 : t;
    stop = sweep(from, 1, 2600 * (1 - from) + 300, (v) => { t = v; slider.value = String(v); update(); });
  });
  onThemeChange(() => { paintLegend(); update(); });
  paintLegend();
  rebuild();
}
