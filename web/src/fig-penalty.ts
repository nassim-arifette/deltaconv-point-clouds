import { contours, geoPath, scaleLinear, line, range } from 'd3';
import type { CloudExample } from './data';
import { globalWidth, localFit, tangentBasis, type Vec3 } from './mls';
import { fieldInterp, fieldInterpDark, isDark } from './colors';
import { cssVar, el, fmtRel, onThemeChange, sweep } from './util';

const DEG = Math.PI / 180;

function rotatedAxes(x: Vec3, y: Vec3, theta: number): [Vec3, Vec3] {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [
    [c * x[0] + s * y[0], c * x[1] + s * y[1], c * x[2] + s * y[2]],
    [-s * x[0] + c * y[0], -s * x[1] + c * y[1], -s * x[2] + c * y[2]],
  ];
}

/** Figure 2: the ridge penalty of one real neighbourhood as its frame turns. */
export function initPenalty(ex: CloudExample) {
  const fig = document.getElementById('fig-penalty')!;
  const patch = fig.querySelector<SVGSVGElement>('[data-role=patch]')!;
  const curve = fig.querySelector<SVGSVGElement>('[data-role=curve]')!;
  const readout = fig.querySelector<HTMLElement>('[data-role=readout]')!;
  const slider = fig.querySelector<HTMLInputElement>('[data-ctl=theta]')!;
  const out = fig.querySelector<HTMLOutputElement>('[data-out=theta]')!;
  const playBtn = fig.querySelector<HTMLButtonElement>('[data-ctl=play]')!;
  const nextBtn = fig.querySelector<HTMLButtonElement>('[data-ctl=next]')!;

  const basis = tangentBasis(ex.normals);
  const width = globalWidth(ex.points, ex.index);
  const fit = (i: number, theta: number, reg: number, penalty: 'upstream' | 'tensor') => {
    const [bx, by] = rotatedAxes(basis.x[i], basis.y[i], theta);
    return { ...localFit(ex.points, ex.index[i], i, bx, by, ex.signal, reg, penalty, width), bx, by };
  };
  const penalties = (coef: number[]) => {
    const [, , , a, b, c] = coef;
    return { up: a * a + b * b + c * c, tensor: a * a + (b * b) / 2 + c * c };
  };

  // Pick points whose upstream penalty swings the most with the frame angle.
  const candidates = range(ex.points.length).map((i) => {
    const samples = range(0, 180, 15).map((d) => penalties(fit(i, d * DEG, 0, 'upstream').coef).up);
    const hi = Math.max(...samples), lo = Math.min(...samples);
    return { i, swing: (hi - lo) / hi, size: hi };
  }).filter((c) => c.size > 1e-4).sort((a, b) => b.swing - a.swing).slice(0, 10).map((c) => c.i);
  let pick = 0;
  let theta = 0;

  const W = 400, H = 400;
  patch.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const CW = 400, CH = 210, M = { l: 40, r: 14, t: 14, b: 30 };
  curve.setAttribute('viewBox', `0 0 ${CW} ${CH}`);

  let samples: { d: number; up: number; tensor: number }[] = [];

  function drawStatic() {
    const i = candidates[pick];
    patch.replaceChildren();
    curve.replaceChildren();
    const base = fit(i, 0, 0, 'upstream');
    const uv = base.uv;
    const R = Math.max(...uv.map(([u, v]) => Math.hypot(u, v))) * 1.25;
    const sx = scaleLinear().domain([-R, R]).range([20, W - 20]);
    const sy = scaleLinear().domain([-R, R]).range([H - 20, 20]);
    // contours of the fitted polynomial (frame-independent at λ = 0)
    const N = 64;
    const [c0, gu, gv, a, b, c] = base.coef;
    const grid = new Float64Array(N * N);
    for (let r = 0; r < N; r++) for (let q = 0; q < N; q++) {
      const u = -R + (2 * R * (q + 0.5)) / N, v = R - (2 * R * (r + 0.5)) / N;
      grid[r * N + q] = c0 + gu * u + gv * v + a * u * u + b * u * v + c * v * v;
    }
    const lo = Math.min(...grid), hi = Math.max(...grid);
    const levels = range(12).map((k) => lo + ((hi - lo) * (k + 0.5)) / 12);
    const proj = geoPath().projection({ stream: (s: any) => ({
      point(x: number, y: number) { s.point(20 + (x / N) * (W - 40), 20 + (y / N) * (H - 40)); },
      lineStart() { s.lineStart(); }, lineEnd() { s.lineEnd(); }, polygonStart() { s.polygonStart(); }, polygonEnd() { s.polygonEnd(); }, sphere() {},
    }) } as any);
    const interp = isDark() ? fieldInterpDark : fieldInterp;
    const clipId = `pclip-${Math.random().toString(36).slice(2, 8)}`;
    const defs = el('defs', {}, patch);
    el('circle', { cx: W / 2, cy: H / 2, r: (W - 40) / 2 }, el('clipPath', { id: clipId }, defs));
    const gField = el('g', { 'clip-path': `url(#${clipId})` }, patch);
    el('rect', { x: 0, y: 0, width: W, height: H, fill: interp(0.1), 'fill-opacity': 0.55 }, gField);
    contours().size([N, N]).thresholds(levels)(Array.from(grid)).forEach((cont, k) => {
      el('path', { d: proj(cont as any) ?? '', fill: interp(0.15 + (0.7 * k) / levels.length), 'fill-opacity': 0.55, stroke: cssVar('--paper'), 'stroke-opacity': 0.6, 'stroke-width': 0.6 }, gField);
    });
    // neighbours
    const p = ex.points[i];
    const weights = ex.index[i].map((j) => {
      const q = ex.points[j];
      const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      return Math.exp(-(d * d) / (width * width));
    });
    const gPts = el('g', {}, patch);
    uv.forEach(([u, v], j) => {
      el('circle', { cx: sx(u), cy: sy(v), r: j === 0 ? 6 : 2.5 + 4 * Math.sqrt(weights[j]), fill: j === 0 ? cssVar('--ink') : cssVar('--paper'), stroke: cssVar('--ink'), 'stroke-width': 1.3 }, gPts);
    });
    el('text', { x: 20, y: H - 8, fill: cssVar('--muted'), 'font-size': 11 }, patch).textContent = 'Point size: Gaussian weight. Bands: fitted quadratic.';
    const gAxes = el('g', { 'data-role': 'axes' }, patch);
    const L = (W / 2 - 40) * 0.62;
    for (const [name, color, rot] of [['u', '--u', 0], ['v', '--v', -90]] as const) {
      const g = el('g', { 'data-axis': name, transform: `rotate(${rot} ${W / 2} ${H / 2})` }, gAxes);
      el('line', { x1: W / 2, y1: H / 2, x2: W / 2 + L, y2: H / 2, stroke: cssVar(color), 'stroke-width': 3, 'stroke-linecap': 'round' }, g);
      el('path', { d: `M${W / 2 + L + 9},${H / 2} l-13,-6.5 v13 z`, fill: cssVar(color) }, g);
      el('text', { x: W / 2 + L + 16, y: H / 2 + 5, fill: cssVar(color), 'font-size': 16, 'font-style': 'italic', 'font-family': 'var(--serif)', 'data-label': rot }, g).textContent = name;
    }
    // curve
    samples = range(0, 361, 2).map((d) => ({ d, ...penalties(fit(i, d * DEG, 0, 'upstream').coef) }));
    const ymax = Math.max(...samples.map((s) => s.up)) * 1.15;
    const x = scaleLinear().domain([0, 360]).range([M.l, CW - M.r]);
    const y = scaleLinear().domain([0, ymax]).range([CH - M.b, M.t]);
    const axis = el('g', { class: 'axis' }, curve);
    for (const d of [0, 90, 180, 270, 360]) {
      el('line', { x1: x(d), x2: x(d), y1: M.t, y2: CH - M.b, stroke: cssVar('--rule') }, axis);
      el('text', { x: x(d), y: CH - M.b + 16, 'text-anchor': 'middle', fill: cssVar('--muted'), 'font-size': 11 }, axis).textContent = `${d}°`;
    }
    el('line', { x1: M.l, x2: CW - M.r, y1: y(0), y2: y(0), stroke: cssVar('--rule') }, axis);
    el('text', { x: M.l - 6, y: y(0) + 4, 'text-anchor': 'end', fill: cssVar('--muted'), 'font-size': 11 }, axis).textContent = '0';
    const mk = (key: 'up' | 'tensor', color: string) =>
      el('path', { d: line<typeof samples[0]>().x((s) => x(s.d)).y((s) => y(s[key]))(samples) ?? '', fill: 'none', stroke: color, 'stroke-width': 2.4 }, curve);
    mk('up', cssVar('--drift'));
    mk('tensor', cssVar('--stable'));
    const last = samples[samples.length - 1];
    el('text', { x: CW - M.r, y: y(Math.max(...samples.map((s) => s.up))) - 6, 'text-anchor': 'end', fill: cssVar('--drift'), 'font-size': 12 }, curve).textContent = 'a² + b² + c²  (upstream)';
    el('text', { x: CW - M.r, y: y(last.tensor) + 16, 'text-anchor': 'end', fill: cssVar('--stable'), 'font-size': 12 }, curve).textContent = 'a² + b²/2 + c²  (tensor)';
    const cursor = el('g', { 'data-role': 'cursor' }, curve);
    el('line', { y1: M.t, y2: CH - M.b, stroke: cssVar('--ink'), 'stroke-width': 1, 'stroke-dasharray': '3 3' }, cursor);
    el('circle', { r: 4.5, fill: cssVar('--drift'), 'data-k': 'up' }, cursor);
    el('circle', { r: 4.5, fill: cssVar('--stable'), 'data-k': 'tensor' }, cursor);
    (curve as any)._scales = { x, y };
    update();
  }

  function update() {
    const i = candidates[pick];
    const deg = theta / DEG;
    out.textContent = `${deg.toFixed(0)}°`;
    patch.querySelector('[data-role=axes]')?.setAttribute('transform', `rotate(${-deg} 200 200)`);
    patch.querySelectorAll<SVGTextElement>('[data-role=axes] text').forEach((t) => {
      const x = +t.getAttribute('x')!, y = +t.getAttribute('y')! - 5;
      t.setAttribute('transform', `rotate(${deg - +t.getAttribute('data-label')!} ${x} ${y})`);
    });
    const f = fit(i, theta, 0, 'upstream');
    const pen = penalties(f.coef);
    const { x, y } = (curve as any)._scales;
    const cursor = curve.querySelector('[data-role=cursor]')!;
    cursor.querySelector('line')!.setAttribute('x1', x(deg)); cursor.querySelector('line')!.setAttribute('x2', x(deg));
    cursor.querySelectorAll('circle').forEach((cEl) => {
      cEl.setAttribute('cx', x(deg)); cEl.setAttribute('cy', y(cEl.getAttribute('data-k') === 'up' ? pen.up : pen.tensor));
    });
    // consequence for this point's raw gradient at the real λ = 1e-3
    const grad = (th: number, pen: 'upstream' | 'tensor') => {
      const r = fit(i, th, 1e-3, pen);
      const [, gu, gv] = r.coef;
      return [gu * r.bx[0] + gv * r.by[0], gu * r.bx[1] + gv * r.by[1], gu * r.bx[2] + gv * r.by[2]];
    };
    const rel = (pen: 'upstream' | 'tensor') => {
      const g0 = grad(0, pen), g1 = grad(theta, pen);
      return Math.hypot(g1[0] - g0[0], g1[1] - g0[1], g1[2] - g0[2]) / Math.hypot(...g0);
    };
    const [, , , a, b, c] = f.coef;
    readout.innerHTML = `
      <dt>a, b, c</dt><dd>${a.toFixed(3)}, ${b.toFixed(3)}, ${c.toFixed(3)}</dd>
      <dt>Upstream penalty</dt><dd class="drift">${pen.up.toFixed(4)}</dd>
      <dt>Tensor penalty</dt><dd class="stable">${pen.tensor.toFixed(4)}</dd>
      <dt>Its gradient moved, upstream</dt><dd class="drift">${fmtRel(rel('upstream'), 3)}</dd>
      <dt>Its gradient moved, tensor</dt><dd class="stable">${fmtRel(rel('tensor'))}</dd>`;
  }

  slider.addEventListener('input', () => { stop(); theta = +slider.value * DEG; update(); });
  let stop = () => {};
  playBtn.addEventListener('click', () => {
    stop();
    const from = theta / DEG >= 359 ? 0 : theta / DEG;
    stop = sweep(from, 360, 5200 * (1 - from / 360) + 400, (d) => { theta = d * DEG; slider.value = String(d); update(); });
  });
  nextBtn.addEventListener('click', () => { stop(); pick = (pick + 1) % candidates.length; drawStatic(); });
  onThemeChange(drawStatic);
  drawStatic();
}
