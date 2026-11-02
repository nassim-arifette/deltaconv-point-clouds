import { line, range, scaleLinear } from 'd3';
import type { CloudExample } from './data';
import { buildGradient, METHODS } from './mls';
import { cssVar, el, onThemeChange, sweep } from './util';

const DEG = Math.PI / 180;

/**
 * Figure 3: the stencil of the point that sets the global upstream scale.
 * Stencil vectors come from the real operator with the tensor penalty, which makes them
 * exactly equivariant: turning the frame by θ rotates their coordinates by −θ.
 */
export function initNorm(ex: CloudExample) {
  const fig = document.getElementById('fig-norm')!;
  const svg = fig.querySelector<SVGSVGElement>('[data-role=stencil]')!;
  const curve = fig.querySelector<SVGSVGElement>('[data-role=curve]')!;
  const readout = fig.querySelector<HTMLElement>('[data-role=readout]')!;
  const slider = fig.querySelector<HTMLInputElement>('[data-ctl=theta]')!;
  const out = fig.querySelector<HTMLOutputElement>('[data-out=theta]')!;
  const playBtn = fig.querySelector<HTMLButtonElement>('[data-ctl=play]')!;

  const g = buildGradient(ex.points, ex.normals, ex.index, METHODS.B_tensor_penalty, ex.regularizer);
  const k = g.k;
  // point with the largest upstream per-point scale
  let best = 0, bestI = 0;
  for (let i = 0; i < g.n; i++) {
    let su = 0, sv = 0;
    for (let j = 0; j < k; j++) { su += Math.abs(g.coefficients[(i * k + j) * 2]); sv += Math.abs(g.coefficients[(i * k + j) * 2 + 1]); }
    const s = Math.hypot(su, sv);
    if (s > best) { best = s; bestI = i; }
  }
  const stencil = range(k).map((j) => [g.coefficients[(bestI * k + j) * 2], g.coefficients[(bestI * k + j) * 2 + 1]] as [number, number]);
  const invariant = stencil.reduce((s, [a, b]) => s + Math.hypot(a, b), 0);
  const sums = (theta: number) => {
    const c = Math.cos(theta), s = Math.sin(theta);
    let su = 0, sv = 0;
    for (const [a, b] of stencil) { su += Math.abs(c * a + s * b); sv += Math.abs(-s * a + c * b); }
    return { su, sv, up: Math.hypot(su, sv) };
  };
  const samples = range(0, 361, 1).map((d) => ({ d, ...sums(d * DEG) }));
  let theta = 0;

  const W = 400, H = 400, cx = 150, cy = 210;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const CW = 400, CH = 210, M = { l: 40, r: 14, t: 14, b: 30 };
  curve.setAttribute('viewBox', `0 0 ${CW} ${CH}`);
  const maxArrow = Math.max(...stencil.map(([a, b]) => Math.hypot(a, b)));
  const arrowScale = 128 / maxArrow;
  const boxScale = 72 / invariant; // inset: the sums drawn at their own scale

  function draw() {
    svg.replaceChildren();
    curve.replaceChildren();
    const ink = cssVar('--ink'), muted = cssVar('--muted');
    // world-fixed stencil arrows
    const gArr = el('g', {}, svg);
    for (const [a, b] of stencil) {
      const x2 = cx + a * arrowScale, y2 = cy - b * arrowScale;
      const ang = Math.atan2(-(y2 - cy), x2 - cx);
      el('line', { x1: cx, y1: cy, x2, y2, stroke: ink, 'stroke-width': 1.4, 'stroke-opacity': 0.85 }, gArr);
      el('path', { d: `M${x2},${y2} l${-7 * Math.cos(ang - 0.4)},${7 * Math.sin(ang - 0.4)} M${x2},${y2} l${-7 * Math.cos(ang + 0.4)},${7 * Math.sin(ang + 0.4)}`, stroke: ink, 'stroke-width': 1.4, fill: 'none' }, gArr);
    }
    const gAxes = el('g', { 'data-role': 'axes' }, svg);
    for (const [name, color, rot] of [['u', '--u', 0], ['v', '--v', -90]] as const) {
      const gg = el('g', { transform: `rotate(${rot} ${cx} ${cy})` }, gAxes);
      el('line', { x1: cx - 135, y1: cy, x2: cx + 135, y2: cy, stroke: cssVar(color), 'stroke-width': 1, 'stroke-dasharray': '4 4', 'stroke-opacity': 0.8 }, gg);
      el('text', { x: cx + 142, y: cy + 5, fill: cssVar(color), 'font-size': 16, 'font-style': 'italic', 'font-family': 'var(--serif)', 'data-label': '1' }, gg).textContent = name;
    }
    // inset: per-axis sums (box) vs sum of lengths (circle), lower right
    const ix = 300, iy = 330;
    const inset = el('g', {}, svg);
    el('circle', { cx: ix, cy: iy, r: invariant * boxScale, fill: 'none', stroke: cssVar('--stable'), 'stroke-width': 2 }, inset);
    el('g', { 'data-role': 'box' }, inset);
    el('circle', { cx: ix, cy: iy, r: 2.5, fill: ink }, inset);
    el('text', { x: ix, y: iy - invariant * boxScale - 8, 'text-anchor': 'middle', fill: muted, 'font-size': 11 }, inset).textContent = 'sums of the stencil';
    el('text', { x: 12, y: 22, fill: muted, 'font-size': 11 }, svg).textContent = `${k} stencil vectors gᵢⱼ, fixed in space`;
    (svg as any)._inset = { ix, iy };

    const x = scaleLinear().domain([0, 360]).range([M.l, CW - M.r]);
    // zoomed axis: the variation is a few percent, so start just below the minimum
    const lo = Math.min(...samples.map((s) => s.up)), hi = Math.max(invariant, ...samples.map((s) => s.up));
    const pad = (hi - lo) * 0.25;
    const y = scaleLinear().domain([lo - pad, hi + pad]).range([CH - M.b, M.t]).nice();
    const axis = el('g', { class: 'axis' }, curve);
    for (const d of [0, 90, 180, 270, 360]) {
      el('line', { x1: x(d), x2: x(d), y1: M.t, y2: CH - M.b, stroke: cssVar('--rule') }, axis);
      el('text', { x: x(d), y: CH - M.b + 16, 'text-anchor': 'middle', fill: muted, 'font-size': 11 }, axis).textContent = `${d}°`;
    }
    for (const tv of y.ticks(4)) {
      el('line', { x1: M.l, x2: CW - M.r, y1: y(tv), y2: y(tv), stroke: cssVar('--rule'), 'stroke-dasharray': '2 3' }, axis);
      el('text', { x: M.l - 6, y: y(tv) + 4, 'text-anchor': 'end', fill: muted, 'font-size': 11 }, axis).textContent = String(tv);
    }
    el('path', { d: line<typeof samples[0]>().x((s) => x(s.d)).y((s) => y(s.up))(samples) ?? '', fill: 'none', stroke: cssVar('--drift'), 'stroke-width': 2.4 }, curve);
    el('line', { x1: M.l, x2: CW - M.r, y1: y(invariant), y2: y(invariant), stroke: cssVar('--stable'), 'stroke-width': 2.4 }, curve);
    el('text', { x: CW - M.r, y: y(invariant) - 7, 'text-anchor': 'end', fill: cssVar('--stable'), 'font-size': 12 }, curve).textContent = 'Σ ‖gᵢⱼ‖  (invariant)';
    el('text', { x: CW - M.r, y: y(Math.min(...samples.map((s) => s.up))) + 17, 'text-anchor': 'end', fill: cssVar('--drift'), 'font-size': 12 }, curve).textContent = '‖(Σ|gᵘ|, Σ|gᵛ|)‖  (upstream)';
    const cursor = el('g', { 'data-role': 'cursor' }, curve);
    el('line', { y1: M.t, y2: CH - M.b, stroke: ink, 'stroke-dasharray': '3 3' }, cursor);
    el('circle', { r: 4.5, fill: cssVar('--drift') }, cursor);
    (curve as any)._s = { x, y };
    update();
  }

  function update() {
    const deg = theta / DEG;
    out.textContent = `${deg.toFixed(0)}°`;
    svg.querySelector('[data-role=axes]')!.setAttribute('transform', `rotate(${-deg} ${cx} ${cy})`);
    svg.querySelectorAll<SVGTextElement>('[data-label]').forEach((t) => {
      const parentRot = t.parentElement!.getAttribute('transform')!.match(/rotate\((-?\d+)/)![1];
      t.setAttribute('transform', `rotate(${deg - +parentRot} ${t.getAttribute('x')} ${+t.getAttribute('y')! - 5})`);
    });
    const s = sums(theta);
    const { ix, iy } = (svg as any)._inset;
    const box = svg.querySelector('[data-role=box]')!;
    box.replaceChildren();
    const bw = s.su * boxScale, bh = s.sv * boxScale;
    const gb = el('g', { transform: `rotate(${-deg} ${ix} ${iy})` }, box);
    el('rect', { x: ix, y: iy - bh, width: bw, height: bh, fill: cssVar('--drift'), 'fill-opacity': 0.14, stroke: cssVar('--drift'), 'stroke-width': 1.6 }, gb);
    el('line', { x1: ix, y1: iy, x2: ix + bw, y2: iy - bh, stroke: cssVar('--drift'), 'stroke-width': 2.4 }, gb);
    el('line', { x1: ix, y1: iy, x2: ix + bw, y2: iy, stroke: cssVar('--u'), 'stroke-width': 3 }, gb);
    el('line', { x1: ix, y1: iy, x2: ix, y2: iy - bh, stroke: cssVar('--v'), 'stroke-width': 3 }, gb);
    const { x, y } = (curve as any)._s;
    const cur = curve.querySelector('[data-role=cursor]')!;
    cur.querySelector('line')!.setAttribute('x1', x(deg)); cur.querySelector('line')!.setAttribute('x2', x(deg));
    const dot = cur.querySelector('circle')!; dot.setAttribute('cx', x(deg)); dot.setAttribute('cy', y(s.up));
    const lo = Math.min(...samples.map((q) => q.up)), hi = Math.max(...samples.map((q) => q.up));
    readout.innerHTML = `
      <dt>Σ|gᵘ| along u</dt><dd>${s.su.toFixed(3)}</dd>
      <dt>Σ|gᵛ| along v</dt><dd>${s.sv.toFixed(3)}</dd>
      <dt>Upstream scale of this point</dt><dd class="drift">${s.up.toFixed(3)}</dd>
      <dt>Invariant scale of this point</dt><dd class="stable">${invariant.toFixed(3)}</dd>
      <dt>Upstream range over all angles</dt><dd>${lo.toFixed(3)} to ${hi.toFixed(3)} (±${((hi - lo) / (hi + lo) * 100).toFixed(1)} %)</dd>`;
  }

  let stop = () => {};
  slider.addEventListener('input', () => { stop(); theta = +slider.value * DEG; update(); });
  playBtn.addEventListener('click', () => {
    stop();
    const from = theta / DEG >= 359 ? 0 : theta / DEG;
    stop = sweep(from, 360, 5200 * (1 - from / 360) + 400, (d) => { theta = d * DEG; slider.value = String(d); update(); });
  });
  onThemeChange(draw);
  draw();
}
