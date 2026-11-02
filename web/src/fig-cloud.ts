import type { CloudExample } from './data';
import { exampleLabel } from './data';
import { applyGradient, buildGradient, METHODS, type Vec3 } from './mls';
import { Arrows, frameCloud, median, norm3, PointSprites, Viewer } from './three-view';
import { col, errorInterp, errorInterpDark, fieldInterp, fieldInterpDark, isDark } from './colors';
import { cssVar, onThemeChange, segmented } from './util';

/** Figure 1: the field, its exact gradient and the upstream MLS estimate on a saved audit cloud. */
export function initCloud(examples: CloudExample[]) {
  const fig = document.getElementById('fig-cloud')!;
  const canvas = fig.querySelector<HTMLCanvasElement>('canvas')!;
  const view = new Viewer(canvas);
  view.camera.up.set(0, 0, 1);
  view.camera.position.set(2.8, 2.2, 1.9);

  const select = fig.querySelector<HTMLSelectElement>('[data-ctl=example]')!;
  examples.forEach((e, i) => select.add(new Option(exampleLabel(e), String(i))));

  let ex = examples[0];
  let arrowsMode = 'estimate';
  let colourMode = 'field';
  let dots: PointSprites | null = null;
  let arrows: Arrows | null = null;
  const estimates = new Map<string, Vec3[]>();

  const estimate = (e: CloudExample) => {
    if (!estimates.has(e.case_id)) {
      const g = buildGradient(e.points, e.normals, e.index, METHODS.A_upstream, e.regularizer);
      estimates.set(e.case_id, applyGradient(g, e.signal));
    }
    return estimates.get(e.case_id)!;
  };

  function rebuild() {
    if (dots) view.scene.remove(dots.mesh);
    if (arrows) view.scene.remove(arrows.group);
    dots = new PointSprites(ex.points.length, 0.02);
    dots.setPositions(ex.points);
    arrows = new Arrows(ex.points.length, cssVar('--arrow'), 0.005);
    view.scene.add(dots.mesh, arrows.group);
    frameCloud(view.camera, ex.points, view.controls, ex.shape === 'sphere' ? 4.4 : 3.6);
    update();
  }

  function update() {
    if (!dots || !arrows) return;
    const est = estimate(ex);
    const shown = arrowsMode === 'truth' ? ex.truth : est;
    const scale = 0.11 / median(ex.truth.map(norm3));
    arrows.set(ex.points, shown, scale, ex.normals, 0.0);
    arrows.material.color.set(cssVar('--arrow'));
    const dark = isDark();
    if (colourMode === 'field') {
      const lo = Math.min(...ex.signal), hi = Math.max(...ex.signal);
      const f = dark ? fieldInterpDark : fieldInterp;
      dots.setColors(ex.signal.map((v) => col(f((v - lo) / (hi - lo)))));
    } else {
      const err = est.map((g, i) => norm3([g[0] - ex.truth[i][0], g[1] - ex.truth[i][1], g[2] - ex.truth[i][2]]) / norm3(ex.truth[i]));
      const f = dark ? errorInterpDark : errorInterp;
      dots.setColors(err.map((e) => col(f(Math.min(1, e / 0.5)))));
    }
    const caption = fig.querySelector('[data-caption]')!;
    caption.textContent = colourMode === 'field'
      ? `The field f = sin 2x + yz/4 on the cloud “${exampleLabel(ex).toLowerCase()}”, purple for low values and green for high.`
      : `Relative error of the MLS estimate at each point on the cloud “${exampleLabel(ex).toLowerCase()}”, from grey (exact) to dark purple (50 % or more).`;
    view.invalidate();
  }

  select.addEventListener('change', () => { ex = examples[+select.value]; rebuild(); });
  segmented(fig.querySelector('[data-ctl=arrows]')!, (v) => { arrowsMode = v; update(); });
  segmented(fig.querySelector('[data-ctl=colour]')!, (v) => { colourMode = v; update(); });
  onThemeChange(update);
  rebuild();
}
