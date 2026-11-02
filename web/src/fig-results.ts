import { scaleLinear, scaleLog, scaleBand, select, axisBottom, type ScaleLinear, type ScaleLogarithmic } from 'd3';
import type { Audit, AuditRow } from './data';
import { isStable, METHOD_LABEL } from './data';
import { METHOD_NAMES, type MethodName } from './mls';
import { cssVar, fmtRel, onThemeChange, segmented } from './util';

type Metric = 'gauge_normalized_relative_l2' | 'gauge_raw_relative_l2' | 'gradient_relative_l2';

const CAPTION: Record<Metric, string> = {
  gauge_normalized_relative_l2: 'Relative change of the normalised gradient when every tangent frame is rotated at random, log scale. One dot per condition. Below about 10⁻¹³ the change is floating-point rounding.',
  gauge_raw_relative_l2: 'The same test before normalisation. The tensor penalty alone (B) already brings the raw gradient to rounding level; only the upstream normaliser keeps B moving in the normalised version.',
  gradient_relative_l2: 'Relative L2 error of the raw gradient against the exact tangent gradient, linear scale. The invariance fix leaves accuracy essentially where it was.',
};

/** Figure 5: all 288 measurements as a strip plot with filters and paired highlighting. */
export function initResults(audit: Audit) {
  const fig = document.getElementById('fig-results')!;
  const svgEl = fig.querySelector<SVGSVGElement>('[data-role=chart]')!;
  const tip = fig.querySelector<HTMLElement>('[data-role=tip]')!;
  const caption = fig.querySelector<HTMLElement>('[data-role=caption]')!;
  const stage = svgEl.parentElement!;
  let metric: Metric = 'gauge_normalized_relative_l2';
  const filters: Record<string, string> = {};
  let hovered: string | null = null;

  const W = 900, H = 330, M = { l: 132, r: 24, t: 16, b: 40 };
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = select(svgEl);
  const y = scaleBand<MethodName>().domain(METHOD_NAMES).range([M.t, H - M.b]).padding(0.18);
  const jitter = (r: AuditRow) => {
    // deterministic vertical spread inside the band: by seed and noise level
    const sig = r.sigma === 0 ? 0 : r.sigma === 0.005 ? 1 : 2;
    return ((r.seed * 3 + sig) / 8 - 0.5) * y.bandwidth() * 0.78 + (r.normal_source === 'pca' ? 2 : -2);
  };
  const matches = (r: AuditRow) => Object.entries(filters).every(([k, v]) => !v || String((r as any)[k]) === v);

  fig.querySelectorAll<HTMLSelectElement>('[data-filter]').forEach((s) => s.addEventListener('change', () => {
    filters[s.dataset.filter!] = s.value; draw();
  }));
  segmented(fig.querySelector('[data-ctl=metric]')!, (v) => { metric = v as Metric; draw(); });

  function xScale(): ScaleLinear<number, number> | ScaleLogarithmic<number, number> {
    if (metric === 'gradient_relative_l2') {
      const max = Math.max(...audit.rows.map((r) => r[metric]));
      return scaleLinear().domain([0, max * 1.05]).range([M.l, W - M.r]).nice();
    }
    return scaleLog().domain([1e-16, 0.3]).range([M.l, W - M.r]).clamp(true);
  }

  function draw() {
    caption.textContent = CAPTION[metric];
    const x = xScale();
    svg.selectAll('*').remove();
    const ink = cssVar('--ink'), muted = cssVar('--muted'), rule = cssVar('--rule');
    // rounding-level band
    if (metric !== 'gradient_relative_l2') {
      svg.append('rect').attr('x', M.l).attr('y', M.t).attr('width', x(1e-13) - M.l).attr('height', H - M.b - M.t)
        .attr('fill', cssVar('--stable')).attr('fill-opacity', 0.07);
      svg.append('text').attr('x', M.l + 6).attr('y', M.t + 12).attr('fill', cssVar('--stable')).attr('font-size', 11)
        .text('floating-point rounding');
    }
    const ax = axisBottom(x as any).ticks(metric === 'gradient_relative_l2' ? 6 : 8, metric === 'gradient_relative_l2' ? '.0%' : undefined);
    if (metric !== 'gradient_relative_l2') {
      (ax as any).tickValues([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]).tickFormat((d: number) => `10${supNum(Math.round(Math.log10(d)))}`);
    }
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - M.b})`).call(ax as any);
    METHOD_NAMES.forEach((m) => {
      svg.append('line').attr('x1', M.l).attr('x2', W - M.r).attr('y1', y(m)! + y.bandwidth() / 2).attr('y2', y(m)! + y.bandwidth() / 2).attr('stroke', rule);
      svg.append('text').attr('x', M.l - 12).attr('y', y(m)! + y.bandwidth() / 2 + 4).attr('text-anchor', 'end').attr('font-size', 13)
        .attr('fill', isStable(m) ? cssVar('--stable') : cssVar('--drift')).attr('font-weight', 500).text(METHOD_LABEL[m]);
    });
    // paired line for the hovered condition
    const pair = svg.append('path').attr('fill', 'none').attr('stroke', ink).attr('stroke-width', 1.4).attr('stroke-dasharray', '3 3').attr('pointer-events', 'none');
    const dots = svg.append('g').selectAll('circle').data(audit.rows).join('circle')
      .attr('cx', (r) => x(r[metric]))
      .attr('cy', (r) => y(r.method)! + y.bandwidth() / 2 + jitter(r))
      .attr('r', 4.2)
      .attr('fill', (r) => (isStable(r.method) ? cssVar('--stable') : cssVar('--drift')))
      .attr('fill-opacity', (r) => (matches(r) ? 0.75 : 0.07))
      .attr('stroke', cssVar('--paper')).attr('stroke-width', 0.8)
      .style('cursor', 'pointer');

    // summary marks: mean of the filtered rows
    METHOD_NAMES.forEach((m) => {
      const vals = audit.rows.filter((r) => r.method === m && matches(r)).map((r) => r[metric]);
      if (!vals.length) return;
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const cy = y(m)! + y.bandwidth() / 2;
      svg.append('line').attr('x1', x(mean)).attr('x2', x(mean)).attr('y1', cy - y.bandwidth() / 2).attr('y2', cy + y.bandwidth() / 2)
        .attr('stroke', ink).attr('stroke-width', 2).attr('pointer-events', 'none');
      const right = x(mean) > W - M.r - 90;
      svg.append('text').attr('x', right ? x(mean) - 8 : x(mean) + 8).attr('y', cy + 4).attr('text-anchor', right ? 'end' : 'start')
        .attr('font-size', 11.5).attr('font-weight', 600).attr('fill', ink)
        .attr('stroke', cssVar('--paper-2')).attr('stroke-width', 4).attr('paint-order', 'stroke').attr('stroke-linejoin', 'round')
        .attr('pointer-events', 'none').text(`mean ${fmtRel(mean)}`);
    });

    const show = (r: AuditRow, ev: { clientX: number; clientY: number }) => {
      hovered = r.case_id;
      const rows = METHOD_NAMES.map((m) => audit.rows.find((q) => q.case_id === r.case_id && q.method === m)!);
      pair.attr('d', rows.map((q, i) => `${i ? 'L' : 'M'}${x(q[metric])},${y(q.method)! + y.bandwidth() / 2 + jitter(q)}`).join(''));
      dots.attr('r', (q) => (q.case_id === hovered ? 6 : 4.2)).attr('stroke', (q) => (q.case_id === hovered ? ink : cssVar('--paper')));
      tip.hidden = false;
      tip.replaceChildren();
      const head = document.createElement('div');
      head.innerHTML = `<b></b>`;
      head.querySelector('b')!.textContent = `${r.shape}, ${r.density === 'biased' ? 'biased 90/10' : 'uniform'}, σ = ${r.sigma}, ${r.normal_source === 'pca' ? 'PCA' : 'exact'} normals, seed ${r.seed}`;
      tip.append(head);
      rows.forEach((q) => {
        const line = document.createElement('div');
        line.textContent = `${METHOD_LABEL[q.method]}: ${fmtRel(q[metric])}`;
        line.style.color = isStable(q.method) ? cssVar('--stable') : cssVar('--drift');
        tip.append(line);
      });
      const box = stage.getBoundingClientRect();
      const left = Math.min(ev.clientX - box.left + 14, box.width - 280);
      tip.style.left = `${Math.max(4, left)}px`;
      tip.style.top = `${ev.clientY - box.top + 14}px`;
    };
    dots.on('pointerenter pointermove', (ev: PointerEvent, r) => show(r, ev))
      .on('pointerleave', () => {
        hovered = null; tip.hidden = true; pair.attr('d', null);
        dots.attr('r', 4.2).attr('stroke', cssVar('--paper'));
      });
  }

  // summary table
  const body = document.querySelector<HTMLElement>('[data-role=summary]')!;
  body.replaceChildren();
  for (const s of audit.summary) {
    const m = s.method as MethodName;
    const tr = document.createElement('tr');
    const cells = [METHOD_LABEL[m], fmtRel(s.gradient_relative_l2_mean as number), fmtRel(s.gauge_normalized_relative_l2_mean as number), fmtRel(s.gauge_normalized_relative_l2_max as number)];
    cells.forEach((c, i) => {
      const td = document.createElement(i ? 'td' : 'th');
      if (!i) td.setAttribute('scope', 'row');
      td.textContent = c;
      if (i >= 2) td.style.color = isStable(m) ? cssVar('--stable') : cssVar('--drift');
      tr.append(td);
    });
    body.append(tr);
  }

  onThemeChange(draw);
  draw();
}

const supNum = (n: number) => String(n).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
