import './style.css';
import katex from 'katex';
import { loadAudit, loadClouds, type CloudExample } from './data';
import { whenVisible } from './util';
import { initHero } from './fig-hero';
import { initCloud } from './fig-cloud';
import { initPenalty } from './fig-penalty';
import { initNorm } from './fig-norm';
import { initGauge } from './fig-gauge';
import { initResults } from './fig-results';

// math
document.querySelectorAll<HTMLElement>('.m').forEach((n) => katex.render(n.textContent ?? '', n, { throwOnError: false }));
document.querySelectorAll<HTMLElement>('.mb').forEach((n) => katex.render(n.textContent ?? '', n, { throwOnError: false, displayMode: true }));

// theme
document.getElementById('theme-toggle')?.addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch { /* storage unavailable */ }
  window.dispatchEvent(new Event('themechange'));
});

function failed(id: string, err: unknown) {
  console.error(err);
  const fig = document.getElementById(id);
  if (!fig) return;
  const p = document.createElement('p');
  p.className = 'small';
  p.textContent = 'This figure could not load its data. Serve the site over HTTP (for example with npm run preview) rather than opening the file directly.';
  fig.prepend(p);
}

const find = (xs: CloudExample[], shape: string, density: string) => xs.find((e) => e.shape === shape && e.density === density)!;

loadClouds().then((clouds) => {
  try { initHero(find(clouds, 'sphere', 'uniform')); } catch (e) { failed('top', e); }
  const lazy: [string, () => void][] = [
    ['fig-cloud', () => initCloud(clouds)],
    ['fig-penalty', () => initPenalty(find(clouds, 'saddle', 'uniform'))],
    ['fig-norm', () => initNorm(find(clouds, 'sphere', 'biased'))],
    ['fig-gauge', () => initGauge(clouds)],
  ];
  for (const [id, init] of lazy) {
    whenVisible(document.getElementById(id)!, () => { try { init(); } catch (e) { failed(id, e); } });
  }
}).catch((e) => ['top', 'fig-cloud', 'fig-penalty', 'fig-norm', 'fig-gauge'].forEach((id) => failed(id, e)));

loadAudit().then((audit) => {
  whenVisible(document.getElementById('fig-results')!, () => { try { initResults(audit); } catch (e) { failed('fig-results', e); } });
}).catch((e) => failed('fig-results', e));
