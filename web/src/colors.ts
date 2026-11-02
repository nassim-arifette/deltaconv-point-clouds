import * as THREE from 'three';
import { interpolateRgbBasis } from 'd3';

/** Diverging purple–green map for the scalar field (kept apart from the drift/stable accents). */
export const fieldInterp = interpolateRgbBasis(['#5b3a8e', '#9d84c4', '#e6e3ec', '#7fbf96', '#1d7a4d']);
export const fieldInterpDark = interpolateRgbBasis(['#9a7fd1', '#6f5d96', '#3a3f4a', '#4f9a6c', '#7fd6a0']);

/** Sequential map for per-point estimation error. */
export const errorInterp = interpolateRgbBasis(['#d9dee6', '#b9a6d4', '#7a4fb0', '#3d1c6e']);
export const errorInterpDark = interpolateRgbBasis(['#3a404c', '#6d5a96', '#a57de0', '#e0c8ff']);

/** Log-scaled drift map: neutral when nothing moved, amber to rust as drift grows. */
export const driftInterp = interpolateRgbBasis(['#c5cbd4', '#c5cbd4', '#f2b25c', '#d9730d', '#8a3205']);
export const driftInterpDark = interpolateRgbBasis(['#3a404c', '#3a404c', '#a8661c', '#f59f3a', '#ffd29a']);

export const isDark = () => {
  const t = document.documentElement.dataset.theme;
  if (t) return t === 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};

export const col = (css: string) => new THREE.Color(css);
