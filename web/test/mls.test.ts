import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { applyGradient, buildGradient, METHOD_NAMES, METHODS, relativeL2, type Vec3 } from '../src/mls';

const clouds = JSON.parse(readFileSync(new URL('../public/data/clouds.json', import.meta.url), 'utf8'));
const ref = JSON.parse(readFileSync(new URL('./fixtures/reference_gradients.json', import.meta.url), 'utf8'));

describe('TypeScript MLS gradient matches the NumPy audit', () => {
  for (const ex of clouds.examples) {
    for (const name of METHOD_NAMES) {
      it(`${ex.case_id} / ${name}`, () => {
        const settings = METHODS[name];
        const g = buildGradient(ex.points, ex.normals, ex.index, settings, ex.regularizer);
        const grad = applyGradient(g, ex.signal);
        const expected: Vec3[] = ref.examples.find((e: any) => e.case_id === ex.case_id).gradients[name];
        expect(relativeL2(grad, expected)).toBeLessThan(1e-11);
        const m = ex.measured[name];
        expect(g.scale).toBeCloseTo(m.normalization_scale, 9);
        expect(relativeL2(grad, ex.truth)).toBeCloseTo(m.gradient_relative_l2, 10);
        const rotated = buildGradient(ex.points, ex.normals, ex.index, settings, ex.regularizer, ex.angles);
        const driftN = relativeL2(applyGradient(rotated, ex.signal, true), applyGradient(g, ex.signal, true));
        if (m.gauge_normalized_relative_l2 > 1e-10) {
          expect(driftN / m.gauge_normalized_relative_l2).toBeCloseTo(1, 6);
        } else {
          expect(driftN).toBeLessThan(1e-13);
        }
      });
    }
  }
});
