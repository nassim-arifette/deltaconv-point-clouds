/**
 * TypeScript port of deltaconv_audit/operators.py (itself a NumPy transcription of
 * rubenwiersma/deltaconv@186fec3 deltaconv/geometry/grad_div_mls.py).
 *
 * Float64 throughout, so drift values near 1e-15 are meaningful. Parity with the
 * NumPy reference is checked in test/mls.test.ts.
 */
export type Vec3 = [number, number, number];

export type Penalty = 'upstream' | 'tensor';
export type Normalization = 'upstream' | 'invariant' | 'none';
export type Bandwidth = 'global' | 'local';

export interface MethodSettings {
  penalty: Penalty;
  normalization: Normalization;
  bandwidth: Bandwidth;
}

export const METHODS = {
  A_upstream: { penalty: 'upstream', normalization: 'upstream', bandwidth: 'global' },
  B_tensor_penalty: { penalty: 'tensor', normalization: 'upstream', bandwidth: 'global' },
  C_invariant: { penalty: 'tensor', normalization: 'invariant', bandwidth: 'global' },
  D_local_bandwidth: { penalty: 'tensor', normalization: 'invariant', bandwidth: 'local' },
} as const satisfies Record<string, MethodSettings>;

export type MethodName = keyof typeof METHODS;
export const METHOD_NAMES = Object.keys(METHODS) as MethodName[];

/** Tangent basis per point: x and y axes, each a Vec3. Mirrors operators.tangent_basis. */
export function tangentBasis(normals: Vec3[], angles?: ArrayLike<number>): { x: Vec3[]; y: Vec3[] } {
  const xs: Vec3[] = [];
  const ys: Vec3[] = [];
  for (let i = 0; i < normals.length; i++) {
    const [a, b, c] = normals[i];
    const len = Math.hypot(a, b, c);
    const n: Vec3 = [a / len, b / len, c / len];
    const axis: Vec3 = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
    let x = cross(axis, n);
    const xl = Math.hypot(...x);
    x = [x[0] / xl, x[1] / xl, x[2] / xl];
    let y = cross(n, x);
    if (angles) {
      const cs = Math.cos(angles[i]);
      const sn = Math.sin(angles[i]);
      const nx: Vec3 = [cs * x[0] + sn * y[0], cs * x[1] + sn * y[1], cs * x[2] + sn * y[2]];
      const ny: Vec3 = [-sn * x[0] + cs * y[0], -sn * x[1] + cs * y[1], -sn * x[2] + cs * y[2]];
      x = nx;
      y = ny;
    }
    xs.push(x);
    ys.push(y);
  }
  return { x: xs, y: ys };
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Solve A X = B in place for a small dense system (Gaussian elimination, partial pivoting). */
export function solve(A: Float64Array, B: Float64Array, n: number, m: number): Float64Array {
  const a = A.slice();
  const b = B.slice();
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(a[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(a[r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best === 0) throw new Error('singular MLS system');
    if (piv !== col) {
      for (let c = 0; c < n; c++) { const t = a[col * n + c]; a[col * n + c] = a[piv * n + c]; a[piv * n + c] = t; }
      for (let c = 0; c < m; c++) { const t = b[col * m + c]; b[col * m + c] = b[piv * m + c]; b[piv * m + c] = t; }
    }
    const d = a[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = a[r * n + col] / d;
      if (f === 0) continue;
      for (let c = col; c < n; c++) a[r * n + c] -= f * a[col * n + c];
      for (let c = 0; c < m; c++) b[r * m + c] -= f * b[col * m + c];
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    for (let c = 0; c < m; c++) {
      let s = b[r * m + c];
      for (let k = r + 1; k < n; k++) s -= a[r * n + k] * b[k * m + c];
      b[r * m + c] = s / a[r * n + r];
    }
  }
  return b;
}

export interface Gradient {
  n: number;
  k: number;
  index: number[][];
  /** coefficients[(i*k + j)*2 + d]: weight of neighbour j on tangent component d at point i. */
  coefficients: Float64Array;
  basis: { x: Vec3[]; y: Vec3[] };
  scale: number;
}

export function buildGradient(
  points: Vec3[], normals: Vec3[], index: number[][], settings: MethodSettings,
  regularizer = 1e-3, angles?: ArrayLike<number>,
): Gradient {
  const n = points.length;
  const k = index[0].length;
  const basis = tangentBasis(normals, angles);
  // Distances and Gaussian width.
  const dist = new Float64Array(n * k);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const p = points[i];
    for (let j = 0; j < k; j++) {
      const q = points[index[i][j]];
      const d = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      dist[i * k + j] = d;
      total += d;
    }
  }
  const globalWidth = total / (n * k);
  const diag = settings.penalty === 'upstream' ? [1, 1, 1, 1, 1, 1] : [1, 1, 1, 1, 0.5, 1];
  const coefficients = new Float64Array(n * k * 2);
  const gram = new Float64Array(36);
  const BT = new Float64Array(6 * k);
  const w = new Float64Array(k);
  const tmpB = new Float64Array(6 * k);
  for (let i = 0; i < n; i++) {
    let width = globalWidth;
    if (settings.bandwidth === 'local') {
      let s = 0;
      for (let j = 0; j < k; j++) s += dist[i * k + j];
      width = s / k;
    }
    let wsum = 0;
    for (let j = 0; j < k; j++) {
      const d = dist[i * k + j];
      w[j] = Math.exp(-(d * d) / (width * width));
      wsum += w[j];
    }
    wsum = Math.max(wsum, 1e-5);
    const p = points[i];
    const bx = basis.x[i];
    const by = basis.y[i];
    for (let j = 0; j < k; j++) {
      const q = points[index[i][j]];
      const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
      const u = dx * bx[0] + dy * bx[1] + dz * bx[2];
      const v = dx * by[0] + dy * by[1] + dz * by[2];
      const wj = w[j] / wsum;
      const row = [1, u, v, u * u, u * v, v * v];
      for (let r = 0; r < 6; r++) BT[r * k + j] = wj * row[r];
      // keep raw B row for the Gram product below
      tmpB[j * 6 + 0] = 1; tmpB[j * 6 + 1] = u; tmpB[j * 6 + 2] = v;
      tmpB[j * 6 + 3] = u * u; tmpB[j * 6 + 4] = u * v; tmpB[j * 6 + 5] = v * v;
    }
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        let s = 0;
        for (let j = 0; j < k; j++) s += BT[r * k + j] * tmpB[j * 6 + c];
        gram[r * 6 + c] = s + (r === c ? regularizer * diag[r] : 0);
      }
    }
    const X = solve(gram, BT, 6, k);
    for (let j = 0; j < k; j++) {
      coefficients[(i * k + j) * 2] = X[1 * k + j];
      coefficients[(i * k + j) * 2 + 1] = X[2 * k + j];
    }
  }
  let scale = 1;
  if (settings.normalization !== 'none') {
    let best = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      if (settings.normalization === 'upstream') {
        let su = 0, sv = 0;
        for (let j = 0; j < k; j++) {
          su += Math.abs(coefficients[(i * k + j) * 2]);
          sv += Math.abs(coefficients[(i * k + j) * 2 + 1]);
        }
        s = Math.hypot(su, sv);
      } else {
        for (let j = 0; j < k; j++) {
          s += Math.hypot(coefficients[(i * k + j) * 2], coefficients[(i * k + j) * 2 + 1]);
        }
      }
      if (s > best) best = s;
    }
    scale = best > 1e-5 ? best : 1;
  }
  return { n, k, index, coefficients, basis, scale };
}


/** Ambient 3D gradient of a scalar signal. */
export function applyGradient(g: Gradient, signal: ArrayLike<number>, normalized = false): Vec3[] {
  const out: Vec3[] = [];
  for (let i = 0; i < g.n; i++) {
    let lu = 0, lv = 0;
    for (let j = 0; j < g.k; j++) {
      const f = signal[g.index[i][j]];
      lu += g.coefficients[(i * g.k + j) * 2] * f;
      lv += g.coefficients[(i * g.k + j) * 2 + 1] * f;
    }
    const x = g.basis.x[i], y = g.basis.y[i];
    const s = normalized ? g.scale : 1;
    out.push([(x[0] * lu + y[0] * lv) / s, (x[1] * lu + y[1] * lv) / s, (x[2] * lu + y[2] * lv) / s]);
  }
  return out;
}

export function relativeL2(actual: Vec3[], expected: Vec3[], mask?: ArrayLike<number>, keep = 1): number {
  let num = 0, den = 0;
  for (let i = 0; i < actual.length; i++) {
    if (mask && mask[i] !== keep) continue;
    const a = actual[i], e = expected[i];
    num += (a[0] - e[0]) ** 2 + (a[1] - e[1]) ** 2 + (a[2] - e[2]) ** 2;
    den += e[0] ** 2 + e[1] ** 2 + e[2] ** 2;
  }
  return Math.sqrt(num) / Math.max(Math.sqrt(den), 1e-12);
}

/**
 * Least-squares quadratic fit of one neighbourhood in (u, v) coordinates.
 * Returns the 6 polynomial coefficients [c0, gu, gv, a, b, c] for q = a u² + b uv + c v².
 */
export function localFit(
  points: Vec3[], index: number[], center: number, bx: Vec3, by: Vec3, signal: ArrayLike<number>,
  regularizer: number, penalty: Penalty, width: number,
): { coef: number[]; uv: [number, number][] } {
  const k = index.length;
  const p = points[center];
  const uv: [number, number][] = [];
  const w: number[] = [];
  let wsum = 0;
  for (let j = 0; j < k; j++) {
    const q = points[index[j]];
    const dx = q[0] - p[0], dy = q[1] - p[1], dz = q[2] - p[2];
    uv.push([dx * bx[0] + dy * bx[1] + dz * bx[2], dx * by[0] + dy * by[1] + dz * by[2]]);
    const d = Math.hypot(dx, dy, dz);
    w.push(Math.exp(-(d * d) / (width * width)));
    wsum += w[j];
  }
  const diag = penalty === 'upstream' ? [1, 1, 1, 1, 1, 1] : [1, 1, 1, 1, 0.5, 1];
  const gram = new Float64Array(36);
  const rhs = new Float64Array(6);
  for (let j = 0; j < k; j++) {
    const [u, v] = uv[j];
    const row = [1, u, v, u * u, u * v, v * v];
    const wj = w[j] / Math.max(wsum, 1e-5);
    for (let r = 0; r < 6; r++) {
      rhs[r] += wj * row[r] * signal[index[j]];
      for (let c = 0; c < 6; c++) gram[r * 6 + c] += wj * row[r] * row[c];
    }
  }
  for (let r = 0; r < 6; r++) gram[r * 6 + r] += regularizer * diag[r];
  return { coef: Array.from(solve(gram, rhs, 6, 1)), uv };
}

/** Mean neighbour distance of a whole cloud: the global Gaussian width. */
export function globalWidth(points: Vec3[], index: number[][]): number {
  let total = 0, count = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    for (const j of index[i]) {
      const q = points[j];
      total += Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      count++;
    }
  }
  return total / count;
}
