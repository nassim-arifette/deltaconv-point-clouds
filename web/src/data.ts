import type { MethodName, Vec3 } from './mls';

export interface CloudExample {
  case_id: string;
  shape: 'sphere' | 'saddle';
  density: 'uniform' | 'biased';
  seed: number;
  sigma: number;
  normal_source: string;
  k: number;
  regularizer: number;
  points: Vec3[];
  normals: Vec3[];
  signal: number[];
  truth: Vec3[];
  angles: number[];
  index: number[][];
  sparse: number[];
  measured: Record<MethodName, Record<string, number>>;
}

export interface AuditRow {
  case_id: string; shape: string; density: string; seed: number; sigma: number; normal_source: string;
  method: MethodName;
  gradient_relative_l2: number; sparse_gradient_relative_l2: number; dense_gradient_relative_l2: number;
  gauge_raw_relative_l2: number; gauge_normalized_relative_l2: number; condition_p95: number; normalization_scale: number;
}

export interface Audit {
  rows: AuditRow[];
  summary: Array<Record<string, number | string>>;
  wall_seconds: number;
}

const base = import.meta.env.BASE_URL;
let cloudsPromise: Promise<CloudExample[]> | null = null;
let auditPromise: Promise<Audit> | null = null;

export function loadClouds(): Promise<CloudExample[]> {
  cloudsPromise ??= fetch(`${base}data/clouds.json`).then((r) => {
    if (!r.ok) throw new Error(`clouds.json: HTTP ${r.status}`);
    return r.json();
  }).then((d) => d.examples);
  return cloudsPromise;
}

export function loadAudit(): Promise<Audit> {
  auditPromise ??= fetch(`${base}data/audit.json`).then((r) => {
    if (!r.ok) throw new Error(`audit.json: HTTP ${r.status}`);
    return r.json();
  });
  return auditPromise;
}

export function exampleLabel(e: Pick<CloudExample, 'shape' | 'density' | 'sigma'>): string {
  const shape = e.shape === 'sphere' ? 'Sphere' : 'Saddle';
  const density = e.density === 'uniform' ? 'uniform sampling' : 'biased 90/10 sampling';
  const noise = e.sigma === 0 ? 'no noise' : `noise σ = ${e.sigma}`;
  return `${shape}, ${density}, ${noise}`;
}

export const METHOD_LABEL: Record<MethodName, string> = {
  A_upstream: 'A · upstream',
  B_tensor_penalty: 'B · tensor penalty',
  C_invariant: 'C · invariant',
  D_local_bandwidth: 'D · local width',
};
export const METHOD_LETTER: Record<MethodName, string> = {
  A_upstream: 'A', B_tensor_penalty: 'B', C_invariant: 'C', D_local_bandwidth: 'D',
};
export const isStable = (m: MethodName) => m === 'C_invariant' || m === 'D_local_bandwidth';
