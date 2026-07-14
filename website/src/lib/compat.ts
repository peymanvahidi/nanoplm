// website/src/lib/compat.ts
// Recycling compatibility matrix, sourced from the keystone schema's
// x-nanoplm-compat block (itself derived from recycling_compatibility.yaml).
// Single source of truth = the generated schema; no separate YAML copy to drift.
import pretrain from '../generated/nanoplm.pretrain.schema.json';

export type CompatStatus = 'reject' | 'warn' | 'caveat' | 'allow';

export interface CompatFlag {
  flag: string;
  status: CompatStatus;
  reason?: string;
  message?: string;
  condition?: string;
}

export interface CompatMatrix {
  trigger: string;
  feature: string;
  model: CompatFlag[];
  pipeline: CompatFlag[];
}

interface RawRule { flag: string; reason?: string; message?: string; condition?: string }
interface RawCompat {
  trigger: string;
  feature: string;
  model_reject: RawRule[];
  model_warn: RawRule[];
  model_allow: string[];
  pipeline_reject: RawRule[];
  pipeline_caveat: RawRule[];
  pipeline_allow: string[];
}

function modelFlags(raw: RawCompat): CompatFlag[] {
  return [
    ...raw.model_reject.map((r) => ({ ...r, status: 'reject' as const })),
    ...raw.model_warn.map((r) => ({ ...r, status: 'warn' as const })),
    ...raw.model_allow.map((flag) => ({ flag, status: 'allow' as const })),
  ];
}

function pipelineFlags(raw: RawCompat): CompatFlag[] {
  return [
    ...raw.pipeline_reject.map((r) => ({ ...r, status: 'reject' as const })),
    ...raw.pipeline_caveat.map((r) => ({ ...r, status: 'caveat' as const })),
    ...raw.pipeline_allow.map((flag) => ({ flag, status: 'allow' as const })),
  ];
}

export function loadCompat(): CompatMatrix {
  const raw = (pretrain as unknown as { 'x-nanoplm-compat': RawCompat })['x-nanoplm-compat'];
  return {
    trigger: raw.trigger,
    feature: raw.feature,
    model: modelFlags(raw),
    pipeline: pipelineFlags(raw),
  };
}
