// website/src/lib/schema.test.ts
import { describe, it, expect } from 'vitest';
import { toRows, type Surface } from './schema';
import pretrain from '../generated/nanoplm.pretrain.schema.json';
import evalSchema from '../generated/nanoplm.eval.schema.json';

describe('toRows', () => {
  it('flattens a section into reference rows with default/type/description', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    const mb = rows.find((r) => r.key === 'micro_batch_seqs')!;
    expect(mb.type).toBe('integer');
    expect(mb.default).toBe(64);
  });

  it('surfaces enum and backend tags', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    const opt = rows.find((r) => r.key === 'optimizer')!;
    expect(opt.enum).toContain('normuon');
    const model = toRows(pretrain as Surface, 'model');
    const canon = model.find((r) => r.key === 'use_canon_layers')!;
    expect(canon.backend).toContain('pure_torch');
  });

  it('includes the commented-out duration budgets, marked advanced', () => {
    // train_tokens/max_steps/stable_steps/decay_steps ship commented out in the
    // get-yaml template (each is mutually exclusive with its *_tokens twin), so
    // they reach the schema via overlay.yaml's x-extra-fields.
    const rows = toRows(pretrain as Surface, 'pretraining');
    const tt = rows.find((r) => r.key === 'train_tokens')!;
    expect(tt.type).toBe('integer');
    expect(tt.advanced).toBe(true);
    expect(tt.default).toBeNull();
    for (const key of ['max_steps', 'stable_steps', 'decay_steps']) {
      expect(rows.find((r) => r.key === key)?.advanced).toBe(true);
    }
    const resume = toRows(pretrain as Surface, 'resume');
    expect(resume.find((r) => r.key === 'decay_steps')?.advanced).toBe(true);
  });

  it('surfaces a recycling visibility gate', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    const rb = rows.find((r) => r.key === 'recycling_bptt')!;
    expect(rb.visibleWhen?.field).toBe('recycling_loop_count');
  });

  it('reads a flat (sectionless) eval surface and marks required fields', () => {
    const rows = toRows(evalSchema as Surface);
    const cd = rows.find((r) => r.key === 'checkpoint_dir')!;
    expect(cd.required).toBe(true);
    expect(cd.default).toBeUndefined();
    const fw = rows.find((r) => r.key === 'framework')!;
    expect(fw.type).toBe('array');
  });
});
