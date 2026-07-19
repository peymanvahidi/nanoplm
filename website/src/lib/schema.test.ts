// website/src/lib/schema.test.ts
import { describe, it, expect } from 'vitest';
import { toRows, stripEnumRestatement, type Surface, type SectionSchema } from './schema';
import pretrain from '../generated/nanoplm.pretrain.schema.json';
import evalSchema from '../generated/nanoplm.eval.schema.json';

describe('toRows', () => {
  it('flattens a section into reference rows with default/type/description', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    const mb = rows.find((r) => r.key === 'micro_batch_seqs')!;
    expect(mb.type).toBe('integer');
    // Compared against the schema rather than a literal: the value is the tool's
    // to change, and pinning it here turns every legitimate `schema_gen` run into
    // a test failure. What is being tested is that toRows carries the default
    // through, not what the default happens to be.
    const source = (pretrain as Surface).properties.pretraining as SectionSchema;
    expect(mb.default).toBe(source.properties.micro_batch_seqs.default);
    expect(mb.default).toEqual(expect.any(Number));
    expect(mb.description).toBeTruthy();
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

  it('drops descriptions that only restate the enum', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    // "adamw | stable_adamw | muon | ..." — the choice list already says this.
    expect(rows.find((r) => r.key === 'optimizer')?.description).toBeUndefined();
    // "Duration + LR schedule (token-budget surface) wsd | warmup_stable | ..."
    expect(rows.find((r) => r.key === 'lr_schedule')?.description).toBe(
      'Duration + LR schedule (token-budget surface)',
    );
  });

  it('keeps descriptions that explain the values rather than list them', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    // Per-value meaning is information the choice chips cannot carry.
    expect(rows.find((r) => r.key === 'mlm_masking_strategy')?.description).toContain(
      'BERT-style',
    );
    expect(rows.find((r) => r.key === 'muonh_scope')?.description).toContain('MLP matrices');
    const model = toRows(pretrain as Surface, 'model');
    // The run sits mid-sentence; removing it would leave "(allowed: , default: 5)".
    expect(model.find((r) => r.key === 'canon_layers_kernel_size')?.description).toBe(
      'symmetric Canon kernel size (allowed: 3/5/7, default: 5)',
    );
  });

  describe('stripEnumRestatement', () => {
    it('strips a leading or trailing run and tidies the punctuation holding it', () => {
      expect(stripEnumRestatement('a | b | c', ['a', 'b', 'c'])).toBe('');
      expect(stripEnumRestatement('mode: a | b', ['a', 'b'])).toBe('mode');
      expect(stripEnumRestatement('(a | b), a is faster', ['a', 'b'])).toBe('a is faster');
      expect(stripEnumRestatement('a | b (b has no bias)', ['a', 'b'])).toBe('b has no bias');
      expect(stripEnumRestatement('TE only: a | b (b is new)', ['a', 'b'])).toBe(
        'TE only (b is new)',
      );
      expect(stripEnumRestatement('Scores it. One of: a, b.', ['a', 'b'])).toBe('Scores it.');
    });

    it('leaves a run alone unless it lists every value at an edge', () => {
      // Partial list — the rest of the sentence may depend on it.
      expect(stripEnumRestatement('a | b then something', ['a', 'b', 'c'])).toBe(
        'a | b then something',
      );
      // Mid-sentence.
      expect(stripEnumRestatement('size (allowed: a/b, default: b)', ['a', 'b'])).toBe(
        'size (allowed: a/b, default: b)',
      );
      // Values carry their own explanations.
      expect(stripEnumRestatement('a (fast) | b (slow)', ['a', 'b'])).toBe('a (fast) | b (slow)');
    });

    it('respects word boundaries so a value does not match inside a longer token', () => {
      // `decay` must not match inside `decay_tokens`.
      const desc = 'one of decay_tokens / decay_steps. continue | decay';
      expect(stripEnumRestatement(desc, ['continue', 'decay'])).toBe(
        'one of decay_tokens / decay_steps.',
      );
    });
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
