// website/src/lib/schema.test.ts
import { describe, it, expect } from 'vitest';
import {
  toRows,
  toGroups,
  groupSlug,
  stripEnumRestatement,
  type Surface,
  type SectionSchema,
} from './schema';
import { withFieldGroups, type TocItem } from './toc';
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
    // No per-field description on purpose: the template documents this one in
    // its block's intro prose, which now lives on the group (see toGroups).
    expect(rows.find((r) => r.key === 'run_name')?.description).toBeTruthy();
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
    // "adamw | stable_adamw | muon | ...": the choice list already says this.
    expect(rows.find((r) => r.key === 'optimizer')?.description).toBeUndefined();
    // Likewise "wsd | warmup_stable | cosine | linear".
    expect(rows.find((r) => r.key === 'lr_schedule')?.description).toBeUndefined();
  });

  it('keeps descriptions that explain the values rather than list them', () => {
    const rows = toRows(pretrain as Surface, 'pretraining');
    // Per-value meaning is information the choice chips cannot carry.
    expect(rows.find((r) => r.key === 'mlm_masking_strategy')?.description).toContain(
      'BERT-style',
    );
    expect(rows.find((r) => r.key === 'muon_adjust_lr')?.description).toContain(
      'Shared matrix-optimizer knobs',
    );
    // The mid-sentence case ("size (allowed: a/b, default: b)") no longer occurs
    // in the real template, so it is exercised by the unit tests below instead.
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
      // Partial list, so the rest of the sentence may depend on it.
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

  describe('toGroups', () => {
    it('folds a section into the template blocks, in template order', () => {
      const pre = pretrain as Surface;
      const groups = toGroups(toRows(pre, 'model'), pre, 'model');
      expect(groups[0].title).toBe('Architecture (edit these first)');
      // Block intros are keyed in template order, independent of the field order
      // that builds the groups, so the two orderings have to agree.
      const noted = Object.keys(
        (pre.properties.model as SectionSchema)['x-nanoplm-group-notes'] ?? {},
      );
      const titles = groups.map((g) => g.title);
      expect(titles.filter((t) => noted.includes(t))).toEqual(
        noted.filter((t) => titles.includes(t)),
      );
      // Every field lands in exactly one block, and none is left ungrouped.
      expect(groups.flatMap((g) => g.rows)).toHaveLength(toRows(pre, 'model').length);
      expect(groups.every((g) => g.title)).toBe(true);
    });

    it('carries the block intro prose as the group note, not the first field', () => {
      const pre = pretrain as Surface;
      const groups = toGroups(toRows(pre, 'pretraining'), pre, 'pretraining');
      const batch = groups.find((g) => g.title === 'Batch (token-budget surface)')!;
      // This sentence describes the whole block. Before it had a home it was
      // glued to micro_batch_seqs, which read as a note about that one key.
      expect(batch.note).toContain('Gradient accumulation is inferred');
      expect(batch.rows.find((r) => r.key === 'micro_batch_seqs')?.description).toBeUndefined();
    });

    it('files the commented-out step overrides under the schedule block', () => {
      // They reach the schema via x-extra-fields, which appends them to the end
      // of the section; the generator regroups so they sit with their twins.
      const pre = pretrain as Surface;
      const groups = toGroups(toRows(pre, 'pretraining'), pre, 'pretraining');
      const duration = groups.find((g) => g.title === 'Duration + LR schedule')!;
      const keys = duration.rows.map((r) => r.key);
      for (const key of ['stable_tokens', 'stable_steps', 'decay_steps', 'max_steps']) {
        expect(keys).toContain(key);
      }
    });

    it('marks a block advanced only when every field in it is', () => {
      const pre = pretrain as Surface;
      const groups = toGroups(toRows(pre, 'pretraining'), pre, 'pretraining');
      expect(groups.find((g) => g.title === 'Paths & run identity')!.advanced).toBe(false);
      expect(groups.find((g) => g.title!.startsWith('torch.compile'))!.advanced).toBe(true);
    });

    it('yields one untitled block for a template with no banners', () => {
      const groups = toGroups(toRows(evalSchema as Surface), evalSchema as Surface);
      expect(groups).toHaveLength(1);
      expect(groups[0].title).toBe('');
      expect(groups[0].slug).toBe('');
    });
  });

  describe('groupSlug', () => {
    it('keeps underscores so a backend name survives', () => {
      expect(
        groupSlug('pretrain-model', 'Recycling / Huginn architecture (pure_torch only)'),
      ).toBe('pretrain-model-group-recycling-huginn-architecture-pure_torch-only');
      expect(groupSlug('pretrain-model', 'Norms & MLP')).toBe('pretrain-model-group-norms-mlp');
    });
  });

  describe('withFieldGroups', () => {
    const toc = (): TocItem[] => [
      { depth: 2, slug: 'the-fields', text: 'The fields', children: [
        { depth: 3, slug: 'model', text: 'model', children: [] },
      ] },
    ];

    it('nests every block under its section, with ids the renderer emits', () => {
      const out = withFieldGroups(toc(), 'reference/pretrain');
      const model = out[0].children[0];
      expect(model.children.length).toBeGreaterThan(10);
      expect(model.children[0]).toEqual({
        depth: 4,
        slug: groupSlug('pretrain-model', 'Architecture (edit these first)'),
        text: 'Architecture (edit these first)',
        children: [],
      });
    });

    it('leaves other pages and the caller’s array alone', () => {
      const input = toc();
      expect(withFieldGroups(input, 'guides/optimizers')).toBe(input);
      withFieldGroups(input, 'reference/pretrain');
      expect(input[0].children[0].children).toEqual([]);
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
