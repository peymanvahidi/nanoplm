// website/src/lib/compat.test.ts
import { describe, it, expect } from 'vitest';
import { loadCompat } from './compat';

describe('loadCompat', () => {
  it('marks use_moe as rejected with its reason', () => {
    const moe = loadCompat().model.find((f) => f.flag === 'use_moe')!;
    expect(moe.status).toBe('reject');
    expect(moe.reason).toContain('MoE');
  });

  it('keeps the inverted-polarity condition on pure_torch', () => {
    const pt = loadCompat().pipeline.find((f) => f.flag === 'pure_torch')!;
    expect(pt.status).toBe('reject');
    expect(pt.condition).toBe('pure_torch is False');
  });

  it('lists explicitly-allowed flags and the trigger', () => {
    const m = loadCompat();
    expect(m.model.some((f) => f.flag === 'use_qk_norm' && f.status === 'allow')).toBe(true);
    expect(m.trigger).toBe('recycling_loop_count > 1');
  });
});
