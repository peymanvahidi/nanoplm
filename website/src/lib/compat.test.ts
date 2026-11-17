// website/src/lib/compat.test.ts
import { describe, it, expect } from 'vitest';
import { loadCompat } from './compat';

describe('loadCompat', () => {
  it('marks use_moe as allowed-with-warning', () => {
    const moe = loadCompat().model.find((f) => f.flag === 'use_moe')!;
    expect(moe.status).toBe('warn');
    expect(moe.message).toContain('MoE');
  });

  it('marks use_x0_lambdas as rejected with its reason', () => {
    const lam = loadCompat().model.find((f) => f.flag === 'use_x0_lambdas')!;
    expect(lam.status).toBe('reject');
    expect(lam.reason).toBeTruthy();
  });

  it('has no pure_torch pipeline entry any more', () => {
    expect(loadCompat().pipeline.some((f) => f.flag === 'pure_torch')).toBe(false);
  });

  it('lists explicitly-allowed flags and the trigger', () => {
    const m = loadCompat();
    expect(m.model.some((f) => f.flag === 'use_qk_norm' && f.status === 'allow')).toBe(true);
    expect(m.trigger).toBe('recycling_loop_count > 1');
  });
});
