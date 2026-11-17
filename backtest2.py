#!/usr/bin/env python3
"""Model variants for gta6-twitch-ccu-forecast-01 backtest."""
import math, statistics
from backtest import (load_case, run_model, make_quantiles, log_ratios,
                      median_log_ratios, iqr_spread, comp_diagnostics, Z)

def usable_comps(case):
    """Comparables with a valid hour-1 (excludes the GTA VI category row)."""
    out = []
    for c in case['comparables']:
        lr = log_ratios(c)
        if lr is not None:
            out.append(c)
    return out

def wmedian(pairs):
    """Weighted median of (value, weight)."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]

def wiqr(pairs):
    """Weighted IQR/1.349."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    def wq(q):
        acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= tot * q:
                return v
        return pairs[-1][0]
    return max((wq(0.75) - wq(0.25)) / 1.349, 0.05)

def build(case, comps, shrink=1.0, spread_scale=1.0, kernel_sigma=None,
          smooth=0):
    p = case['known'][0]['viewers']
    all_lr = [log_ratios(c) for c in comps]
    all_lr = [x for x in all_lr if x is not None]
    # weights per comparable (kernel on log h1 vs log p)
    ws = []
    for c in comps:
        lr = log_ratios(c)
        if lr is None:
            continue
        if kernel_sigma is None:
            ws.append(1.0)
        else:
            import math as _m
            d = comp_diagnostics(c)
            ws.append(_m.exp(-((_m.log(d['h1']) - _m.log(p)) ** 2) / (2 * kernel_sigma ** 2)))
    med, spread = [], []
    for h in range(23):
        pairs = [(x[h], w) for x, w in zip(all_lr, ws) if x[h] is not None]
        med.append(wmedian(pairs) if len(pairs) >= 3 else 0.0)
        spread.append(wiqr(pairs) if len(pairs) >= 4 else 0.15 * math.sqrt(h + 1))
    if smooth:
        sm = []
        for i in range(23):
            lo, hi = max(0, i - smooth), min(23, i + smooth + 1)
            sm.append(sum(med[lo:hi]) / (hi - lo))
        med = sm
    return make_quantiles(p, med, spread, shrink=shrink, spread_scale=spread_scale)

def filt_peak(case, max_ratio=20.0):
    return [c for c in usable_comps(case)
            if (comp_diagnostics(c) or {}).get('peak_over_h1', 1e9) <= max_ratio]

def filt_peak_noea(case, max_ratio=20.0):
    return [c for c in filt_peak(case, max_ratio)
            if (comp_diagnostics(c) or {}).get('tier') != 'early-access']

def filt_tier_comp(case):
    return [c for c in usable_comps(case)
            if (comp_diagnostics(c) or {}).get('tier') == 'comparable']

if __name__ == '__main__':
    print("== variants (mean pinball loss, lower better; baseline 0.3871) ==")
    run_model('V1 median-ratio ALL (37 comps)',
              lambda c: build(c, usable_comps(c)))
    run_model('V2 median-ratio tier==comparable only',
              lambda c: build(c, filt_tier_comp(c)))
    run_model('V3 median-ratio peak/h1<=20',
              lambda c: build(c, filt_peak(c)))
    run_model('V4 V3 + drop early-access tier',
              lambda c: build(c, filt_peak_noea(c)))
    run_model('V4s V4 + smooth3 (3h moving avg of log ratios)',
              lambda c: build(c, filt_peak_noea(c), smooth=1))
    run_model('V5 V4 + kernel sigma=1.0 on log h1',
              lambda c: build(c, filt_peak_noea(c), kernel_sigma=1.0))
    run_model('V5b V4 + kernel sigma=0.7 on log h1',
              lambda c: build(c, filt_peak_noea(c), kernel_sigma=0.7))
    run_model('V6 V4 + shrink 0.85',
              lambda c: build(c, filt_peak_noea(c), shrink=0.85))
    run_model('V6b V4 + shrink 0.7',
              lambda c: build(c, filt_peak_noea(c), shrink=0.7))
    run_model('V7 V4 + spread_scale 0.8',
              lambda c: build(c, filt_peak_noea(c), spread_scale=0.8))
    run_model('V7b V4 + spread_scale 1.25',
              lambda c: build(c, filt_peak_noea(c), spread_scale=1.25))
