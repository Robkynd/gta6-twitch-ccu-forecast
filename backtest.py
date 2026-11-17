#!/usr/bin/env python3
"""Backtest harness for gta6-twitch-ccu-forecast-01.

Ground truth: SullyGnome 'Average viewers' hourly, release day per practice
launch (see answers/answers.json). Anchor verified: each case's
history_hourly tail matches the SullyGnome values exactly up to the known
hour-1 print; hours 2-24 are the 23 values after the anchor.

Scoring (from the competition spec): mean pinball loss on the log scale,
  loss = 2 * max(tau*d, (tau-1)*d), d = ln y - ln q,
averaged over every printed hour x 5 levels [0.05,0.25,0.5,0.75,0.95].
"""
import json, math, statistics
from datetime import datetime

KIT = '/home/hatch/workspace/makefun/gta6-kit/gta6-twitch-ccu-forecast-01-kit/input/'
ANS = json.load(open('/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast/answers/answers.json'))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

def load_case(pid):
    return json.load(open(KIT + pid + '.in'))

def pinball(y, q, tau):
    d = math.log(y) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)

def score_case(quantiles, ytrue):
    """quantiles: list of 23 rows, each [q05,q25,q50,q75,q95]; ytrue: 23 values."""
    tot = 0.0; n = 0
    for qs, y in zip(quantiles, ytrue):
        for tau, q in zip(LEVELS, qs):
            tot += pinball(y, q, tau); n += 1
    return tot / n

def baseline_quantiles(p):
    """House starter: flat at hour-1 print, spread 0.15*sqrt(h-1), h=2..24."""
    out = []
    for h in range(2, 25):
        s = 0.15 * math.sqrt(h - 1)
        out.append([p * math.exp(z * s) for z in Z])
    return out

def parse_day_hours(comp):
    """Return dict hour(0..23 of release UTC day) -> viewers, or None if bad."""
    ds = comp['day_start']
    try:
        day = datetime.fromisoformat(ds.replace('Z', '+00:00'))
    except Exception:
        return None
    vals = {}
    for r in comp['hourly']:
        if r.get('viewers') is None:
            continue
        try:
            t = datetime.fromisoformat(r['from'].replace('Z', '+00:00'))
        except Exception:
            continue
        dh = (t - day).total_seconds() / 3600.0
        if 0 <= dh < 24 and dh == int(dh):
            vals[int(dh)] = r['viewers']
    return vals

def tier_of(comp):
    u = str(comp.get('unlock') or '')
    if 'early-access-contaminated' in u or 'early-access contaminated' in u:
        return 'early-access'
    if 'Tier: comparable' in u:
        return 'comparable'
    if 'Tier: weak' in u:
        return 'weak'
    if 'Tier: early' in u:
        return 'early-access'
    return 'unknown'

def comp_diagnostics(comp):
    vals = parse_day_hours(comp)
    if not vals or 0 not in vals:
        return None
    h1 = vals[0]
    peak = max(vals.values())
    # pre-launch day mean (24h before day_start) / h1
    ds = datetime.fromisoformat(comp['day_start'].replace('Z', '+00:00'))
    pre = []
    for r in comp['hourly']:
        if r.get('viewers') is None:
            continue
        t = datetime.fromisoformat(r['from'].replace('Z', '+00:00'))
        dh = (t - ds).total_seconds() / 3600.0
        if -24 <= dh < 0:
            pre.append(r['viewers'])
    pre_mean = sum(pre) / len(pre) if pre else None
    return {
        'name': comp['name'], 'h1': h1, 'peak': peak,
        'peak_over_h1': peak / h1 if h1 else float('inf'),
        'pre_mean_over_h1': (pre_mean / h1) if (pre_mean is not None and h1) else None,
        'tier': tier_of(comp), 'nhours': len(vals),
    }

def log_ratios(comp):
    """log(v[h]/v[0]) for h=1..23 of the release day; None if h0 missing."""
    vals = parse_day_hours(comp)
    if not vals or 0 not in vals or vals[0] <= 0:
        return None
    out = []
    for h in range(1, 24):
        v = vals.get(h)
        out.append(math.log(v / vals[0]) if v and v > 0 else None)
    return out

def median_log_ratios(comps):
    """Per-hour median of log ratios across comparables (None if <3 values)."""
    all_lr = [log_ratios(c) for c in comps]
    all_lr = [x for x in all_lr if x is not None]
    med = []
    for h in range(23):
        xs = sorted(x[h] for x in all_lr if x[h] is not None)
        med.append(statistics.median(xs) if len(xs) >= 3 else None)
    return med, all_lr

def iqr_spread(all_lr):
    """Per-hour robust spread (IQR/1.349) of log ratios."""
    out = []
    for h in range(23):
        xs = sorted(x[h] for x in all_lr if x[h] is not None)
        if len(xs) < 4:
            out.append(None); continue
        n = len(xs)
        q1 = xs[n // 4]; q3 = xs[(3 * n) // 4]
        out.append(max((q3 - q1) / 1.349, 0.05))
    return out

def make_quantiles(p, med_lr, spread, shrink=1.0, spread_scale=1.0):
    out = []
    for i, h in enumerate(range(2, 25)):
        m = med_lr[i] if med_lr[i] is not None else 0.0
        s = spread[i] if spread[i] is not None else 0.15 * math.sqrt(h - 1)
        s = s * spread_scale
        center = p * math.exp(shrink * m)
        out.append([center * math.exp(z * s) for z in Z])
    return out

def run_model(name, fn):
    """fn(case) -> 23x5 quantiles. Returns per-case + mean loss."""
    losses = {}
    for pid in ['p00', 'p01', 'p02', 'p03']:
        case = load_case(pid)
        qs = fn(case)
        y = ANS[pid]['h2_24']
        losses[pid] = score_case(qs, y)
    mean = sum(losses.values()) / 4
    print(f"{name:55s} mean={mean:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in losses.items()))
    return mean

if __name__ == '__main__':
    print("== baseline (house starter) ==")
    run_model('baseline flat p, spread 0.15*sqrt(h-1)',
              lambda c: baseline_quantiles(c['known'][0]['viewers']))
    print()
    print("== comparable diagnostics (p00 set, same 38 in all practice cases) ==")
    case = load_case('p00')
    for c in case['comparables']:
        d = comp_diagnostics(c)
        if d is None:
            print(f"{c['name'][:38]:38s} NO-H1")
        else:
            pm = d['pre_mean_over_h1']
            print(f"{d['name'][:38]:38s} h1={d['h1']:>8.0f} peak/h1={d['peak_over_h1']:6.2f} "
                  f"pre/h1={pm if pm is None else round(pm,3)} tier={d['tier']} nh={d['nhours']}")
