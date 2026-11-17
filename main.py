"""gta6-twitch-ccu-forecast-01 entry: tier-weighted comparable-kernel ensemble.

Method (backtested on the 4 practice launches; ground truth reconstructed from
SullyGnome's hourly "Average viewers" for each launch's release day, anchor
verified against each case's history_hourly tail):

  1. Anchor on hour 1's print p1.
  2. For every comparable launch, take its release UTC day (day_start + 0..23h)
     and form log ratios r_h = ln(viewers_h / viewers_hour1). A launch is
     usable only if its hour-1 point exists and at least 20 of the 24 release
     hours are present.
  3. Filter out launches that cannot anchor on midnight the way the target
     does:
       - release-day peak / hour-1 > 20  (single global unlock later in the
         UTC day: Silksong, Black Myth, Battlefield 6, ... -- their hour 1 is
         pre-unlock, so their ratios explode to 100-2000x);
       - Tier: early-access-contaminated in the case's own unlock note
         (Hogwarts Legacy, Diablo IV, Starfield, ... -- already playing for
         days, so their ratios are compressed toward 1);
       - Tier: target-history / the target game's own category row.
  4. Center: per-hour weighted median of the log ratios, Gaussian kernel
     sigma=0.8 on |ln(hour1_comp) - ln(p1)| -- launches that opened at a
     similar scale behave alike. q50_h = p1 * exp(median_h).
  5. Spread: per-hour weighted IQR/1.349 of the same log ratios (floor 0.05,
     cap 1.0); q_tau = q50 * exp(z_tau * s_h) with z the standard-normal
     quantiles at the five scored levels.

Backtest (mean pinball loss on the log scale, lower is better):
  house baseline (flat p1, spread 0.15*sqrt(h-1)):  0.3871
  this model:                                        0.2001-0.2004
    p00 Wolverine 0.315, p01 RDR2 0.142, p02 GoW 0.148, p03 Spider-Man 2 0.197
  90% interval coverage 0.78-0.96 (mean 0.89, nominal 0.90).

Rejected after backtesting (kept out deliberately): pre-launch-shape
similarity kernel (0.23-0.34, worse), LOO bias correction (mixed per case,
chases the Wolverine outlier), shrinkage toward persistence (worse),
hour smoothing (worse), hard top-K nearest (unstable across K).

Fallbacks: if fewer than 3 comparables survive the filter, relax the
early-access exclusion, then the peak-ratio filter, then fall back to the
house baseline. Any error in the model path falls back to the baseline for
that case, so the program always emits a valid output.

Standard library only, deterministic (no randomness, no network, no clock),
runs in well under the 8 s / 1 CPU / 1024 MB limits.
"""

import datetime as _dt
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
BASE_SIGMA = 0.15

# --- model constants (chosen by backtest, see module docstring) ---
KERNEL_SIGMA = 0.8       # Gaussian kernel on |ln h1_comp - ln p1|
MAX_PEAK_OVER_H1 = 20.0  # drop single-global-unlock launches
MIN_DAY_HOURS = 20       # release-day hours that must be present
SPREAD_FLOOR = 0.05
SPREAD_CAP = 1.0
TIER_RE = re.compile(r"Tier:\s*([A-Za-z][A-Za-z0-9-]*)")


def round2(x):
    """x rounded to 2 decimals, half up (as the house starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def clamp_q(x):
    return min(max(x, 0.01), 1_000_000_000.0)


def baseline_quantiles(p1, h):
    s = BASE_SIGMA * math.sqrt(h - 1)
    return [round2(clamp_q(p1 * math.exp(z * s))) for z in Z]


def parse_day_hours(comp):
    """Release-day (UTC) hourly viewers {0..23: viewers}, or None."""
    try:
        day = comp["day_start"]
        y, m, d = int(day[0:4]), int(day[5:7]), int(day[8:10])
    except Exception:
        return None
    vals = {}
    for r in comp.get("hourly") or []:
        v = r.get("viewers")
        if v is None or v <= 0:
            continue
        f = r.get("from") or ""
        try:
            ry, rm, rd, rh = int(f[0:4]), int(f[5:7]), int(f[8:10]), int(f[11:13])
        except Exception:
            continue
        # Day offset in whole hours from day_start (dates are UTC, no DST).
        try:
            t = _dt.datetime(ry, rm, rd, rh, tzinfo=_dt.timezone.utc)
            d0 = _dt.datetime(y, m, d, tzinfo=_dt.timezone.utc)
        except Exception:
            continue
        dh = int((t - d0).total_seconds() // 3600)
        if 0 <= dh < 24:
            vals[dh] = v
    return vals if vals else None


def tier_of(comp):
    m = TIER_RE.search(str(comp.get("unlock") or ""))
    return m.group(1).lower() if m else "comparable"


def usable_comparables(case, drop_early_access, drop_peak_filter):
    """(log-ratio curves, kernel weights) after the filters."""
    target_name = str((case.get("game") or {}).get("name") or "").strip().lower()
    p1 = case["_p1"]
    curves = []
    for comp in case.get("comparables") or []:
        vals = parse_day_hours(comp)
        if not vals or 0 not in vals or len(vals) < MIN_DAY_HOURS:
            continue
        name = str(comp.get("name") or "").strip().lower()
        if target_name and (name == target_name or target_name in name):
            continue
        tier = tier_of(comp)
        if tier == "target-history":
            continue
        if drop_early_access and tier.startswith("early-access"):
            continue
        h1 = vals[0]
        peak = max(vals.values())
        if not drop_peak_filter and peak / h1 > MAX_PEAK_OVER_H1:
            continue
        lr = []
        for h in range(1, 24):
            v = vals.get(h)
            lr.append(math.log(v / h1) if v else None)
        w = math.exp(-((math.log(h1) - math.log(p1)) ** 2) / (2 * KERNEL_SIGMA ** 2))
        curves.append((lr, w))
    return curves


def weighted_median(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def weighted_iqr(pairs):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)

    def wq(q):
        acc = 0.0
        for v, w in pairs:
            acc += w
            if acc >= total * q:
                return v
        return pairs[-1][0]

    return max((wq(0.75) - wq(0.25)) / 1.349, SPREAD_FLOOR)


def model_quantiles(case):
    """23 rows x 5 quantiles from the comparable-kernel ensemble."""
    p1 = case["_p1"]
    for drop_ea, drop_pk in ((True, False), (False, False), (False, True)):
        curves = usable_comparables(case, drop_ea, drop_pk)
        if len(curves) >= 3:
            break
    else:
        return None
    rows = []
    prev_med, prev_spr = 0.0, BASE_SIGMA
    for i, h in enumerate(range(2, 25)):
        pairs = [(lr[i], w) for lr, w in curves if lr[i] is not None]
        if len(pairs) >= 3:
            med = weighted_median(pairs)
            prev_med = med
        else:
            med = prev_med
        if len(pairs) >= 4:
            spr = min(weighted_iqr(pairs), SPREAD_CAP)
            prev_spr = spr
        else:
            spr = prev_spr
        center = p1 * math.exp(med)
        qs = [round2(clamp_q(center * math.exp(z * spr))) for z in Z]
        # Rounding can only preserve order, but enforce it explicitly.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append(qs)
    return rows


def forecast_case(case, name):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None or not first.get("viewers"):
        hist = case.get("history_hourly") or []
        p1 = max(1, hist[-1]["viewers"]) if hist else 1
    else:
        p1 = max(1, first["viewers"])
    case["_p1"] = p1
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))
    try:
        rows = model_quantiles(case)
    except Exception:
        rows = None
    if rows is None or len(rows) != len(hours):
        rows = [baseline_quantiles(p1, h) for h in hours]
    return {"hours": [{"hour": h, "q": q} for h, q in zip(hours, rows)]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
