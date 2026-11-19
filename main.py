"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
rate on Exophase at a case's target time, as five quantiles (5, 25, 50, 75,
95%) in percent. Run once per case: it reads every <id>.in in the input
directory and writes <id>.out to the output directory.

How it works
------------
Exophase does not refresh a game's counts continuously: the page sits frozen
for hours, then jumps. A case's history is therefore a staircase of distinct
page states, each with a players-tracked count N and an earners count E_i per
trophy (recovered from the printed rate, which is E_i / N to two decimals).
The forecast is the median of three views of where that staircase is heading,
averaged on the logit scale, with a spread widened by how far the target is
and how noisy the history has been:

1. Cohort split. Between two states, new earners of a trophy come from two
   causes: players newly tracked, and players already tracked who kept
   playing. A small ridge regression splits each trophy's growth between
   them, dE = p * dN + h * N * dt, and both terms are pushed forward: p
   times the new players expected by the target, h times the player-hours
   in between.
2. Saturating earners. Each trophy's earner count is fitted with a
   saturating exponential in time (players earn fastest early, then slow),
   and read at the target.
3. Log-time trend. The logit of each trophy's rate is fitted with a straight
   line against log(1 + hours since release) over the last day of states,
   and read at the target.

The players-tracked base N is extrapolated with a saturating exponential
too (launch-week tracking bends down as the early wave gets scanned),
blended with the last growth rate decaying at the fitted time constant, in
case a fresh surge arrives.

When the history is a single frozen page, none of the three views can see a
refresh coming, so the forecast falls back on what the first refresh did:
the tracked base grew about fivefold and freshly tracked players held each
trophy at roughly 7% of the old cohort's rate, so rates fall to about a
quarter of where they sit, with a wide spread.

Standard library only. No network, no randomness, no wall clock: the same
case file always yields the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime

IN_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Normal quantiles at the five scored levels, in the order rows list them.
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)


def clamp_rate(x):
    """A percent into the scorer's logit range."""
    return min(99.99, max(0.01, x))


def to_logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(value):
    if value > 30.0:
        return 100.0
    if value < -30.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-value))


def round4(x):
    """Round to four decimals, halves away from zero, like the graders do."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def read_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def page_states(case):
    """Collapse the history into distinct page states.

    Consecutive reads with the same tracked count and the same rates are one
    state, kept at its first slot. Each state carries hours since the game's
    release day began, the tracked count, and the rates.
    """
    release = case.get("game", {}).get("release_date", "2026-11-19")
    day0 = datetime.fromisoformat(release + "T00:00:00+00:00")
    states = []
    for read in case.get("history", []):
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if states and states[-1]["key"] == key:
            continue
        slot = read_time(read["slot"])
        states.append({
            "key": key,
            "hours": (slot - day0).total_seconds() / 3600.0,
            "players": read.get("players_tracked") or 0,
            "rates": list(read.get("rates") or []),
        })
    return states


def head_count(rate, players):
    """Earners behind a printed rate, or None where the rate is missing."""
    if rate is None or players <= 0:
        return None
    return int(round(rate / 100.0 * players))


def ridge_split(rows, ys, prior, ridge=0.05):
    """Least squares y ~ p*x1 + h*x2, pulled toward `prior` by a ridge.

    rows is a list of (x1, x2); ys the targets. Returns (p, h).
    """
    s11 = sum(x1 * x1 for x1, _ in rows)
    s12 = sum(x1 * x2 for x1, x2 in rows)
    s22 = sum(x2 * x2 for _, x2 in rows)
    t1 = sum(x1 * y for (x1, _), y in zip(rows, ys))
    t2 = sum(x2 * y for (_, x2), y in zip(rows, ys))
    lam1 = ridge * s11 + 1e-9
    lam2 = ridge * s22 + 1e-9
    a11, a12, a22 = s11 + lam1, s12, s22 + lam2
    b1, b2 = t1 + lam1 * prior[0], t2 + lam2 * prior[1]
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12:
        return prior
    p = (b1 * a22 - b2 * a12) / det
    h = (a11 * b2 - a12 * b1) / det
    return p, h


def project_players(states, target_hours):
    """Players tracked at the target, and the curve getting there.

    A saturating exponential fitted by a grid over its time constant (the
    launch wave gets scanned and growth bends down), blended in log space
    70/30 with the last growth rate decaying at the fitted constant, as a
    hedge against a fresh surge. Returns (n_target, n_at).
    """
    n_last = states[-1]["players"]
    h_last = states[-1]["hours"]
    if len(states) < 2 or target_hours <= h_last:
        return float(n_last), (lambda h: float(n_last))
    h0 = states[0]["hours"]
    n0 = states[0]["players"]
    best = None
    for tau in (6, 12, 18, 24, 36, 48, 72, 96, 144, 216, 336, 504, 768):
        us = [1.0 - math.exp(-(s["hours"] - h0) / tau) for s in states]
        ys = [s["players"] - n0 for s in states]
        suu = sum(u * u for u in us)
        suy = sum(u * y for u, y in zip(us, ys))
        slope = suy / suu if suu > 1e-12 else 0.0
        err = sum((y - slope * u) ** 2 for u, y in zip(us, ys))
        if best is None or err < best[0]:
            best = (err, tau, slope)
    _, tau, slope = best
    tau = min(max(tau, 12.0), 96.0)

    def n_sat(h):
        return n0 + slope * (1.0 - math.exp(-(h - h0) / tau))

    growth = 0.0
    span = states[-1]["hours"] - states[-2]["hours"]
    if span > 0 and states[-2]["players"] > 0:
        growth = (math.log(max(1, states[-1]["players"]))
                  - math.log(max(1, states[-2]["players"]))) / span
        growth = min(max(growth, 0.0), 0.5)

    def n_decay(h):
        if growth <= 0:
            return float(n_last)
        return n_last * math.exp(growth * tau
                                 * (1.0 - math.exp(-(h - h_last) / tau)))

    def blend(h):
        a = max(n_sat(h), 1.0)
        b = max(n_decay(h), 1.0)
        return math.exp(0.7 * math.log(a) + 0.3 * math.log(b))

    n_target = min(max(blend(target_hours), float(n_last)),
                   8.0 * float(n_last))

    def n_at(h):
        if h <= h_last:
            return float(n_last)
        return min(max(blend(h), float(n_last)), n_target)

    return n_target, n_at


def saturating_earners(states, idx, target_hours):
    """Earners of one trophy at the target, saturating exponential in time.

    Returns None when the trophy's rate is missing from every state.
    """
    h0 = states[0]["hours"]
    first = head_count(states[0]["rates"][idx]
                       if idx < len(states[0]["rates"]) else None,
                       states[0]["players"])
    if first is None:
        return None
    best = None
    for tau in (12, 24, 48, 96, 192, 384, 768, 1536, 3072):
        us = [1.0 - math.exp(-(s["hours"] - h0) / tau) for s in states]
        ys = []
        usable = True
        for s in states:
            e = head_count(s["rates"][idx]
                           if idx < len(s["rates"]) else None,
                           s["players"])
            if e is None:
                usable = False
                break
            ys.append(e - first)
        if not usable:
            return None
        suu = sum(u * u for u in us)
        suy = sum(u * y for u, y in zip(us, ys))
        slope = suy / suu if suu > 1e-12 else 0.0
        err = sum((y - slope * u) ** 2 for u, y in zip(us, ys))
        if best is None or err < best[0]:
            best = (err, tau, slope)
    _, tau, slope = best
    reach = 1.0 - math.exp(-(target_hours - h0) / tau)
    return max(first + slope * reach, 0.0)


def trend_rate(states, idx, target_hours):
    """A trophy's rate at the target from a log-time trend, or None.

    Fits the logit of the rate against log(1 + hours since release) over the
    states of the last day (all states when fewer than three fall there).
    Needs at least three points, so it abstains on a short history.
    """
    h_last = states[-1]["hours"]
    window = [(s["hours"], s["rates"][idx])
              for s in states
              if s["hours"] >= h_last - 24.0
              and idx < len(s["rates"])
              and s["rates"][idx] is not None
              and 0.0 < s["rates"][idx] < 100.0]
    if len(window) < 3:
        window = [(s["hours"], s["rates"][idx])
                  for s in states
                  if idx < len(s["rates"])
                  and s["rates"][idx] is not None
                  and 0.0 < s["rates"][idx] < 100.0]
    if len(window) < 3:
        return None
    pts = [(math.log1p(h), to_logit(r)) for h, r in window]
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    intercept = my - slope * mx
    return clamp_rate(from_logit(intercept + slope * math.log1p(target_hours)))


def forecast_case(case):
    """One case's forecast rows: [{"id": n, "q": [q05..q95]}, ...]."""
    trophies = case["trophies"]
    count = len(trophies)
    horizon_min = case["horizon_minutes"]
    horizon_days = horizon_min / 1440.0
    release = case.get("game", {}).get("release_date", "2026-11-19")
    day0 = datetime.fromisoformat(release + "T00:00:00+00:00")
    target_hours = (read_time(case["target"]) - day0).total_seconds() / 3600.0
    states = page_states(case)

    if not states:
        return [{"id": t["id"], "q": [0.05, 0.2, 0.5, 1.5, 5.0]}
                for t in trophies]

    last = states[-1]
    n_last = last["players"]
    rates_last = []
    for i in range(count):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        if rate is None:
            for s in reversed(states[:-1]):
                if i < len(s["rates"]) and s["rates"][i] is not None:
                    rate = s["rates"][i]
                    break
        rates_last.append(0.0 if rate is None else float(rate))
    earn_last = [head_count(r, n_last) or 0 for r in rates_last]

    # Intervals between states, for the cohort split.
    gaps = []
    for a, b in zip(states, states[1:]):
        dt = b["hours"] - a["hours"]
        dn = b["players"] - a["players"]
        de = []
        for i in range(count):
            ea = head_count(a["rates"][i] if i < len(a["rates"]) else None,
                            a["players"])
            eb = head_count(b["rates"][i] if i < len(b["rates"]) else None,
                            b["players"])
            de.append(None if (ea is None or eb is None) else eb - ea)
        gaps.append({"dt": dt, "dn": dn, "n_prev": a["players"], "de": de})

    # Players tracked at the target, and player-hours over the horizon.
    if len(states) == 1:
        if horizon_min >= 60:
            growth = 5.0
        elif horizon_min >= 15:
            growth = 2.0
        else:
            growth = 1.0  # No time for a refresh before the target.
        n_target = float(n_last) * growth
        horizon_h = target_hours - last["hours"]
        player_hours = (float(n_last) + n_target) / 2.0 * max(horizon_h, 0.0)
    else:
        n_target, n_at = project_players(states, target_hours)
        horizon_h = target_hours - last["hours"]
        steps = 24
        player_hours = 0.0
        prev_h, prev_n = last["hours"], float(n_last)
        for k in range(1, steps + 1):
            h = last["hours"] + horizon_h * k / steps
            nv = n_at(h)
            player_hours += (prev_n + nv) / 2.0 * (h - prev_h)
            prev_h, prev_n = h, nv

    medians = []
    for i in range(count):
        rate = rates_last[i]
        if rate >= 100.0:
            medians.append(100.0)
            continue
        if rate <= 0.0:
            medians.append(0.01)
            continue
        views = []
        # View 1: cohort split.
        if len(gaps) >= 2:
            rows, ys = [], []
            for g in gaps:
                if g["de"][i] is None:
                    continue
                rows.append((float(g["dn"]),
                             float(g["n_prev"]) * max(g["dt"], 0.0)))
                ys.append(float(g["de"][i]))
            tot_dn = sum(x1 for x1, _ in rows)
            tot_de = sum(ys)
            p0 = min(max(tot_de / tot_dn, 0.0), 1.0) if tot_dn > 0 \
                else 0.07 * rate / 100.0
            p, h = ridge_split(rows, ys, (p0, 0.0))
            p = min(max(p, 0.0), 1.0)
            h = min(max(h, 0.0), 0.2)
        elif len(gaps) == 1:
            g = gaps[0]
            if g["de"][i] is None or g["dn"] <= 0:
                p, h = 0.07 * rate / 100.0, 0.0
            else:
                p = min(max(g["de"][i] / g["dn"], 0.0), 1.0)
                h = 0.0
        else:
            # No refresh seen: fresh players hold the trophy at ~7% of the
            # old cohort's rate, and tracked players add nothing yet.
            p, h = 0.07 * rate / 100.0, 0.0
        e_tgt = earn_last[i] + p * max(n_target - n_last, 0.0) \
            + h * player_hours
        e_tgt = min(max(e_tgt, 0.0), n_target)
        view1 = 100.0 * e_tgt / n_target if n_target > 0 else rate
        views.append(min(max(view1, 0.10 * rate),
                         max(3.0 * rate, rate + 10.0)))
        # View 2: saturating earners.
        e_sat = saturating_earners(states, i, target_hours)
        if e_sat is not None and n_target > 0:
            view2 = 100.0 * e_sat / n_target
            views.append(min(max(view2, 0.10 * rate),
                             max(3.0 * rate, rate + 10.0)))
        # View 3: log-time trend (abstains on a short history).
        view3 = trend_rate(states, i, target_hours)
        if view3 is not None:
            views.append(min(max(view3, 0.10 * rate),
                             max(3.0 * rate, rate + 10.0)))
        # Average the views on the logit scale.
        logits = [to_logit(v) for v in views]
        medians.append(clamp_rate(from_logit(sum(logits) / len(logits))))

    # Spread: residual scatter around the log-time trend, pooled across
    # trophies, widened with how far the target sits past the history.
    if len(states) == 1:
        if horizon_min >= 60:
            flat_spread = 0.90
        elif horizon_min >= 15:
            flat_spread = 0.60
        else:
            flat_spread = 0.25
        spreads = [flat_spread] * count
    else:
        span_h = max(states[-1]["hours"] - states[0]["hours"], 6.0)
        resid = []
        for i in range(count):
            pts = []
            for s in states:
                if i < len(s["rates"]) and s["rates"][i] is not None \
                        and 0.0 < s["rates"][i] < 100.0:
                    pts.append((math.log1p(s["hours"]),
                                to_logit(s["rates"][i])))
            if len(pts) >= 3:
                mx = sum(x for x, _ in pts) / len(pts)
                my = sum(y for _, y in pts) / len(pts)
                sxx = sum((x - mx) ** 2 for x, _ in pts)
                sxy = sum((x - mx) * (y - my) for x, y in pts)
                slope = sxy / sxx if sxx > 1e-9 else 0.0
                icept = my - slope * mx
                resid.extend(y - (icept + slope * x) for x, y in pts)
        if len(resid) >= 5:
            resid.sort()
            sigma = 1.4826 * abs(resid[len(resid) // 2])
            sigma = min(max(sigma, 0.15), 0.80)
        else:
            sigma = 0.30
        if len(states) == 2:
            base = 0.75
        else:
            base = math.sqrt(sigma * sigma
                             * (1.0 + horizon_days / max(span_h / 24.0, 0.25))
                             + 0.10 * 0.10)
        spreads = []
        for i in range(count):
            s = base
            if rates_last[i] < 0.2:
                s += 0.15
            if earn_last[i] < 20:
                s += 0.10
            spreads.append(min(max(s, 0.40), 1.50))

    rows = []
    for t, med, s in zip(trophies, medians, spreads):
        mid = to_logit(med)
        qs = [round4(from_logit(mid + z * s)) for z in Z_SCORES]
        qs = sorted(min(max(q, 0.01), 100.0) for q in qs)
        rows.append({"id": t["id"], "q": qs})
    return rows


def fallback_rows(case):
    """A valid forecast when the main path fails: last rate, wide spread."""
    rows = []
    history = case.get("history", [])
    last_rates = history[-1].get("rates", []) if history else []
    for i, t in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) \
            and last_rates[i] is not None else 0.5
        mid = to_logit(rate)
        qs = [round4(from_logit(mid + z * 1.0)) for z in Z_SCORES]
        qs = sorted(min(max(q, 0.01), 100.0) for q in qs)
        rows.append({"id": t["id"], "q": qs})
    return rows


def main():
    in_dir, out_dir = IN_DIR, OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        try:
            rows = forecast_case(case)
        except Exception:
            rows = fallback_rows(case)
        doc = {"trophies": rows}
        text = json.dumps(doc, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(name + " forecast over the size cap")
        with open(os.path.join(out_dir, name[:-3] + ".out"),
                  "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
