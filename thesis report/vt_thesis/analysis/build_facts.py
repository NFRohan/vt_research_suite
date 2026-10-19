"""Single source of truth for every number quoted in the thesis.

Reads the raw result CSVs under ../../../Findings and writes:
  facts.json  - machine-readable, consumed by make_figures.py
  facts.md    - human-readable provenance table (every number -> its CSV)

Selection logic is copied from Findings/reselect_operating_points.py so the
thesis reproduces the defense numbers exactly.
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINDINGS = HERE.parents[2] / "Findings"
OUT_JSON = HERE.parent / "facts.json"
OUT_MD = HERE.parent / "facts.md"

# name -> (folder, backbone, regime, schedule)
CONFIGS = {
    "r18_clean_T8":   ("ResNet Clean Trained",            "ResNet-18",   "Clean", "T8"),
    "r18_adv_T8":     ("ResNet Adversarial Trained",      "ResNet-18",   "Adv",   "T8"),
    "r18_adv_T8_rep": ("ResNet Adv T8 Rerun",             "ResNet-18",   "Adv",   "T8"),
    "r18_clean_mixT": ("Resnet Clean MixT",               "ResNet-18",   "Clean", "mixT"),
    "r18_adv_mixT":   ("Resnet ReRun",                    "ResNet-18",   "Adv",   "mixT"),
    "vgg_clean_T8":   ("VGG Clean Trained",               "VGG16",       "Clean", "T8"),
    "vgg_adv_T8":     ("VGG Adversarial Trained",         "VGG16",       "Adv",   "T8"),
    "mbv2_clean_T8":  ("MobileNetV2 Clean Trained",       "MobileNetV2", "Clean", "T8"),
    "mbv2_adv_T8":    ("MobileNetV2 Adversarial Trained", "MobileNetV2", "Adv",   "T8"),
}


def _n_from_hist(h) -> int:
    return sum(int(x) for x in re.findall(r":\s*(\d+)", str(h)))


def _fixed(df):
    return df[df["mode"] == "fixed"] if "mode" in df.columns else df


def _full(fx):
    t = int(fx["time_steps"].max())
    return t, float(fx.loc[fx["time_steps"] == t, "accuracy"].iloc[0])


def _match(vt_syn, fx):
    f = fx.dropna(subset=["avg_synops"]).copy()
    f["gap"] = (f["avg_synops"] - vt_syn).abs()
    r = f.sort_values("gap").iloc[0]
    return int(r["time_steps"]), float(r["accuracy"]), float(r["avg_synops"])


def neutral_select(cv, cf, delta):
    _, full = _full(cf)
    cand = cv[cv["accuracy"] >= full - delta]
    fallback = cand.empty
    if fallback:
        cand = cv.sort_values("accuracy", ascending=False).head(1)
    return float(cand.sort_values("avg_synops").iloc[0]["threshold"]), fallback


def biased_select(cv, cf):
    best, best_d = None, -1e9
    for _, r in cv.dropna(subset=["avg_synops"]).iterrows():
        _, facc, _ = _match(float(r["avg_synops"]), cf)
        d = float(r["accuracy"]) - facc
        if d > best_d:
            best, best_d = float(r["threshold"]), d
    return best


def op_point(tv, tf, theta):
    row = tv.loc[(tv["threshold"] - theta).abs().idxmin()]
    vt_acc, vt_syn = float(row["accuracy"]), float(row["avg_synops"])
    mt, macc, msyn = _match(vt_syn, tf)
    tfull, facc = _full(tf)
    full_syn = float(tf.loc[tf["time_steps"] == tfull, "avg_synops"].iloc[0])
    return dict(theta=theta, vt_acc=vt_acc, vt_synops=vt_syn,
                depth=float(row["avg_stopping_depth"]), matched_T=mt,
                matched_acc=macc, matched_synops=msyn, gain=vt_acc - macc,
                full_acc=facc, full_synops=full_syn, below_full=facc - vt_acc,
                synops_saving=1.0 - vt_syn / full_syn)


def binom_ci95(acc_pct, n):
    p = acc_pct / 100.0
    return 1.96 * np.sqrt(p * (1 - p) / n) * 100.0


def config_facts(key):
    folder, bb, reg, sched = CONFIGS[key]
    root = FINDINGS / folder
    cv = pd.read_csv(root / "main_vt_threshold_sweep.csv")
    cf = _fixed(pd.read_csv(root / "main_fixed_t_sweep.csv"))
    tv = pd.read_csv(root / "test_vt_threshold_sweep.csv")
    tf = _fixed(pd.read_csv(root / "test_fixed_t_sweep.csv"))
    tf = tf.dropna(subset=["avg_synops"]).sort_values("time_steps")

    out = dict(key=key, folder=folder, backbone=bb, regime=reg, schedule=sched)
    out["fixed_sweep"] = [dict(T=int(r.time_steps), acc=float(r.accuracy),
                               synops=float(r.avg_synops))
                          for r in tf.itertuples()]
    out["vt_sweep"] = [dict(theta=float(r.threshold), acc=float(r.accuracy),
                            depth=float(r.avg_stopping_depth),
                            synops=float(r.avg_synops))
                       for r in tv.sort_values("threshold").itertuples()]
    out["dense_ops_T8"] = float(tf.loc[tf["time_steps"] == tf["time_steps"].max(), "avg_dense_ops"].iloc[0]) \
        if "avg_dense_ops" in tf.columns else None

    sel = {}
    for d in (0.5, 1.0, 2.0):
        th, fb = neutral_select(cv, cf, d)
        sel[f"neutral_{d}"] = {**op_point(tv, tf, th), "fallback": fb}
    sel["biased"] = {**op_point(tv, tf, biased_select(cv, cf)), "fallback": False}
    out["selection"] = sel
    out["n_test"] = 10000
    out["ci95_at_neutral"] = binom_ci95(sel["neutral_1.0"]["vt_acc"], 10000)

    # VT-above-fixed check across the whole sweep
    above = sum(1 for p in out["vt_sweep"]
                if p["acc"] > np.interp(p["synops"], tf["avg_synops"], tf["accuracy"]))
    out["vt_points_above_fixed"] = [above, len(out["vt_sweep"])]

    # attacks
    a = pd.read_csv(root / "attack_results.csv")
    att = []
    for r in a.itertuples():
        att.append(dict(mode=r.mode, role=getattr(r, "threshold_role", None),
                        theta=None if pd.isna(r.threshold) else float(r.threshold),
                        attack=r.attack_name, objective=r.attack_objective,
                        acc=float(r.accuracy), depth=float(r.avg_stopping_depth),
                        n=_n_from_hist(r.step_histogram)))
    out["attacks"] = att

    # stopping rules
    sr = pd.read_csv(root / "stopping_rule_test_results.csv")
    out["stopping_rules"] = [dict(rule=r.stop_rule,
                                  theta=None if pd.isna(r.threshold) else float(r.threshold),
                                  acc=float(r.accuracy), depth=float(r.avg_stopping_depth),
                                  synops=float(r.avg_synops))
                             for r in sr.itertuples()]
    return out


def compute_overuse(cfg):
    """Depth inflation at the originally selected threshold."""
    co = [x for x in cfg["attacks"] if x["objective"] == "compute_overuse"]
    if not co:
        return None
    sel = [x for x in co if x["role"] == "selected"] or co
    x = sel[0]
    th = x["theta"]
    clean_depth = min(cfg["vt_sweep"], key=lambda p: abs(p["theta"] - th))["depth"]
    return dict(theta=th, clean_depth=clean_depth, attacked_depth=x["depth"],
                inflation=x["depth"] - clean_depth, acc_kept=x["acc"], n=x["n"])


def single_run_robustness(cfg):
    u = [x for x in cfg["attacks"] if x["objective"] == "untargeted"]
    def pick(mode, atk):
        q = [x for x in u if x["mode"] == mode and x["attack"] == atk]
        if mode == "vt":
            s = [x for x in q if x["role"] == "selected"]
            q = s or q
        return q[0] if q else None
    res = {}
    for atk in ("pgd", "autoattack"):
        f, v = pick("fixed", atk), pick("vt", atk)
        if f and v:
            res[atk] = dict(fixed_T8=f["acc"], vt=v["acc"], delta=v["acc"] - f["acc"],
                            theta=v["theta"], n=f["n"])
    return res


def multiseed(folder):
    p = FINDINGS / folder / "multiseed_train_seed_raw_adv.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    out = {"thetas": sorted(d["selected_threshold"].unique().tolist()),
           "matched_T": sorted(int(x) for x in d["matched_fixed_t"].unique())}
    for cond in ("clean", "pgd", "autoattack"):
        q = d[d["attack"] == cond]
        deltas, per = [], {}
        for ck in sorted(q["checkpoint"].unique()):
            r = q[q["checkpoint"] == ck]
            vt = float(r[r["operating_point"] == "vt_selected"]["accuracy"].iloc[0])
            mf = float(r[r["operating_point"] == "fixed_matched"]["accuracy"].iloc[0])
            f8 = float(r[r["operating_point"] == "fixed_T8"]["accuracy"].iloc[0])
            deltas.append(vt - mf)
            per[ck] = dict(vt=vt, matched=mf, fixed_T8=f8)
        m, s = st.mean(deltas), st.stdev(deltas)
        out[cond] = dict(per_seed=deltas, mean=m, std=s, per_checkpoint=per,
                         verdict="real" if abs(m) > 2 * s else ("noise" if abs(m) < s else "marginal"))
    return out


def attack_seed_summary(folder):
    p = FINDINGS / folder / "multiseed_attack_seed_summary_adv.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return {f"{r.operating_point}/{r.attack}": dict(mean=float(r.mean), std=float(r.std))
            for r in d.itertuples()}


def train_histories(folder):
    out = []
    for h in sorted((FINDINGS / folder).glob("train_history_adv_seed*.csv")):
        d = pd.read_csv(h)
        vc = d["val_clean_acc"].dropna()
        out.append(dict(file=h.name, epochs=len(d),
                        final_train_adv=float(d["train_adv_acc"].iloc[-1]),
                        best_val_clean=float(vc.max()) if len(vc) else None))
    return out


def old_subset_matched():
    p = FINDINGS / "ResNet Adversarial Trained" / "subset_consistent_selected_attack_comparison.csv"
    d = pd.read_csv(p)
    return [dict(mode=r.mode, T=int(r.time_steps),
                 theta=None if pd.isna(r.threshold) else float(r.threshold),
                 attack=r.attack_name, acc=float(r.accuracy), n=_n_from_hist(r.step_histogram))
            for r in d.itertuples()]


def probe():
    p = HERE.parents[2] / "defense" / "figs" / "_probe.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return [dict(step=int(r.step), condition=r.condition, acc=float(r.accuracy),
                 conf=float(r.confidence)) for r in d.itertuples()]


def main():
    facts = {"configs": {k: config_facts(k) for k in CONFIGS}}
    for k, c in facts["configs"].items():
        c["compute_overuse"] = compute_overuse(c)
        c["robustness"] = single_run_robustness(c)
    facts["multiseed"] = {
        "r18_adv": multiseed("Multiseed ResNet18"),
        "vgg_adv": multiseed("Multiseed VGG16"),
    }
    facts["attack_seed"] = {
        "first_mixT_ckpt": attack_seed_summary("Multiseed"),
        "r18_adv": attack_seed_summary("Multiseed ResNet18"),
        "vgg_adv": attack_seed_summary("Multiseed VGG16"),
        "mbv2_adv": attack_seed_summary("Multiseed MobileNetV2"),
    }
    facts["train_histories"] = {
        "r18_adv": train_histories("Multiseed ResNet18"),
        "vgg_adv": train_histories("Multiseed VGG16"),
        "mbv2_adv": train_histories("Multiseed MobileNetV2"),
    }
    facts["old_subset_matched"] = old_subset_matched()
    facts["probe"] = probe()
    OUT_JSON.write_text(json.dumps(facts, indent=1), encoding="utf-8")
    write_md(facts)
    print("wrote", OUT_JSON.name, "and", OUT_MD.name)


def write_md(f):
    L = ["# Thesis facts sheet (generated by analysis/build_facts.py)", "",
         "Every number in the thesis traces to a row below. Do not hand-edit; rerun the script.", ""]
    L += ["## Operating points (neutral criterion, delta = 1.0; test set, n = 10000)", "",
          "| config | theta | depth | VT acc | matched T | fixed acc | gain | below full | SynOps saving | fallback | VT>fixed pts |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in f["configs"].items():
        s = c["selection"]["neutral_1.0"]
        L.append(f"| {k} | {s['theta']:.2f} | {s['depth']:.2f} | {s['vt_acc']:.2f} | {s['matched_T']} | "
                 f"{s['matched_acc']:.2f} | {s['gain']:+.2f} | {s['below_full']:.2f} | "
                 f"{100*s['synops_saving']:.1f}% | {s['fallback']} | "
                 f"{c['vt_points_above_fixed'][0]}/{c['vt_points_above_fixed'][1]} |")
    L += ["", "## Biased (max accuracy_delta) selection", "",
          "| config | theta | depth | VT acc | matched T | fixed acc | gain | below full |", "|---|---|---|---|---|---|---|---|"]
    for k, c in f["configs"].items():
        s = c["selection"]["biased"]
        L.append(f"| {k} | {s['theta']:.2f} | {s['depth']:.2f} | {s['vt_acc']:.2f} | {s['matched_T']} | "
                 f"{s['matched_acc']:.2f} | {s['gain']:+.2f} | {s['below_full']:.2f} |")
    L += ["", "## Delta sensitivity (gain at delta = 0.5 / 1.0 / 2.0)", "", "| config | 0.5 | 1.0 | 2.0 |", "|---|---|---|---|"]
    for k, c in f["configs"].items():
        g = [c["selection"][f"neutral_{d}"]["gain"] for d in (0.5, 1.0, 2.0)]
        L.append(f"| {k} | {g[0]:+.2f} | {g[1]:+.2f} | {g[2]:+.2f} |")
    L += ["", "## Single-run robustness (fixed T=8 vs VT selected)", "",
          "| config | attack | fixed T8 | VT | delta | theta | n |", "|---|---|---|---|---|---|---|"]
    for k, c in f["configs"].items():
        for atk, r in c["robustness"].items():
            L.append(f"| {k} | {atk} | {r['fixed_T8']:.2f} | {r['vt']:.2f} | {r['delta']:+.2f} | {r['theta']} | {r['n']} |")
    L += ["", "## Compute-overuse (depth inflation, label preserved)", "",
          "| config | theta | clean depth | attacked depth | inflation | acc kept | n |", "|---|---|---|---|---|---|---|"]
    for k, c in f["configs"].items():
        co = c["compute_overuse"]
        if co:
            L.append(f"| {k} | {co['theta']} | {co['clean_depth']:.3f} | {co['attacked_depth']:.3f} | "
                     f"{co['inflation']:+.3f} | {co['acc_kept']:.2f} | {co['n']} |")
    L += ["", "## Training-seed replication (paired VT - matched control, 3 seeds)", ""]
    for name, m in f["multiseed"].items():
        if not m:
            continue
        L.append(f"**{name}** (thetas {m['thetas']}, matched T {m['matched_T']})")
        for cond in ("clean", "pgd", "autoattack"):
            c = m[cond]
            L.append(f"- {cond}: per-seed {[round(x, 2) for x in c['per_seed']]} -> "
                     f"{c['mean']:+.2f} +/- {c['std']:.2f} ({c['verdict']})")
        L.append("")
    L += ["## Stopping rules (test set)", ""]
    for k, c in f["configs"].items():
        rows = ", ".join(f"{r['rule']}={r['acc']:.2f}@{r['depth']:.2f}" for r in c["stopping_rules"])
        L.append(f"- {k}: {rows}")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
