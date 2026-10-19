"""Write chapters/appendix_tables.tex (long result tables) from facts.json.

Run from the thesis root:  python analysis/make_tables.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = json.loads((ROOT / "facts.json").read_text())
C = F["configs"]
NAME = {
    "r18_clean_T8": "R18 clean", "r18_adv_T8": "R18 adv (1)", "r18_adv_T8_rep": "R18 adv (2)",
    "r18_clean_mixT": "R18 clean mixT", "r18_adv_mixT": "R18 adv mixT", "vgg_clean_T8": "VGG clean",
    "vgg_adv_T8": "VGG adv", "mbv2_clean_T8": "MBv2 clean", "mbv2_adv_T8": "MBv2 adv",
}
ORDER = list(NAME)
NL = "\\\\"
out = []


def table(caption, label, colspec, header, rows, size="\\small", short=None):
    cap = f"\\caption[{short}]{{{caption}}}" if short else f"\\caption{{{caption}}}"
    out.append("\\begin{table}[htbp]\n\\centering\n" + cap + f"\n\\label{{{label}}}\n{size}\n"
               f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} {NL}\n\\midrule")
    out.extend(rows)
    out.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ---- fixed sweeps with SynOps
rows = []
for k in ORDER:
    s = C[k]["fixed_sweep"]
    rows.append(f"{NAME[k]} & " + " & ".join(f"{r['synops'] / 1e6:.1f}" for r in s) + f" {NL}")
table("Mean SynOps per image ($10^6$) of fixed-timestep inference at each horizon (test set).",
      "tab:app-fixed-synops", "@{}l rrrrrrrr@{}",
      "Configuration & $T{=}1$ & 2 & 3 & 4 & 5 & 6 & 7 & 8", rows, size="\\footnotesize",
      short="SynOps of fixed-timestep inference")

# ---- VT sweeps
for k in ORDER:
    rows = []
    for r in C[k]["vt_sweep"]:
        rows.append(f"{r['theta']:.2f} & {r['acc']:.2f} & {r['depth']:.2f} & {r['synops'] / 1e6:.1f} {NL}")
    table(f"VT sweep for {NAME[k]} (test set, max-softmax rule).", f"tab:app-vt-{k}", "@{}c r r r@{}",
          "$\\theta$ & Accuracy (\\%) & Mean depth & SynOps ($10^6$)", rows, size="\\footnotesize")

# ---- neutral op points with SynOps
rows = []
for k in ORDER:
    o = C[k]["selection"]["neutral_1.0"]
    mism = 100 * (o["vt_synops"] / o["matched_synops"] - 1)
    rows.append(f"{NAME[k]} & {o['theta']:.2f} & {o['vt_synops'] / 1e6:.1f} & {o['matched_T']} & "
                f"{o['matched_synops'] / 1e6:.1f} & {mism:+.1f}\\% & {o['full_synops'] / 1e6:.1f} & "
                f"$\\pm${C[k]['ci95_at_neutral']:.2f} {NL}")
table("Compute at the VT-neutral operating points ($\\delta = 1$, test set). Mismatch: VT SynOps relative to the "
      "matched control. CI: binomial 95\\% half-width of the VT accuracy.",
      "tab:app-neutral-synops", "@{}l c r c r r r r@{}",
      "Configuration & $\\theta$ & VT SynOps & $T^*$ & Matched SynOps & Mismatch & $T{=}8$ SynOps & CI",
      rows, size="\\footnotesize", short="Compute at the VT-neutral operating points")

# ---- stopping rules for degenerate configs
rows = []
for k in ["r18_clean_mixT", "mbv2_clean_T8"]:
    for r in C[k]["stopping_rules"]:
        rows.append(f"{NAME[k]} & {r['rule'].replace('_', ' ')} & {r['acc']:.2f} & {r['depth']:.2f} {NL}")
    rows.append("\\addlinespace")
rows.pop()
table("Stopping-rule results for the two configurations whose reference depth came from a degenerate max-gain "
      "operating point (test set).", "tab:app-stoprules-degenerate", "@{}l l r r@{}",
      "Configuration & Rule & Accuracy (\\%) & Mean depth", rows, size="\\footnotesize",
      short="Stopping rules at degenerate reference depths")

# ---- all attack rows
rows = []
for k in ORDER:
    if k == "r18_adv_T8":
        continue
    for r in C[k]["attacks"]:
        th = "---" if r["theta"] is None else f"{r['theta']:.2f}"
        rows.append(f"{NAME[k]} & {r['mode']} & {th} & {r['role'].replace('_', ' ')} & {r['attack']} & "
                    f"{r['objective'].replace('_', ' ')} & {r['acc']:.2f} & {r['depth']:.2f} & {r['n']} {NL}")
    rows.append("\\addlinespace")
rows.pop()
out.append("{\\footnotesize\\setlength{\\tabcolsep}{3pt}")
out.append("\\begin{longtable}{@{}l l c l l l r r r@{}}\n"
           "\\caption[All single-run attack results]{All single-run attack results, including the neighbouring "
           "thresholds of each attacked VT operating point. The run-1 adversarial ResNet-18 is omitted (see "
           "\\secref{sec:res-threats}).}\\label{tab:app-attacks}" + NL + "\n\\toprule\n"
           "Configuration & Mode & $\\theta$ & Role & Attack & Objective & Acc. & Depth & $n$ " + NL +
           "\n\\midrule\n\\endfirsthead\n\\toprule\n"
           "Configuration & Mode & $\\theta$ & Role & Attack & Objective & Acc. & Depth & $n$ " + NL +
           "\n\\midrule\n\\endhead")
out.extend(rows)
out.append("\\bottomrule\n\\end{longtable}}\n")

# ---- attack-seed variance
rows = []
names = {"r18_adv": "ResNet-18", "vgg_adv": "VGG16",
         "mbv2_adv": "MobileNetV2"}
for key, nm in names.items():
    a = F["attack_seed"].get(key)
    if not a:
        continue
    cells = []
    for mode in ["fixed_T8", "fixed_matched", "vt_selected"]:
        for atk in ["pgd", "autoattack"]:
            v = a[f"{mode}/{atk}"]
            cells.append(f"{v['mean']:.1f} $\\pm$ {v['std']:.2f}")
    rows.append(f"{nm} & " + " & ".join(cells) + f" {NL}")
table("Attack-seed variance: robust accuracy (\\%, mean $\\pm$ SD over three attack seeds) on one adversarially "
      "trained checkpoint per backbone ($n = 1{,}000$).", "tab:app-attack-seed", "@{}l rr rr rr@{}",
      "& \\multicolumn{2}{c}{Full $T{=}8$} & \\multicolumn{2}{c}{Matched} & \\multicolumn{2}{c}{VT} " + NL +
      "\n\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(l){6-7}\nBackbone & PGD & AA & PGD & AA & PGD & AA",
      rows, size="\\footnotesize", short="Attack-seed variance")

# ---- training histories
rows = []
for key, nm in [("r18_adv", "ResNet-18"), ("vgg_adv", "VGG16"), ("mbv2_adv", "MobileNetV2")]:
    for i, h in enumerate(F["train_histories"][key]):
        rows.append(f"{nm} & {i} & {h['epochs']} & {h['final_train_adv']:.2f} & {h['best_val_clean']:.2f} {NL}")
table("Multi-seed training summaries (adversarial fixed-T8 regime). Train adv.: RFGSM training accuracy in the "
      "final epoch. Best val.: best clean validation accuracy at $T = 8$ (evaluated every five epochs).",
      "tab:app-train", "@{}l c c r r@{}", "Backbone & Seed & Epochs & Train adv.\\ (\\%) & Best val.\\ (\\%)",
      rows, size="\\footnotesize", short="Multi-seed training summaries")

(ROOT / "chapters" / "appendix_tables.tex").write_text("\n".join(out), encoding="utf-8")
print("wrote", len(out), "blocks")
