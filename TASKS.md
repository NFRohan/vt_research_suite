# VT Research — DEFENSE PLAN (6 days)

Defense: ~2026-09-19. **Data collection is CLOSED. No new experiments.**
Everything below is writing, except one 20-minute correction.

---

## THE THESIS (write toward this sentence)

> VT's advantage over compute-matched fixed-time truncation is **contingent on the
> backbone's training schedule**. It is real when the backbone is trained only at the
> full horizon (+2.73 clean / +1.37 adv) and vanishes into statistical noise when the
> backbone is trained across horizons (+0.14 / +0.68). We additionally show that the
> operating-point selection criterion in common use **inflates** the reported
> advantage (+4.19 -> +2.73 on our own prior numbers).

This *extends* the paper's original fair-comparison thesis: the field compute-matches
the baseline but never **training**-matches it.

---

## DATA: COMPLETE (do not add to this)

**Fixed-T8 column — 3 backbones, free fold-in of existing VGG/MobileNet data.**
All four VGG/MBv2 runs fingerprint as fixed-T8 trained (T=1 accuracy == 10.00, chance).

| Backbone | clean gain | depth | adv gain | depth |
|---|---|---|---|---|
| ResNet-18   | **+2.73** | 3.01/8 | +1.37 | 5.01/8 |
| VGG16       | **+3.12** | 3.89/8 | +0.89 | 5.74/8 |
| MobileNetV2 | **+3.37** | 7.41/8 | +0.28 | 7.21/8 |

**mixT column — ResNet only:** clean **+0.14**, adv **+0.68**.

Readings:
- VT's advantage is real and consistent for CLEAN-trained backbones (~+3, all three
  architectures, all outside the noise floor).
- It is weak-to-absent for ADVERSARIALLY trained backbones (2 of 3 inside noise).
- It also vanishes under mixed-horizon training (ResNet: +2.73 -> +0.14 clean).
- CAVEAT: MobileNet selects depth 7.41/8 -- only ~7% step saving, so its +3.37 is
  near-full-horizon inference, not an efficiency win. Always report depth with gain.

All numbers: `Findings/reselect_operating_points.py` (neutral criterion, delta=1.0,
selected on calibration, reported on test). Noise floor at n=10000 ~ +/-0.6 to +/-1.0.

## DAY 1 — Correct + rebuild numbers

- [ ] **1.1 (20 min, URGENT) Fix the false status note** in the draft. It states the
      *annealing* story as fact; that is wrong. Real cause: `mixed_time_steps=(8,)` vs
      `(1..8)`. Rewrite or delete the box.
- [ ] **1.2 Re-select every operating point** with the neutral criterion and update
      Table 1 + all in-text numbers. Script is post-hoc, no GPU.
- [ ] **1.3 Regenerate ResNet figures** — `plot_regime_comparison_suite.py` +
      `plot_pareto_frontiers.py`, copy into `paper_figures/`.

## DAY 2-3 — Rewrite the argument

- [ ] **2.1 Abstract + Intro** around the thesis above.
- [ ] **2.2 Add a Methods subsection on operating-point selection** — state the neutral
      criterion and *why* the max-delta criterion is biased. This is a contribution,
      not an apology.
- [ ] **2.3 Results: the 2x2** as the core table.
- [ ] **2.4 Results: the fixed-T sweep asymmetry** (mixT helps adv everywhere, clean
      only at T=1).

## DAY 4 — Robustness + close

- [ ] **4.1 Simplify the robustness section.** Keep: clean-trained collapse (replicated),
      attack-dependence. **DROP the subset-based matched-control tables** — that removes
      the unresolved 20.96-vs-44.1 conflict without any rerun. Report full-test only and
      state `n` explicitly.
- [ ] **4.2 Discussion + Limitations + Conclusion.**

## DAY 5 — Format + polish

- [ ] **5.1 Add binomial CIs** to every headline number. Mandatory: three of four 2x2
      cells are inside noise and must not be claimed as wins.
- [ ] **5.2 Proper paper formatting** (the draft is a working doc, not a paper yet).
- [ ] **5.3 Limitations, stated plainly:** single training seed; one backbone for the
      schedule claim; VGG/MobileNet not re-run under mixT; robustness at differing `n`.

## DAY 6 — Buffer + defense prep

- [ ] **6.1 Anticipate the obvious questions:** why the selection criterion changed;
      why ResNet only for the schedule claim; why single-seed.

---

## EXPLICITLY DROPPED (do not reopen before the defense)

- No new experiments of any kind. The 2x2 is complete.
- Do NOT patch `experiments.py` — selection is post-hoc; patching would force reruns
  and split the corpus across two criteria for zero gain.
- MobileNet/VGG fold-in: DONE (free, existing data, fixed-T8 column).
- Still dropped: VGG-mixT, MobileNet-mixT, multi-seed training bars.
- No sponge / compute-overuse paper (that is the *next* project).
- No resolution of the 20.96/44.1 conflict — sidestepped by dropping those tables.
