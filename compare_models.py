#!/usr/bin/env python3
"""
Model validation / comparison for the Deepfake & Image Forensics Toolkit.
Developed by Gourav Shrivastava

Compares every local ML model in ./models against the SAME labelled dataset,
so you can decide which one(s) to keep. Reuses evaluate.py's features.csv
(each model's raw P(fake) is already captured per image as ml:<name>).

Usage:
    python3 evaluate.py --data dataset --out eval_out      # 1) extract features once (see step 3)
    python3 compare_models.py --features eval_out/features.csv   # 2) compare the models

    python3 compare_models.py --features eval_out/features.csv --normalize-features eval_out_norm/features.csv
        # optional: also compare on a --normalize run, to see if a model is really reading
        # pixels or leaning on format/exif cues that vanish once everything is re-encoded.

    python3 compare_models.py --features eval_out/features.csv --keep modelA modelB --apply
        # write models/DISABLED_<name> markers for every model NOT listed in --keep,
        # so mlclassifier.py skips them without deleting anything.

Every number is measured on YOUR dataset. A model that wins here can still fail on
generators or cameras you didn't test. Re-run whenever you add data.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

import evaluate as E
import mlclassifier as ml

HERE = os.path.dirname(os.path.abspath(__file__))


def model_columns(rows):
    return sorted({k[3:] for r in rows for k in r if k.startswith("ml:")})


def per_model_stats(rows, y, names):
    out = []
    for name in names:
        key = f"ml:{name}"
        v = E.col(rows, key)
        ok = ~np.isnan(v)
        n_missing = int((~ok).sum())
        if ok.sum() < 5 or len(set(y[ok])) < 2:
            out.append({"name": name, "n": int(ok.sum()), "missing": n_missing, "usable": False})
            continue
        yv, vv = y[ok], v[ok]
        a = E.auc(yv, vv)
        lo, hi = E.auc_ci(yv, vv)
        recall5, thr5 = E.tpr_at_fpr(yv, vv, 0.05)
        recall1, thr1 = E.tpr_at_fpr(yv, vv, 0.01)
        yj = E.youden_threshold(yv, vv)
        m = E.binary_metrics(yv, (vv >= yj).astype(int))
        ms = E.col(rows, f"mlms:{name}")
        ms = ms[~np.isnan(ms)]
        mode = next((r.get(f"mlmode:{name}") for r in rows if r.get(f"mlmode:{name}")), "?")
        out.append({"name": name, "n": int(ok.sum()), "missing": n_missing, "usable": True, "mode": mode,
                    "auc": a, "auc_lo": lo, "auc_hi": hi, "recall_at_5fpr": recall5, "thr_5fpr": thr5,
                    "recall_at_1fpr": recall1, "thr_1fpr": thr1, "youden_thr": yj,
                    "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                    "mean_ms": float(ms.mean()) if len(ms) else None,
                    "real_mean": float(vv[yv == 0].mean()), "fake_mean": float(vv[yv == 1].mean())})
    return out


def correlation_matrix(rows, names):
    cols = {n: E.col(rows, f"ml:{n}") for n in names}
    ok = np.ones(len(rows), bool)
    for c in cols.values():
        ok &= ~np.isnan(c)
    if ok.sum() < 5:
        return None
    mat = np.corrcoef(np.column_stack([cols[n][ok] for n in names]).T)
    return mat, int(ok.sum())


def ensemble_auc(rows, y, names, subset):
    cols = [E.col(rows, f"ml:{n}") for n in subset]
    ok = np.ones(len(rows), bool)
    for c in cols:
        ok &= ~np.isnan(c)
    if ok.sum() < 5 or len(set(y[ok])) < 2:
        return None
    mean = np.mean(np.column_stack([c[ok] for c in cols]), axis=1)
    return E.auc(y[ok], mean), int(ok.sum())


def best_subset(rows, y, names, max_k=3):
    """Try every combination up to max_k models, return the one with highest ensemble AUC."""
    results = []
    for k in range(1, min(max_k, len(names)) + 1):
        for combo in itertools.combinations(names, k):
            r = ensemble_auc(rows, y, names, combo)
            if r:
                results.append((combo, r[0], r[1]))
    results.sort(key=lambda x: -x[1])
    return results


def fmt(x, p=3):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{p}f}"


def failed_models(rows):
    errs = set()
    for r in rows:
        if r.get("ml_errors"):
            errs.update(e.strip() for e in str(r["ml_errors"]).split("|"))
    return sorted(errs)


def build_report(rows, y, stats, corr, best, hand_auc, cur_ensemble_auc):
    L = ["# Model comparison report\n", f"Developed by Gourav Shrivastava | {len(rows)} images "
         f"(real {int((y == 0).sum())}, fake {int((y == 1).sum())})\n"]
    usable = [s for s in stats if s["usable"]]
    unusable = [s for s in stats if not s["usable"]]
    failed = failed_models(rows)
    if failed:
        L.append("## 0. Models that failed to load (excluded below)\n")
        for e in failed:
            L.append(f"- {e}")
        L.append("")

    L.append("## 1. Per-model results\n")
    if not stats:
        L.append("No `ml:*` columns found in features.csv. Either no models are installed, or every model failed "
                 "during extraction (check eval_out/errors.txt and the 'ML Models' page for load errors).")
        return "\n".join(L)
    L.append("| Model | Mode | N | AUC (95% CI) | Recall@5%FPR | Recall@1%FPR | Best-threshold acc | "
             "Mean ms/img | Real mean | Fake mean |\n|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(usable, key=lambda x: -x["auc"]):
        L.append(f"| {s['name']} | {s['mode']} | {s['n']} | {fmt(s['auc'])} ({fmt(s['auc_lo'])}-{fmt(s['auc_hi'])}) | "
                 f"{fmt(s['recall_at_5fpr'])} | {fmt(s['recall_at_1fpr'])} | {fmt(s['accuracy'])} | "
                 f"{fmt(s['mean_ms'], 0)} | {fmt(s['real_mean'])} | {fmt(s['fake_mean'])} |")
    for s in unusable:
        L.append(f"| {s['name']} | - | {s['n']} | not enough data ({s['missing']} missing / failed) | | | | | | |")
    L.append("\nAUC 0.5 = coin flip. Recall@FPR = how many fakes are caught while keeping false alarms on real "
             "images at or below that rate — the number that matters more than raw accuracy for a detector.")

    if len(usable) == 1:
        s = usable[0]
        verdict = ("keep it" if s["auc"] >= 0.75 else
                   "borderline: better than nothing but weak" if s["auc"] >= 0.6 else
                   "not useful on this data: no better than a coin flip, do not rely on it")
        L.append(f"\nOnly one usable model (**{s['name']}**), AUC {fmt(s['auc'])}: {verdict}.")
    elif usable:
        ranked = sorted(usable, key=lambda x: -x["auc"])
        top = ranked[0]
        L.append(f"\nBest single model on this data: **{top['name']}** (AUC {fmt(top['auc'])}).")
        if top["auc"] < 0.65:
            L.append("Every model is weak here (AUC < 0.65). That usually means the dataset is too easy/hard, "
                     "too small, or the models were trained on very different data than yours. Do not deploy "
                     "any of them as the main signal yet; treat forensics.py's other modules as primary.")

    if corr is not None and len(usable) > 1:
        mat, n = corr
        names = [s["name"] for s in usable]
        L.append(f"\n## 2. Agreement between models (n={n})\n")
        L.append("Pearson correlation of raw scores. Low correlation between two strong models means they catch "
                 "different fakes, so an ensemble helps. High correlation means one is redundant.\n")
        L.append("| |" + "|".join(names) + "|\n|" + "---|" * (len(names) + 1))
        for i, ni in enumerate(names):
            L.append("| " + ni + " |" + "|".join(fmt(mat[i, j], 2) for j in range(len(names))) + "|")

    if best and len(usable) > 1:
        L.append("\n## 3. Ensemble search (mean of raw scores)\n")
        L.append(f"Averaging ALL {len(usable)} installed models: AUC {fmt(cur_ensemble_auc)} "
                 f"(this is what the app currently does, mode permitting).\n")
        L.append("| Model(s) | AUC | N |\n|---|---|---|")
        for combo, a, n in best[:8]:
            L.append(f"| {' + '.join(combo)} | {fmt(a)} | {n} |")
        top_combo, top_auc, _ = best[0]
        single_best = max(usable, key=lambda x: x["auc"])
        if top_auc > single_best["auc"] + 0.01:
            L.append(f"\nBest combination beats the best single model ({fmt(single_best['auc'])} -> {fmt(top_auc)}). "
                     f"Recommended: keep {list(top_combo)}.")
        else:
            L.append(f"\nNo combination clearly beats the single best model ({single_best['name']}, "
                     f"AUC {fmt(single_best['auc'])}). Extra models add inference time for no accuracy gain here; "
                     f"recommended: keep only {single_best['name']}.")

    L.append(f"\n## 4. Compared to the non-ML heuristic\n")
    L.append(f"- Heuristic-only AUC (forensics modules, no ML): {fmt(hand_auc)}")
    if usable:
        best_ml = max(s['auc'] for s in usable)
        L.append(f"- Best ML model AUC: {fmt(best_ml)}")
        if best_ml < hand_auc:
            L.append("The heuristic modules alone outperform every installed model on this data. That can be a "
                     "real result for this dataset, or a sign the model's training data doesn't match your "
                     "images (resolution, compression, subject matter). Try a different model before trusting this.")
    L.append("\n## 5. Read this honestly\n")
    L.append("- A model's published benchmark numbers are not these numbers. Only trust what you measured here.\n"
             "- Re-run this after adding new generators. A model that wins today can lose to next month's generator.\n"
             "- If `--normalize-features` was given, check whether AUC dropped a lot after normalization. A big "
             "drop means the model (or the heuristic) was partly reading file format, not pixel content.")
    return "\n".join(L)


def apply_keep(keep):
    if not os.path.isdir(ml.MODELS_DIR):
        sys.exit(f"No models directory at {ml.MODELS_DIR}")
    changed = []
    for name in sorted(os.listdir(ml.MODELS_DIR)):
        folder = os.path.join(ml.MODELS_DIR, name)
        if not os.path.isdir(folder) or name.startswith("."):
            continue
        marker = os.path.join(folder, "DISABLED")
        if name in keep:
            if os.path.exists(marker):
                os.remove(marker)
                changed.append(f"enabled {name}")
        else:
            if not os.path.exists(marker):
                open(marker, "w").write("Disabled by compare_models.py --apply. Delete this file to re-enable.\n")
                changed.append(f"disabled {name}")
    return changed


def main():
    ap = argparse.ArgumentParser(description="Compare installed ML models on a labelled dataset")
    ap.add_argument("--features", required=True, help="features.csv produced by evaluate.py")
    ap.add_argument("--normalize-features", help="optional second features.csv from an evaluate.py --normalize run")
    ap.add_argument("--out", default=os.path.join(HERE, "eval_out"))
    ap.add_argument("--max-ensemble", type=int, default=3, help="max models to try combining")
    ap.add_argument("--keep", nargs="*", help="model names to keep active; all others get disabled")
    ap.add_argument("--apply", action="store_true", help="write DISABLED markers per --keep (models/ not deleted)")
    args = ap.parse_args()

    if args.apply and not args.keep:
        sys.exit("--apply requires --keep <model names...>")

    rows = E.read_csv(args.features)
    y = np.array([r["label"] for r in rows])
    names = model_columns(rows)
    if not names and not failed_models(rows):
        print("No ml:* columns and no recorded model errors in this CSV. Was any model installed when "
              "evaluate.py ran? Install one, re-run evaluate.py, then re-run this.")
        return
    stats = per_model_stats(rows, y, names)
    corr = correlation_matrix(rows, [s["name"] for s in stats if s["usable"]])
    best = best_subset(rows, y, [s["name"] for s in stats if s["usable"]], args.max_ensemble) if len(
        [s for s in stats if s["usable"]]) > 1 else None
    hand = E.col(rows, "hand_score")
    # "current ensemble" = mean of ALL usable models, matching mlclassifier.classify()'s default behaviour
    cur = ensemble_auc(rows, y, [s["name"] for s in stats if s["usable"]], [s["name"] for s in stats if s["usable"]])
    report = build_report(rows, y, stats, corr, best, E.auc(y, hand), cur[0] if cur else None)

    if args.normalize_features:
        nrows = E.read_csv(args.normalize_features)
        ny = np.array([r["label"] for r in nrows])
        nnames = model_columns(nrows)
        nstats = per_model_stats(nrows, ny, nnames)
        report += "\n\n## 6. Same models, normalized (format-neutral) dataset\n\n"
        report += "| Model | AUC (normalized) | AUC (original) | Drop |\n|---|---|---|---|\n"
        by_name = {s["name"]: s for s in stats if s["usable"]}
        for s in nstats:
            if not s["usable"]:
                continue
            orig = by_name.get(s["name"])
            drop = (orig["auc"] - s["auc"]) if orig else None
            flag = "  <- relies on format/metadata cues" if drop is not None and drop > 0.1 else ""
            report += f"| {s['name']} | {fmt(s['auc'])} | {fmt(orig['auc']) if orig else 'n/a'} | {fmt(drop)}{flag} |\n"

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "model_comparison.md"), "w") as f:
        f.write(report)
    with open(os.path.join(args.out, "model_comparison.json"), "w") as f:
        json.dump({"stats": stats, "best_subset": [(list(c), a, n) for c, a, n in (best or [])][:10]},
                   f, indent=2, default=lambda o: None if o != o else float(o))
    print(report)
    print(f"\nFiles written to {args.out}: model_comparison.md, model_comparison.json")

    if args.apply:
        changed = apply_keep(set(args.keep))
        if changed:
            print("\nApplied:", ", ".join(changed))
        else:
            print("\nNo changes (already matches --keep).")
        print("Restart the app / open 'ML Models' to confirm the active set.")


if __name__ == "__main__":
    main()
