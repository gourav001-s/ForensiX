"""
Per-prediction explanation engine for the Deepfake & Image Forensics Toolkit.
Developed by Gourav Shrivastava

Turns the "explanation" structure forensics.py's compute_verdict() builds INLINE
(alongside the actual score computation, using the same variables -- see the comment
there) into ranked, human-readable output. This module only formats; it computes
nothing new, so it cannot drift from what the score actually did.

Two things this exists to make honest, not just readable:

1. WHICH features actually moved the number, and by how much -- not just which
   modules happened to score high (a module can score 0.9 and still contribute little
   if its calibrated weight is near zero; forensics.py's plain module_scores list
   doesn't show that, this does).

2. In calibrated mode, the hand-tuned "reasons" text (AI markers found, C2PA invalid,
   ML strongly fake, ...) is still generated and still shown in the report -- but NONE
   of it drives the final score once a calibration is active; the calibrated logistic
   model's own terms do, and those can legitimately disagree with what the reasons
   text implies. This module explicitly discloses that split so a person reading the
   report isn't misled into thinking every listed reason moved the number.
"""

FEATURE_LABELS = {"ml": "ML classifier", "c2pa": "C2PA provenance", "metadata": "Metadata",
                  "faces": "Face consistency", "ela": "Error Level Analysis", "noise": "Noise residual",
                  "frequency": "FFT spectrum", "copy_move": "Copy-move detection", "jpeg": "JPEG analysis",
                  "pulse": "Pulse periodicity (rPPG)", "blink": "Blink rate", "flicker": "Face-region flicker"}
OVERRIDE_LABELS = {"ai_markers": "AI-generation metadata found", "copy_move": "Cloned region count",
                   "faces": "Face-region inconsistency", "c2pa_invalid": "C2PA signature invalid (tampered)",
                   "c2pa_ai_generated": "C2PA declares AI generation", "ml_strong": "ML classifier: strong fake",
                   "ml_lean": "ML classifier: leans fake"}


def _label(key):
    return FEATURE_LABELS.get(key, key.replace("_", " ").title())


def _direction(x):
    return "up" if x > 1e-9 else ("down" if x < -1e-9 else "flat")


def ranked_contributions(verdict):
    """
    Returns a list of {feature, label, value, contribution, direction}, sorted by
    |contribution| descending -- the actual per-feature drivers of the score that WAS
    returned (calibrated_terms in calibrated mode, baseline_terms in heuristic mode).
    An empty list is possible (e.g. no modules were active) and is not an error.
    """
    e = verdict.get("explanation")
    if not e:
        return []
    terms = e["calibrated_terms"] if e["mode"] == "calibrated" else e["baseline_terms"]
    if not terms:
        return []
    out = [{"feature": t["feature"], "label": _label(t["feature"]), "value": t["value"],
           "contribution": t["contribution"], "direction": _direction(t["contribution"])}
          for t in terms]
    return sorted(out, key=lambda t: -abs(t["contribution"]))


def superseded_overrides(verdict):
    """
    Only meaningful in calibrated mode. Returns the override steps that fired (their
    "reason" text appears in verdict["reasons"]) but whose numeric effect on the score
    was discarded when the calibrated model overwrote it -- i.e. reasons that read as
    if they drove the score but structurally did not. Empty in heuristic mode (there,
    every binding override IS part of the final score by definition) or when no
    override fired at all.
    """
    e = verdict.get("explanation")
    if not e or e["mode"] != "calibrated":
        return []
    return [{"name": s["name"], "label": OVERRIDE_LABELS.get(s["name"], s["name"]), "reason": s["reason"],
             "would_have_set": s["floor"]} for s in e["override_steps"] if s["binding"]]


def self_check(verdict, tol=0.01):
    """
    Defensive consistency check: do the stored per-feature contributions actually sum
    (plus the calibrated intercept, or the heuristic baseline) back to the score that
    was returned? Should always pass given how compute_verdict() builds this -- exists
    as a tripwire if forensics.py is ever edited without updating the explanation
    alongside it, rather than silently shipping a misleading breakdown.
    """
    e = verdict.get("explanation")
    if not e:
        return True, None
    if e["mode"] == "calibrated" and e["calibrated_terms"] is not None:
        reconstructed = e["calibrated_baseline"] + sum(t["contribution"] for t in e["calibrated_terms"])
        import math
        reconstructed_score = 1.0 / (1.0 + math.exp(-reconstructed))
        ok = abs(reconstructed_score - e["final_score"]) < tol
    else:
        reconstructed_score = sum(t["contribution"] for t in e["baseline_terms"])
        # heuristic mode: final_score may differ from the raw baseline sum if an
        # override raised it -- that's expected and IS one of the override_steps, not a
        # self-check failure, so only compare against heuristic_score (pre-clamped-label
        # value), and only when no override fired.
        if e["override_steps"]:
            return True, None
        ok = abs(reconstructed_score - e["heuristic_score"]) < tol
    return ok, (None if ok else f"reconstructed {reconstructed_score:.4f} vs reported "
               f"{e.get('final_score', e.get('heuristic_score')):.4f}")


def explain_text(verdict):
    """One multi-paragraph plain-text explanation, suitable for a report or CLI output."""
    e = verdict.get("explanation")
    if not e:
        return "No explanation data available for this verdict."
    lines = []
    ok, msg = self_check(verdict)
    if not ok:
        lines.append(f"[internal consistency warning: {msg} -- treat this explanation as unreliable]")

    contribs = ranked_contributions(verdict)
    if e["mode"] == "calibrated":
        n = verdict["calibration"].get("n_images", verdict["calibration"].get("n_videos"))
        unit = "images" if "n_images" in verdict["calibration"] else "videos"
        lines.append(f"This score used a CALIBRATED model (fit on {n} labelled {unit}; "
                     f"cross-validated AUC {verdict['calibration']['cv_auc']}). "
                     f"Each module's score is compared to what was typical in that training data, "
                     f"not judged against a fixed threshold.")
    else:
        lines.append("This score used the hand-tuned heuristic (no calibration.json is active). "
                     "Each active module contributes a fixed share of the total, weighted by how "
                     "reliable that module is believed to be in general -- not measured on your data.")

    if contribs:
        lines.append("\nWhat moved the score, ranked by impact:")
        for c in contribs[:8]:
            arrow = {"up": "pushed UP (toward fake)", "down": "pushed DOWN (toward real)",
                     "flat": "had no effect"}[c["direction"]]
            lines.append(f"  - {c['label']}: module score {c['value']:.2f} -> {arrow} "
                         f"(contribution {c['contribution']:+.3f})")
    else:
        lines.append("\nNo module contributed a scored input (nothing was active for this file).")

    sup = superseded_overrides(verdict)
    if sup:
        lines.append("\nNote: the following heuristic finding(s) are shown in the report and DID fire, "
                     "but did NOT determine this calibrated score -- the calibrated model's own terms "
                     "above did instead. Read these as supporting context, not as what set the number:")
        for s in sup:
            lines.append(f"  - {s['label']}: \"{s['reason']}\"")
    elif e["mode"] == "heuristic" and e["override_steps"]:
        binding = [s for s in e["override_steps"] if s["binding"]]
        if binding:
            lines.append("\nOverride(s) that directly set the score (bypassing the weighted average above):")
            for s in binding:
                lines.append(f"  - {OVERRIDE_LABELS.get(s['name'], s['name'])}: raised the score from "
                             f"{s['pre_value']:.2f} to {s['floor']:.2f} -- \"{s['reason']}\"")

    lines.append(f"\nFinal: {verdict['percent']}/100 ({verdict['label']}).")
    return "\n".join(lines)
