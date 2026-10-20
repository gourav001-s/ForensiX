"""
C2PA (Content Credentials) verification -- real cryptographic checks, no API.
Developed by Gourav Shrivastava

Uses the c2pa-python bindings (Rust c2pa-rs under the hood) to:
  1. Detect whether a C2PA manifest is embedded at all.
  2. Verify the manifest's signature and content hashes are internally consistent
     (i.e. the asset was NOT modified after signing).
  3. Report whether the signing certificate chains to a trusted root.
  4. Surface what the manifest actually claims (generator, actions, AI training/generation type).

This replaces the old "string search for the word firefly" check with a real
signature verification. Install with:  pip install c2pa-python

Three independent outcomes, and they must not be collapsed into one:
  - No manifest found      -> proves nothing. Most images never had one, or it was
                               stripped by re-upload/screenshot. Absence != authenticity.
  - Manifest present, INVALID (tampered) -> the strongest single signal this engine can
                               produce: the file was modified after a credential was attached.
  - Manifest present, VALID  -> the file matches what was signed. Whether that ORIGIN is
                               trustworthy depends separately on whether the signer's
                               certificate chains to a trusted root (see "trusted" below).
                               An untrusted signer (e.g. a self-issued test cert) can still
                               produce a "valid" (= internally consistent, unmodified) manifest.
"""

import json
import os

try:
    import c2pa
    AVAILABLE = True
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - exercised when the optional dep is absent
    c2pa = None
    AVAILABLE = False
    _IMPORT_ERROR = str(e)

# Codes from c2pa-rs's validation_results that specifically mean "signer not trusted"
# (as opposed to "content was tampered", which shows up as hash/dataHash mismatches).
_TRUST_FAIL_CODES = {"signingCredential.untrusted", "signingCredential.expired",
                     "signingCredential.revoked", "signingCredential.invalid"}
# Generation/edit hints worth surfacing verbatim from c2pa.actions assertions.
_GENAI_ACTIONS = {"c2pa.created", "c2pa.opened", "c2pa.placed", "c2pa.edited"}


def empty_result(note=None):
    return {"available": AVAILABLE, "has_manifest": False, "validation_state": None,
            "signature_valid": None, "trusted": None, "generator": None, "issuer": None,
            "signed_time": None, "actions": [], "ai_generated": False, "ingredients": 0,
            "findings": [], "score": 0.0, "note": note, "error": None}


def _extract_actions(manifest):
    out = []
    for a in manifest.get("assertions", []) or []:
        if a.get("label", "").startswith("c2pa.actions") or a.get("label", "") == "c2pa.actions":
            data = a.get("data", {})
            for act in data.get("actions", []) or []:
                out.append({"action": act.get("action"), "source_type": act.get("digitalSourceType"),
                           "software": (act.get("softwareAgent") or {}).get("name")
                           if isinstance(act.get("softwareAgent"), dict) else act.get("softwareAgent")})
    return out


def verify(path):
    """Run real C2PA verification on a local file. Never raises; failure is a normal result."""
    if not AVAILABLE:
        return empty_result(f"c2pa-python not installed ({_IMPORT_ERROR}). "
                            f"pip install c2pa-python to enable real signature verification.")
    res = empty_result()
    try:
        with c2pa.Reader(path) as reader:
            res["has_manifest"] = True
            manifest = reader.get_active_manifest() or {}
            state = reader.get_validation_state()  # "Valid" | "Invalid" | "Trusted" (SDK-dependent strings)
            vr = reader.get_validation_results() or {}
            active = vr.get("activeManifest", {}) or {}
            failures = active.get("failure", []) or []
            fail_codes = {f.get("code", "") for f in failures}
            content_tampered = bool(fail_codes - _TRUST_FAIL_CODES)  # any non-trust failure = tamper/hash issue
            trust_failed = bool(fail_codes & _TRUST_FAIL_CODES)

            res["validation_state"] = state
            res["signature_valid"] = bool(state and state.lower() not in ("invalid",) and not content_tampered)
            res["trusted"] = res["signature_valid"] and not trust_failed
            res["generator"] = manifest.get("claim_generator")
            sig = manifest.get("signature_info") or {}
            res["issuer"] = sig.get("issuer") or sig.get("common_name")
            res["signed_time"] = sig.get("time") or sig.get("date")
            res["actions"] = _extract_actions(manifest)
            res["ingredients"] = len(manifest.get("ingredients", []) or [])
            res["ai_generated"] = any(
                a.get("source_type") and "trainedalgorithmicmedia" in a["source_type"].lower() for a in res["actions"])

            f = res["findings"]
            if not res["signature_valid"]:
                f.append("C2PA manifest present but signature/content hash is INVALID: the file was modified "
                        "after the credential was attached. Strong tamper evidence.")
                res["score"] = 0.85
            elif not res["trusted"]:
                f.append("C2PA manifest is internally valid (unmodified since signing) but the signing "
                        "certificate does not chain to a trusted root (e.g. self-issued or unknown issuer). "
                        "Cannot vouch for who actually signed it.")
                res["score"] = 0.05
            else:
                f.append(f"C2PA manifest VALID and TRUSTED, issued by '{res['issuer']}'. Content unmodified "
                        f"since signing.")
                res["score"] = 0.0
            if res["ai_generated"]:
                f.append("Manifest declares AI/algorithmic generation (digitalSourceType: "
                        "trainedAlgorithmicMedia).")
                res["score"] = max(res["score"], 0.8 if res["signature_valid"] else 0.5)
            if res["ingredients"]:
                f.append(f"Manifest lists {res['ingredients']} source ingredient(s) (this asset was composed "
                        f"from other credentialed assets).")
            acts = [a["action"] for a in res["actions"] if a.get("action")]
            if acts:
                f.append("Declared actions: " + ", ".join(dict.fromkeys(acts)))
    except Exception as e:
        name = type(e).__name__
        if c2pa is not None and isinstance(e, c2pa.C2paError):
            # ManifestNotFound (and similar "no CAI data") is the ordinary, common case: just no credentials.
            if "notfound" in name.lower() or "manifestnotfound" in str(e).lower().replace(" ", "") or \
               "no jumbf" in str(e).lower():
                res["note"] = "No C2PA manifest found (most images never had one, or it was stripped on re-save)."
                return res
        res["error"] = f"{name}: {e}"
        res["note"] = "Could not evaluate C2PA data for this file."
    return res
