"""
External "second opinion" integrations for the Deepfake & Image Forensics Toolkit.
Developed by Gourav Shrivastava

Two different trust levels live in this file -- keep them separate:

  OPT-IN (the actual image leaves the machine, only with explicit per-upload consent
  via the "Also check with external services" checkbox in app.py):
    - Reality Defender (check_reality_defender): image-only, 50 free scans/month.
      Setup:  pip install realitydefender
              set REALITY_DEFENDER_API_KEY
    - Sightengine (check_sightengine): image-only, free trial credits.
      Setup:  set SIGHTENGINE_API_USER and SIGHTENGINE_API_SECRET

  AUTOMATIC (no image or video content is ever sent -- only a SHA256 hash, which is a
  fingerprint, not the file itself -- so this runs unconditionally, same as C2PA):
    - VirusTotal (check_virustotal): looks up whether this exact file's hash is known
      malware. Different question from deepfake detection -- "is the file itself
      dangerous," not "is the content synthetic."
      Setup:  set VIRUSTOTAL_API_KEY

Every function here follows the same rule as c2pa_verify.py and mlclassifier.py:
never raise. A missing key, a missing package, or a network failure is a normal,
reported result -- not a crash.
"""
import asyncio
import os

import requests

REALITY_DEFENDER_AVAILABLE = False
_RD_IMPORT_ERROR = None
try:
    from realitydefender import RealityDefender
    REALITY_DEFENDER_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised when the optional dep is absent
    _RD_IMPORT_ERROR = str(e)


def _empty_rd(note=None, error=None):
    return {"available": False, "configured": False, "status": None, "score": None,
            "models": [], "note": note, "error": error}


async def _run_reality_defender(api_key, path):
    rd = RealityDefender(api_key=api_key)
    upload = await rd.upload(file_path=path)
    return await rd.get_result(upload["request_id"])


def check_reality_defender(path):
    """Send ONE image to Reality Defender's cloud API for a second opinion.
    Never raises -- failure here is a normal result, same as every other optional
    module in this toolkit."""
    key = os.environ.get("REALITY_DEFENDER_API_KEY")
    if not key:
        return _empty_rd("REALITY_DEFENDER_API_KEY not set -- this check is skipped.")
    if not REALITY_DEFENDER_AVAILABLE:
        return _empty_rd(f"realitydefender package not installed ({_RD_IMPORT_ERROR}). "
                         f"pip install realitydefender to enable this check.")
    try:
        result = asyncio.run(_run_reality_defender(key, path))
    except Exception as e:
        return {"available": True, "configured": True, "status": None, "score": None,
                "models": [], "note": None, "error": f"{type(e).__name__}: {e}"}
    models = [{"name": m.get("name"), "status": m.get("status"), "score": m.get("score")}
             for m in (result.get("models") or [])]
    return {"available": True, "configured": True, "status": result.get("status"),
            "score": result.get("score"), "models": models, "error": None,
            "note": "Sent to Reality Defender's cloud API (image only, 50 free scans/month). "
                    "Score is their reported manipulation probability -- treat it like this "
                    "toolkit's own ML score: uncalibrated, one more data point, not a verdict."}


SIGHTENGINE_URL = "https://api.sightengine.com/1.0/check.json"


def check_sightengine(path):
    """Send ONE image to Sightengine's genai model for a second opinion.
    Never raises -- same rule as everything else in this file."""
    user = os.environ.get("SIGHTENGINE_API_USER")
    secret = os.environ.get("SIGHTENGINE_API_SECRET")
    if not (user and secret):
        return {"available": False, "configured": False, "ai_generated_score": None,
                "note": "SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET not set -- this check "
                        "is skipped.", "error": None}
    try:
        with open(path, "rb") as f:
            resp = requests.post(SIGHTENGINE_URL, files={"media": f},
                                 data={"models": "genai", "api_user": user, "api_secret": secret},
                                 timeout=30)
        data = resp.json()
    except Exception as e:
        return {"available": True, "configured": True, "ai_generated_score": None,
                "note": None, "error": f"{type(e).__name__}: {e}"}
    if data.get("status") != "success":
        err = (data.get("error") or {}).get("message", "unknown error")
        return {"available": True, "configured": True, "ai_generated_score": None,
                "note": None, "error": f"Sightengine: {err}"}
    return {"available": True, "configured": True, "error": None,
            "ai_generated_score": (data.get("type") or {}).get("ai_generated"),
            "note": "Sent to Sightengine's cloud API (image only). Score is their confidence "
                    "the image was AI-generated, 0-1. Uncalibrated, one more data point, not "
                    "a verdict."}


def _vt_result(available=False, configured=False, found=None, malicious=None, suspicious=None,
               total_engines=None, permalink=None, note=None, error=None):
    """Every check_virustotal() branch returns exactly this shape -- built through one
    place so a future new branch can't silently omit a field (see the bug this replaced:
    one branch used to skip "note" entirely)."""
    return {"available": available, "configured": configured, "found": found,
            "malicious": malicious, "suspicious": suspicious, "total_engines": total_engines,
            "permalink": permalink, "note": note, "error": error}


def check_virustotal(sha256):
    """Look up a file's SHA256 hash against VirusTotal's database. This is a hash lookup
    ONLY -- the file itself is never uploaded or sent anywhere. Never raises."""
    key = os.environ.get("VIRUSTOTAL_API_KEY")
    if not key:
        return _vt_result(note="VIRUSTOTAL_API_KEY not set -- this check is skipped.")
    try:
        resp = requests.get(f"https://www.virustotal.com/api/v3/files/{sha256}",
                            headers={"x-apikey": key}, timeout=15)
    except Exception as e:
        return _vt_result(available=True, configured=True, error=f"{type(e).__name__}: {e}")
    permalink = f"https://www.virustotal.com/gui/file/{sha256}"
    if resp.status_code == 404:
        return _vt_result(available=True, configured=True, found=False, permalink=permalink,
                          note="File hash not in VirusTotal's database -- hash lookup only, "
                               "nothing was uploaded. A miss here does not mean the file is "
                               "safe, only that no one has scanned this exact file before.")
    if resp.status_code != 200:
        return _vt_result(available=True, configured=True,
                          error=f"VirusTotal returned HTTP {resp.status_code}")
    stats = resp.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {}) or {}
    return _vt_result(available=True, configured=True, found=True, permalink=permalink,
                      malicious=stats.get("malicious", 0), suspicious=stats.get("suspicious", 0),
                      total_engines=sum(stats.values()) if stats else 0,
                      note="Hash lookup only -- nothing was uploaded to VirusTotal. Checks "
                           "whether the FILE ITSELF is known malware -- a different question "
                           "from whether its content is a deepfake.")


def run_all(path, kind="image"):
    """Run every configured OPT-IN external check (Reality Defender, Sightengine).
    kind='video' currently runs nothing -- neither API accepts video yet. VirusTotal is
    NOT included here -- it's a hash-only lookup, called unconditionally by app.py
    instead of gated behind the opt-in checkbox. See check_virustotal()."""
    out = {}
    if kind == "image":
        out["reality_defender"] = check_reality_defender(path)
        out["sightengine"] = check_sightengine(path)
    return out
