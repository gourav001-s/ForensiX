#!/usr/bin/env python3
"""
Deepfake & Image Forensics Toolkit (Flask)
Developed by Gourav Shrivastava
"""

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file (REALITY_DEFENDER_API_KEY etc.) if present.
                   # Never crashes the app if python-dotenv isn't installed -- you'd just
                   # need to set the env var another way (e.g. PowerShell:
                   # $env:REALITY_DEFENDER_API_KEY = "...").
except ImportError:
    pass

import json
import os
import re
import time
import uuid

import imagehash
from flask import Flask, Response, abort, redirect, render_template_string, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import forensics
import explain
import mlclassifier
import second_opinion
import video

DEVELOPER = "Gourav Shrivastava"
APP_NAME = "Deepfake & Image Forensics Toolkit"
VERSION = "3.1"

APP_ROOT = os.path.abspath(os.path.dirname(__file__))
REPORTS = os.path.join(APP_ROOT, "data", "reports")
os.makedirs(REPORTS, exist_ok=True)

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "webp"}
VIDEO_EXT = {"mp4", "mov", "avi", "mkv", "webm", "m4v"}
CASE_RE = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB hard ceiling (video needs headroom)
IMAGE_MAX_BYTES = 25 * 1024 * 1024   # tighter cap for images specifically, enforced after save (see analyze_route)
VIDEO_MAX_BYTES = 200 * 1024 * 1024


# ------------------------------------------------------------------ templates
BASE_CSS = """
:root{--bg:#0b0f14;--card:#141b24;--line:#22303f;--fg:#e6edf3;--mut:#8b9bb0;--acc:#00aaff;
--ok:#2ecc71;--warn:#f5b301;--bad:#ff4d4f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;display:flex;flex-direction:column}
a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;gap:20px;align-items:center;padding:14px 28px;border-bottom:1px solid var(--line)}
nav .brand{font-weight:800;color:var(--acc);font-size:1.1rem;margin-right:auto}
main{width:100%;max-width:1200px;margin:0 auto;padding:24px 20px;flex:1}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:20px}
h1,h2,h3{margin:0 0 12px}h2{color:var(--acc);font-size:1.15rem}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:7px 9px;
text-align:left;font-size:.9rem;vertical-align:top;word-break:break-all}th{color:var(--mut);font-weight:600}
pre{background:#0a0e13;border:1px solid var(--line);padding:12px;border-radius:8px;overflow:auto;
max-height:280px;font-size:.8rem;white-space:pre-wrap;word-break:break-all}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.grid img,.mod img{width:100%;border-radius:8px;border:1px solid var(--line)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}
.muted{color:var(--mut);font-size:.85rem}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:700;font-size:.85rem}
.LOW{background:#12351f;color:var(--ok)}.MEDIUM{background:#3a2f0a;color:var(--warn)}.HIGH{background:#3d1416;color:var(--bad)}
.bar{background:#0a0e13;border-radius:6px;height:10px;overflow:hidden;margin:4px 0 10px}
.bar>div{height:100%;background:linear-gradient(90deg,var(--ok),var(--warn),var(--bad))}
.cbar{background:#0a0e13;border-radius:5px;height:8px;overflow:hidden;margin:3px 0;position:relative}
.cbar>div{height:100%;border-radius:5px}
.warnbox{background:#3a2f0a;border:1px solid var(--warn);border-radius:8px;padding:12px;margin-top:10px}
.errbox{background:#3d1416;border:1px solid var(--bad);border-radius:8px;padding:12px;margin-top:10px}
.big{font-size:3rem;font-weight:800;line-height:1}
button,.btn{background:var(--acc);color:#04121c;border:0;padding:11px 18px;border-radius:8px;
font-weight:700;cursor:pointer;font-size:1rem;display:inline-block}
input[type=file],input[type=text]{width:100%;padding:11px;border-radius:8px;border:1px solid var(--line);
background:#0a0e13;color:var(--fg);margin-bottom:12px}
#drop{border:2px dashed var(--line);border-radius:12px;padding:30px;text-align:center;margin-bottom:12px}
#drop.over{border-color:var(--acc)}
footer{text-align:center;color:var(--mut);padding:18px;font-size:.85rem;border-top:1px solid var(--line)}
li{margin-bottom:5px}
@media print{nav,footer,button{display:none}body{background:#fff;color:#000}.card{border-color:#ccc;background:#fff}}
"""

LAYOUT_TOP = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{title}} | {{app}}</title><style>""" + BASE_CSS + """</style></head><body>
<nav><span class="brand">&#128269; {{app}}</span><a href="{{url_for('index')}}">Analyze</a>
<a href="{{url_for('history')}}">Case History</a><a href="{{url_for('compare')}}">Compare</a><a href="{{url_for('models_page')}}">ML Models</a></nav><main>"""

LAYOUT_BOTTOM = """</main><footer>{{app}} v{{ver}} &middot; Developed by {{dev}} &copy; 2025&ndash;2026<br>
Heuristic analysis only. Results are indicators, not legal or definitive proof.</footer></body></html>"""

INDEX = LAYOUT_TOP + """
<div class="card" style="max-width:640px;margin:30px auto">
<h1>Upload an image or video</h1>
<p class="muted">Runs locally. Nothing leaves your machine. Max 200 MB. Images: JPG, PNG, WEBP, BMP, TIFF, GIF.
Video: MP4, MOV, AVI, MKV, WEBM, M4V (frames are sampled evenly across the whole clip, up to 90 frames,
regardless of length).</p>
<form method="POST" action="{{url_for('analyze_route')}}" enctype="multipart/form-data">
<div id="drop"><input type="file" name="file" id="f" accept="image/*,video/*" required>
<div class="muted">or drag and drop a file here</div></div>
<input type="text" name="note" placeholder="Case note (optional)" maxlength="500">
<label style="display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:.9rem">
<input type="checkbox" name="external_check" value="1">
Also check with external services (sends the image to Reality Defender + Sightengine's cloud -- image files only)
</label>
<button type="submit">Analyze</button></form></div>
<div class="card" style="max-width:640px;margin:0 auto"><h2>What it checks</h2>
<ul class="muted"><li>AI-generator / C2PA metadata and PNG prompt chunks</li>
<li>Face region vs background: sharpness, noise, compression</li>
<li>Error Level Analysis, noise residual map, FFT spectrum (GAN artifacts)</li>
<li>Copy-move cloning, JPEG double compression, LSB steganography</li>
<li>Hashes, appended data, extension mismatch, optional CLI tools</li><li>Video: rPPG pulse periodicity, blink rate, face-region flicker, per-frame ML scoring</li></ul></div>
<script>
const d=document.getElementById('drop'),f=document.getElementById('f');
['dragenter','dragover'].forEach(e=>d.addEventListener(e,x=>{x.preventDefault();d.classList.add('over')}));
['dragleave','drop'].forEach(e=>d.addEventListener(e,x=>{x.preventDefault();d.classList.remove('over')}));
d.addEventListener('drop',x=>{f.files=x.dataTransfer.files});
</script>""" + LAYOUT_BOTTOM

REPORT = LAYOUT_TOP + """
<div class="card"><div class="two">
<div><h1>Forensic Report</h1>
<p class="muted">{{meta.filename}} &middot; case {{case}} &middot; {{meta.time}}</p>
{% if meta.note %}<p>Note: {{meta.note}}</p>{% endif %}
<span class="badge {{v.label}}">{{v.label}} RISK</span>
<p>{{v.description}}</p>
{% if v.calibration %}<p class="muted">Calibrated score (trained on {{v.calibration.n_images}} labelled images{% if v.calibration.cv_auc %}, cross-validated AUC {{'%.2f'|format(v.calibration.cv_auc)}}{% endif %}). Valid only for images similar to that data.</p>
{% else %}<p class="muted">Uncalibrated heuristic score. Run evaluate.py to measure and calibrate it.</p>{% endif %}
<a class="btn" href="{{url_for('case_json',case=case)}}">Download JSON</a>
<a class="btn" href="#" onclick="window.print();return false">Print / PDF</a></div>
<div><div class="big">{{v.percent}}<span style="font-size:1.2rem">/100</span></div>
<div class="muted">Manipulation / synthetic indicator score</div>
<div class="bar"><div style="width:{{v.percent}}%"></div></div>
{% for k,s in v.module_scores.items() %}<div class="muted">{{{'ml':'ML classifier','c2pa':'C2PA provenance'}[k] if k in ('ml','c2pa') else k|replace('_',' ')|title}}: {{'%.2f'|format(s)}}</div>
<div class="bar" style="height:5px"><div style="width:{{(s*100)|int}}%"></div></div>{% endfor %}</div></div></div>

<div class="card"><h2>Why this score</h2>
{% if v.reasons %}<ul>{% for x in v.reasons %}<li>{{x}}</li>{% endfor %}</ul>
{% else %}<p class="muted">No anomalies triggered.</p>{% endif %}</div>

{% if contribs is defined %}<div class="card"><h2>Score breakdown</h2>
<p class="muted">{% if v.calibration %}Calibrated model (trained on {{v.calibration.n_images}} labelled images).
Each module's contribution below is measured against what was typical in that training data.
{% else %}Hand-tuned heuristic (no calibration active). Each module contributes a fixed share of the total.{% endif %}</p>
{% if not self_check_ok %}<div class="errbox"><b>Internal consistency warning:</b> {{self_check_msg}}.
This breakdown may not exactly match the score above -- treat it as unreliable and report this as a bug.</div>{% endif %}
{% if contribs %}<table><tr><th>Module</th><th>Score</th><th>Impact</th><th></th></tr>
{% for c in contribs %}<tr><td>{{c.label}}</td><td>{{'%.2f'|format(c.value)}}</td>
<td>{{'%+.3f'|format(c.contribution)}} <span class="muted">({{'toward fake' if c.direction=='up' else ('toward real' if c.direction=='down' else 'no effect')}})</span></td>
<td style="width:35%"><div class="cbar"><div style="width:{{(( c.contribution|abs / max_contrib)*100)|int}}%;
background:{{'var(--bad)' if c.direction=='up' else ('var(--ok)' if c.direction=='down' else 'var(--line)')}};
{% if c.direction=='down' %}margin-left:{{100-((c.contribution|abs / max_contrib)*100)|int}}%{% endif %}"></div></div></td>
</tr>{% endfor %}</table>
<p class="muted">Ranked by impact on the final score. "Toward fake" bars in red push the score up; "toward real" bars
in green pull it down. A module can score high and still barely move the number if its weight here is low.</p>
{% else %}<p class="muted">No module had a scored contribution for this file.</p>{% endif %}
{% if superseded %}<div class="warnbox"><b>Note:</b> the following finding(s) above under "Why this score" fired,
but did <b>not</b> determine this calibrated score -- the module breakdown above did instead. Read these as
supporting context, not as what set the number:<ul>{% for s in superseded %}
<li>{{s.label}}: &ldquo;{{s.reason}}&rdquo;</li>{% endfor %}</ul></div>
{% elif not v.calibration and v.explanation.override_steps %}
{% set binding = v.explanation.override_steps|selectattr('binding')|list %}
{% if binding %}<div class="warnbox"><b>Note:</b> the weighted average above was overridden directly by:
<ul>{% for s in binding %}<li>raised from {{'%.2f'|format(s.pre_value)}} to {{'%.2f'|format(s.floor)}}
&mdash; &ldquo;{{s.reason}}&rdquo;</li>{% endfor %}</ul></div>{% endif %}{% endif %}
</div>{% endif %}

<div class="card"><h2>Visual analysis</h2><div class="grid">
{% for key,title in [('ela','Error Level Analysis'),('noise','Noise residual map'),
('frequency','FFT spectrum'),('copy_move','Copy-move detection'),('faces','Face regions')] %}
{% if r[key].file %}<div class="mod"><img src="{{url_for('case_file',case=case,path='analysis/'+r[key].file)}}" alt="{{title}}">
<b>{{title}}</b><div class="muted">{{r[key].note}}</div></div>{% endif %}{% endfor %}
</div></div>


<div class="card"><h2>Local ML classifier</h2>
{% if r.ml.available %}
<p>Mean P(fake): <b>{{'%.2f'|format(r.ml.score)}}</b> across {{r.ml.models|length}} model(s).
<span class="muted">Uncalibrated model output. Validate on your own labelled samples.</span></p>
<table><tr><th>Model</th><th>Backend</th><th>Mode</th><th>Full image</th><th>Face crops</th><th>Score</th><th>Time</th></tr>
{% for m in r.ml.models %}<tr><td>{{m.name}}</td><td>{{m.backend}}</td><td>{{m.mode}}</td>
<td>{{m.full if m.full is not none else '-'}}</td><td>{{m.faces if m.faces else '-'}}</td>
<td><b>{{m.score}}</b></td><td>{{m.ms}} ms</td></tr>{% endfor %}</table>
{% else %}<p class="muted">No ML model active. {% if r.ml.installed %}{{r.ml.installed}} model folder(s) found but none produced a result.{% else %}Install one to enable it.{% endif %}
See <a href="{{url_for('models_page')}}">ML Models</a>.</p>{% endif %}
{% for e in r.ml.errors %}<p class="muted" style="color:var(--warn)">{{e}}</p>{% endfor %}</div>

{% if r.second_opinion %}<div class="card"><h2>Second opinions</h2>
{% set vt = r.second_opinion.virustotal %}
<h3>VirusTotal <span class="muted" style="font-weight:400">(automatic -- checks the file's hash only, nothing uploaded)</span></h3>
{% if vt.error %}<p class="muted" style="color:var(--warn)">{{vt.error}}</p>
{% else %}{% if vt.found %}<p>Detections: <b>{{vt.malicious}}</b> malicious / {{vt.suspicious}} suspicious of {{vt.total_engines}} engines
&middot; <a href="{{vt.permalink}}" target="_blank" rel="noopener">view on VirusTotal</a></p>{% endif %}
<p class="muted">{{vt.note}}</p>{% endif %}

{% if r.second_opinion.reality_defender %}<h3 style="margin-top:16px">Reality Defender <span class="muted" style="font-weight:400">(opt-in -- image sent to their cloud)</span></h3>
{% set rd = r.second_opinion.reality_defender %}
{% if rd.error %}<p class="muted" style="color:var(--warn)">{{rd.error}}</p>
{% elif not rd.configured %}<p class="muted">{{rd.note}}</p>
{% else %}<p>Status: <b>{{rd.status}}</b>{% if rd.score is not none %} &middot; Score: <b>{{'%.2f'|format(rd.score)}}</b>{% endif %}</p>
{% if rd.models %}<table><tr><th>Model</th><th>Status</th><th>Score</th></tr>
{% for m in rd.models %}<tr><td>{{m.name}}</td><td>{{m.status}}</td><td>{{m.score}}</td></tr>{% endfor %}</table>{% endif %}
<p class="muted">{{rd.note}}</p>{% endif %}{% endif %}

{% if r.second_opinion.sightengine %}<h3 style="margin-top:16px">Sightengine <span class="muted" style="font-weight:400">(opt-in -- image sent to their cloud)</span></h3>
{% set se = r.second_opinion.sightengine %}
{% if se.error %}<p class="muted" style="color:var(--warn)">{{se.error}}</p>
{% elif not se.configured %}<p class="muted">{{se.note}}</p>
{% else %}<p>AI-generated confidence: <b>{{'%.2f'|format(se.ai_generated_score)}}</b></p>
<p class="muted">{{se.note}}</p>{% endif %}{% endif %}
</div>{% endif %}

{% if r.faces.faces %}<div class="card"><h2>Per-face results ({{r.faces.detector}} detector)</h2>
<p class="muted">{{r.faces.note}}</p>
{% if r.faces.detector == 'mediapipe' %}<table>
<tr><th>#</th><th>Sharpness</th><th>Noise</th><th>ELA</th><th>Roll</th><th>Eye region</th><th>Mouth region</th>
<th>EAR L/R</th><th>EAR asym</th><th>Symmetry dev</th><th>Score</th></tr>
{% for f in r.faces.faces %}<tr><td>{{loop.index}}</td><td>{{f.sharpness_ratio}}</td><td>{{f.noise_ratio}}</td>
<td>{{f.ela_ratio}}</td><td>{{f.roll_deg}}&deg;</td>
<td>{% if f.region_scores.left_eye %}ELA {{f.region_scores.left_eye.ela_ratio}} / noise {{f.region_scores.left_eye.noise_ratio}}{% else %}-{% endif %}</td>
<td>{% if f.region_scores.lips %}ELA {{f.region_scores.lips.ela_ratio}} / noise {{f.region_scores.lips.noise_ratio}}{% else %}-{% endif %}</td>
<td>{{f.ear_left}} / {{f.ear_right}}</td><td>{{f.ear_asymmetry}}</td><td>{{f.symmetry_dev}}</td>
<td><b>{{f.score}}</b></td></tr>{% endfor %}</table>
<p class="muted">Region ratios near 1.0 mean that region matches the rest of the same face. EAR asymmetry and
symmetry deviation are weak, supplementary signals -- a real turned or expressive face triggers them too.</p>
{% else %}<table>
<tr><th>#</th><th>Sharpness ratio</th><th>Noise ratio</th><th>ELA ratio</th><th>L/R asym</th><th>Score</th></tr>
{% for f in r.faces.faces %}<tr><td>{{loop.index}}</td><td>{{f.sharpness_ratio}}</td><td>{{f.noise_ratio}}</td>
<td>{{f.ela_ratio}}</td><td>{{f.lr_brightness_asym}}</td><td>{{f.score}}</td></tr>{% endfor %}</table>
<p class="muted">Ratios near 1.0 mean the face matches its surroundings. Install the MediaPipe face model
(get_facemesh.py) for rotation-aware detection and eye/mouth-level checks.</p>{% endif %}</div>{% endif %}

<div class="two"><div class="card"><h2>File &amp; hashes</h2><table>
<tr><td>Real type</td><td>{{r.structure.real_type}}</td></tr>
<tr><td>Dimensions</td><td>{{r.original_size[0]}} x {{r.original_size[1]}}</td></tr>
<tr><td>Size</td><td>{{r.hashes.size}} bytes</td></tr>
<tr><td>MD5</td><td>{{r.hashes.md5}}</td></tr><tr><td>SHA1</td><td>{{r.hashes.sha1}}</td></tr>
<tr><td>SHA256</td><td>{{r.hashes.sha256}}</td></tr>
{% for k,x in r.img_hashes.items() %}<tr><td>{{k}}</td><td>{{x}}</td></tr>{% endfor %}
<tr><td>Appended bytes</td><td>{{r.structure.appended_bytes}}</td></tr></table></div>
<div class="card"><h2>Module details</h2><table>
<tr><td>ELA hotspot ratio</td><td>{{r.ela.hotspot_ratio}}</td></tr>
<tr><td>Noise variation / mean</td><td>{{r.noise.variation}} / {{r.noise.mean_noise}}</td></tr>
<tr><td>FFT peaks / z</td><td>{{r.frequency.peak_count}} / {{r.frequency.peak_z}}</td></tr>
<tr><td>Cloned pairs</td><td>{{r.copy_move.matched_pairs}}</td></tr>
<tr><td>JPEG</td><td>{% if r.jpeg.applicable %}est. quality {{r.jpeg.estimated_quality}}, periodicity {{r.jpeg.periodicity}}{% else %}n/a{% endif %}</td></tr>
<tr><td>LSB ratios</td><td>{{r.lsb.lsb_ratios}}</td></tr>
<tr><td>LSB payload suspected</td><td>{{'YES' if r.lsb.suspicious else 'no'}}</td></tr></table></div></div>

<div class="card"><h2>C2PA Content Credentials</h2>
{% if not r.c2pa.available %}<p class="muted">Not installed. <code>pip install c2pa-python</code> to enable real signature verification (currently the engine cannot check for Content Credentials at all).</p>
{% elif not r.c2pa.has_manifest %}<p class="muted">{{r.c2pa.note}}</p>
{% else %}
<table>
<tr><td>Validation state</td><td>{{r.c2pa.validation_state}}</td></tr>
<tr><td>Signature valid (unmodified since signing)</td><td>{{'YES' if r.c2pa.signature_valid else 'NO - tampered after signing'}}</td></tr>
<tr><td>Signer trusted (chains to a trusted root)</td><td>{{'YES' if r.c2pa.trusted else 'no - untrusted/self-issued certificate'}}</td></tr>
<tr><td>Issuer</td><td>{{r.c2pa.issuer or '-'}}</td></tr>
<tr><td>Claim generator</td><td>{{r.c2pa.generator or '-'}}</td></tr>
<tr><td>Declared AI/algorithmic generation</td><td>{{'YES' if r.c2pa.ai_generated else 'no'}}</td></tr>
<tr><td>Ingredients (source assets)</td><td>{{r.c2pa.ingredients}}</td></tr>
</table>
{% if r.c2pa.actions %}<p class="muted">Declared actions: {% for a in r.c2pa.actions %}{{a.action}}{% if not loop.last %}, {% endif %}{% endfor %}</p>{% endif %}
<p class="muted">A missing manifest proves nothing -- most images never had one, or it was stripped on re-upload. Absence is not evidence of authenticity.</p>
{% endif %}
{% if r.c2pa.error %}<p class="muted" style="color:var(--warn)">{{r.c2pa.error}}</p>{% endif %}</div>

<div class="card"><h2>Metadata</h2>
{% if r.metadata.png_text %}<h3>PNG text chunks</h3><pre>{% for k,x in r.metadata.png_text.items() %}{{k}}: {{x}}
{% endfor %}</pre>{% endif %}
{% if r.metadata.exif %}<table>{% for k,x in r.metadata.exif.items() %}<tr><td>{{k}}</td><td>{{x}}</td></tr>{% endfor %}</table>
{% else %}<p class="muted">No EXIF data.</p>{% endif %}</div>

<div class="card"><h2>Visual clones</h2><div class="grid">
{% for name,fn in r.clones.items() %}<div><img src="{{url_for('case_file',case=case,path='clones/'+fn)}}" alt="{{name}}">
<b>{{name|replace('_',' ')}}</b> <a href="{{url_for('case_file',case=case,path='clones/'+fn)}}" download>download</a></div>{% endfor %}
</div></div>

<div class="card"><h2>CLI tool output</h2>
{% for name,text in r.tools.items() %}<h3>{{name}}</h3><pre>{{text}}</pre>{% endfor %}</div>
""" + LAYOUT_BOTTOM

HISTORY = LAYOUT_TOP + """<div class="card"><h1>Case History</h1>
{% if cases %}<table><tr><th>Time</th><th>Type</th><th>File</th><th>Verdict</th><th>Score</th><th></th></tr>
{% for c in cases %}<tr><td>{{c.time}}</td><td>{{'&#127909; video' if c.kind=='video' else '&#128247; image'}}</td><td>{{c.filename}}</td>
<td><span class="badge {{c.label}}">{{c.label}}</span></td><td>{{c.percent}}</td>
<td><a href="{{url_for('view_case',case=c.id)}}">open</a></td></tr>{% endfor %}</table>
{% else %}<p class="muted">No cases yet.</p>{% endif %}</div>""" + LAYOUT_BOTTOM

COMPARE = LAYOUT_TOP + """<div class="card"><h1>Compare two cases</h1>
<p class="muted">Perceptual-hash distance tells you whether two images are the same picture (resized, recompressed or lightly edited).</p>
<form method="GET"><select name="a" style="padding:10px">{% for c in cases %}<option value="{{c.id}}" {{'selected' if c.id==a}}>{{c.filename}} ({{c.id[:6]}})</option>{% endfor %}</select>
vs <select name="b" style="padding:10px">{% for c in cases %}<option value="{{c.id}}" {{'selected' if c.id==b}}>{{c.filename}} ({{c.id[:6]}})</option>{% endfor %}</select>
<button type="submit">Compare</button></form>
{% if result and result.error %}<p class="muted" style="color:var(--warn)">{{result.error}}</p>
{% elif result %}<table style="margin-top:16px"><tr><th>Hash</th><th>Hamming distance</th></tr>
{% for k,x in result.dist.items() %}<tr><td>{{k}}</td><td>{{x}} / 64</td></tr>{% endfor %}
<tr><td>Identical bytes (SHA256)</td><td>{{result.same}}</td></tr></table>
<p><b>{{result.verdict}}</b></p>{% endif %}</div>""" + LAYOUT_BOTTOM


MODELS = LAYOUT_TOP + """<div class="card"><h1>Local ML models</h1>
<p class="muted">Folder: {{mdir}}</p>
{% if models %}<table><tr><th>Name</th><th>Status</th><th>Backend</th><th>Labels</th><th>Counted as fake</th><th>Mode</th></tr>
{% for m in models %}<tr><td>{{m.name}}</td>
{% if m.ok %}<td style="color:var(--ok)">loaded</td><td>{{m.backend}}</td><td>{{m.labels}}</td><td>{{m.fake_labels}}</td><td>{{m.mode}}</td>
{% else %}<td style="color:var(--bad)">error</td><td colspan="4">{{m.error}}</td>{% endif %}</tr>{% endfor %}</table>
{% else %}<p>No models installed.</p>{% endif %}
<p><a class="btn" href="{{url_for('models_reload')}}">Rescan models</a></p></div>
<div class="card"><h2>Install a model</h2>
<pre>pip install -r requirements-ml.txt
python3 get_model.py --repo &lt;huggingface-repo-id&gt; --name mymodel
# then click "Rescan models"</pre>
<p class="muted">Or copy any ONNX classifier into models/&lt;name&gt;/model.onnx and add a classifier.json
(see models/README.txt). Scores are uncalibrated. Benchmark each model on your own real/fake samples first.</p></div>""" + LAYOUT_BOTTOM



VIDEO_REPORT = LAYOUT_TOP + """
<div class="card"><div class="two">
<div><h1>Video Forensic Report</h1>
<p class="muted">{{meta.filename}} &middot; case {{case}} &middot; {{meta.time}}</p>
{% if meta.note %}<p>Note: {{meta.note}}</p>{% endif %}
<span class="badge {{v.label}}">{{v.label}} RISK</span>
<p>{{v.description}}</p>
<a class="btn" href="{{url_for('case_json',case=case)}}">Download JSON</a></div>
<div><div class="big">{{v.percent}}<span style="font-size:1.2rem">/100</span></div>
<div class="muted">Video-specific indicator score (temporal signals only -- does not include the
per-image forensic checks used for still photos)</div>
<div class="bar"><div style="width:{{v.percent}}%"></div></div>
{% for k,s in v.module_scores.items() %}<div class="muted">{{k|title}}: {{'%.2f'|format(s)}}</div>
<div class="bar" style="height:5px"><div style="width:{{(s*100)|int}}%"></div></div>{% endfor %}</div></div></div>

{% if v.reasons %}<div class="card"><h2>Why this score</h2><ul>{% for x in v.reasons %}<li>{{x}}</li>{% endfor %}</ul></div>{% endif %}

{% if contribs is defined %}<div class="card"><h2>Score breakdown</h2>
<p class="muted">{% if v.calibration %}Calibrated model (trained on {{v.calibration.n_videos}} labelled videos).
Each signal's contribution below is measured against what was typical in that training data.
{% else %}Hand-tuned heuristic (no video calibration active). Each available signal contributes a fixed share of the total.{% endif %}</p>
{% if not self_check_ok %}<div class="errbox"><b>Internal consistency warning:</b> {{self_check_msg}}.
This breakdown may not exactly match the score above -- treat it as unreliable and report this as a bug.</div>{% endif %}
{% if contribs %}<table><tr><th>Signal</th><th>Score</th><th>Impact</th><th></th></tr>
{% for c in contribs %}<tr><td>{{c.label}}</td><td>{{'%.2f'|format(c.value)}}</td>
<td>{{'%+.3f'|format(c.contribution)}} <span class="muted">({{'toward fake' if c.direction=='up' else ('toward real' if c.direction=='down' else 'no effect')}})</span></td>
<td style="width:35%"><div class="cbar"><div style="width:{{(( c.contribution|abs / max_contrib)*100)|int}}%;
background:{{'var(--bad)' if c.direction=='up' else ('var(--ok)' if c.direction=='down' else 'var(--line)')}};
{% if c.direction=='down' %}margin-left:{{100-((c.contribution|abs / max_contrib)*100)|int}}%{% endif %}"></div></div></td>
</tr>{% endfor %}</table>
<p class="muted">Ranked by impact on the final score. Video signals (pulse, blink rate, face-region flicker) are
soft/weak by nature -- lighting, compression and ordinary motion all affect them. Treat this breakdown as
context for manual review, more than the image pipeline's equivalent card.</p>
{% else %}<p class="muted">No signal had a scored contribution for this video.</p>{% endif %}
{% if superseded %}<div class="warnbox"><b>Note:</b> the following finding(s) above under "Why this score" fired,
but did <b>not</b> determine this calibrated score -- the signal breakdown above did instead. Read these as
supporting context, not as what set the number:<ul>{% for s in superseded %}
<li>{{s.label}}: &ldquo;{{s.reason}}&rdquo;</li>{% endfor %}</ul></div>
{% elif not v.calibration and v.explanation.override_steps %}
{% set binding = v.explanation.override_steps|selectattr('binding')|list %}
{% if binding %}<div class="warnbox"><b>Note:</b> the weighted average above was overridden directly by:
<ul>{% for s in binding %}<li>raised from {{'%.2f'|format(s.pre_value)}} to {{'%.2f'|format(s.floor)}}
&mdash; &ldquo;{{s.reason}}&rdquo;</li>{% endfor %}</ul></div>{% endif %}{% endif %}
</div>{% endif %}

<div class="two">
<div class="card"><h2>Clip info</h2><table>
<tr><td>Resolution</td><td>{{r.info.width}} x {{r.info.height}}</td></tr>
<tr><td>FPS</td><td>{{r.info.fps}}</td></tr>
<tr><td>Frame count</td><td>{{r.info.frame_count}}</td></tr>
<tr><td>Duration</td><td>{{r.info.duration_s}}s</td></tr>
<tr><td>Frames sampled</td><td>{{r.n_frames_sampled}}</td></tr>
<tr><td>Frames with a tracked face</td><td>{{r.n_frames_tracked}}</td></tr>
<tr><td>Face detector used</td><td>{{r.detector or "none found"}}</td></tr>
</table></div>
<div class="card"><h2>Signal summary</h2><table>
<tr><td>rPPG pulse</td><td>{% if r.pulse.available %}periodicity {{r.pulse.periodicity}}, ~{{r.pulse.dominant_bpm}} bpm over {{r.pulse.duration_s}}s{% else %}{{r.pulse.note or "not available"}}{% endif %}</td></tr>
<tr><td>Blink rate</td><td>{% if r.blink.available %}{{r.blink.blinks}} blinks, {{r.blink.rate_per_min}}/min over {{r.blink.duration_s}}s{% else %}{{r.blink.note or "not available"}}{% endif %}</td></tr>
<tr><td>Face-region flicker</td><td>{% if r.flicker.available %}{{r.flicker.ratio}}x background volatility{% else %}{{r.flicker.note or "not available"}}{% endif %}</td></tr>
<tr><td>Per-frame ML</td><td>{% if r.ml.available %}mean {{r.ml.mean}}, max {{r.ml.max}}, std {{r.ml.std}} across {{r.ml.n_scored}} frames{% else %}no model installed{% endif %}</td></tr>
</table>
<p class="muted">Every signal here is heuristic and weak on its own -- lighting, compression, motion and normal
human variation can all trigger them. Treat this as a lead for manual review, not a verdict.</p></div>
</div>

{% if r.second_opinion %}<div class="card"><h2>File reputation</h2>
{% set vt = r.second_opinion.virustotal %}
<h3>VirusTotal <span class="muted" style="font-weight:400">(automatic -- checks the file's hash only, nothing uploaded)</span></h3>
{% if vt.error %}<p class="muted" style="color:var(--warn)">{{vt.error}}</p>
{% else %}{% if vt.found %}<p>Detections: <b>{{vt.malicious}}</b> malicious / {{vt.suspicious}} suspicious of {{vt.total_engines}} engines
&middot; <a href="{{vt.permalink}}" target="_blank" rel="noopener">view on VirusTotal</a></p>{% endif %}
<p class="muted">{{vt.note}}</p>{% endif %}</div>{% endif %}

{% if r.contact_sheet %}<div class="card"><h2>Sampled frames</h2>
<img src="{{url_for('case_file',case=case,path='video/'+r.contact_sheet)}}" style="width:100%;border-radius:8px">
<p class="muted">Box color: green = mediapipe detector, orange = Haar fallback.</p></div>{% endif %}

{% if r.pulse_plot %}<div class="card"><h2>Pulse signal (rPPG input)</h2>
<img src="{{url_for('case_file',case=case,path='video/'+r.pulse_plot)}}" style="width:100%;border-radius:8px"></div>{% endif %}

{% if r.ml_plot %}<div class="card"><h2>Per-frame ML score over time</h2>
<img src="{{url_for('case_file',case=case,path='video/'+r.ml_plot)}}" style="width:100%;border-radius:8px"></div>{% endif %}
""" + LAYOUT_BOTTOM

# --------------------------------------------------------------------- helpers
def render(tpl, **kw):
    return render_template_string(tpl, app=APP_NAME, ver=VERSION, dev=DEVELOPER, **kw)


def valid_case(case):
    if not CASE_RE.match(case) or not os.path.isdir(os.path.join(REPORTS, case)):
        abort(404)
    return os.path.join(REPORTS, case)


def load_case(case):
    d = valid_case(case)
    p = os.path.join(d, "report.json")
    if not os.path.exists(p):
        abort(404)
    with open(p) as f:
        return json.load(f)


def list_cases():
    out = []
    for cid in os.listdir(REPORTS):
        p = os.path.join(REPORTS, cid, "report.json")
        if CASE_RE.match(cid) and os.path.exists(p):
            try:
                d = json.load(open(p))
                out.append({"id": cid, "filename": d["meta"]["filename"], "time": d["meta"]["time"],
                            "kind": d["meta"].get("kind", "image"),
                            "label": d["result"]["verdict"]["label"],
                            "percent": d["result"]["verdict"]["percent"], "ts": d["meta"]["ts"]})
            except Exception:
                continue
    return sorted(out, key=lambda c: -c["ts"])


# ---------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render(INDEX, title="Analyze")


@app.route("/analyze", methods=["POST"])
def analyze_route():
    f = request.files.get("file") or request.files.get("image")  # "image" kept for old bookmarks/scripts
    if not f or not f.filename:
        return "No file uploaded", 400
    name = secure_filename(f.filename) or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext in ALLOWED_EXT:
        kind = "image"
    elif ext in VIDEO_EXT:
        kind = "video"
    else:
        return ("Invalid file type. Allowed images: " + ", ".join(sorted(ALLOWED_EXT)) +
                ". Allowed video: " + ", ".join(sorted(VIDEO_EXT))), 400

    case = uuid.uuid4().hex
    case_dir = os.path.join(REPORTS, case)
    os.makedirs(case_dir)
    file_path = os.path.join(case_dir, name)
    f.save(file_path)

    import shutil
    # The global MAX_CONTENT_LENGTH (200 MB) exists to give video room; images have no
    # reason to be that large and a huge "image" (valid header, bloated payload) would
    # still cost disk I/O and PIL parsing time before load_working_image() downsizes it.
    # Enforce the tighter, original per-type image cap explicitly, after saving (Flask's
    # global limit is the only thing enforced before the body is fully read).
    if kind == "image" and os.path.getsize(file_path) > IMAGE_MAX_BYTES:
        size_mb = os.path.getsize(file_path) // (1024 * 1024)
        shutil.rmtree(case_dir, ignore_errors=True)
        return f"Image too large ({size_mb} MB). Max {IMAGE_MAX_BYTES // (1024*1024)} MB for images.", 400
    if kind == "video" and os.path.getsize(file_path) > VIDEO_MAX_BYTES:
        size_mb = os.path.getsize(file_path) // (1024 * 1024)
        shutil.rmtree(case_dir, ignore_errors=True)
        return f"Video too large ({size_mb} MB). Max {VIDEO_MAX_BYTES // (1024*1024)} MB for video.", 400
    if kind == "image":
        try:
            from PIL import Image
            with Image.open(file_path) as im:
                im.verify()
        except Exception:
            shutil.rmtree(case_dir, ignore_errors=True)
            return "File is not a valid image", 400
        try:
            result = forensics.analyze(file_path, case_dir)
        except Exception as e:
            shutil.rmtree(case_dir, ignore_errors=True)
            return f"Analysis failed: {e}", 500
        # VirusTotal is a hash-only lookup (nothing uploaded), so -- like C2PA -- it runs
        # unconditionally, never gated behind the opt-in checkbox below.
        result["second_opinion"] = {"virustotal": second_opinion.check_virustotal(result["hashes"]["sha256"])}
        if request.form.get("external_check"):
            # Opt-in only: this is the one point in the whole app where the uploaded
            # file actually leaves the machine. See second_opinion.py's docstring.
            result["second_opinion"].update(second_opinion.run_all(file_path, kind="image"))
    else:
        try:
            result = video.analyze_video(file_path, case_dir)
        except Exception as e:
            shutil.rmtree(case_dir, ignore_errors=True)
            return f"Video analysis failed: {e}", 500
        if "error" in result:
            shutil.rmtree(case_dir, ignore_errors=True)
            return result["error"], 400
        result["second_opinion"] = {"virustotal": second_opinion.check_virustotal(
            forensics.compute_hashes(file_path)["sha256"])}

    note = request.form.get("note", "")[:500]
    meta = {"filename": name, "note": note, "ts": time.time(), "kind": kind,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "developer": DEVELOPER, "version": VERSION}
    with open(os.path.join(case_dir, "report.json"), "w") as fh:
        json.dump({"meta": meta, "result": result}, fh, indent=2)
    return redirect(url_for("view_case", case=case))


@app.route("/case/<case>")
def view_case(case):
    d = load_case(case)
    kind = d["meta"].get("kind", "image")  # older reports predate the kind field -> always image
    tpl = VIDEO_REPORT if kind == "video" else REPORT
    v = d["result"]["verdict"]
    explain_ctx = {}
    if v.get("explanation"):
        contribs = explain.ranked_contributions(v)
        explain_ctx = {"contribs": contribs,
                       "max_contrib": max((abs(c["contribution"]) for c in contribs), default=0) or 1.0,
                       "superseded": explain.superseded_overrides(v),
                       "self_check_ok": explain.self_check(v)[0], "self_check_msg": explain.self_check(v)[1]}
    return render(tpl, title="Report", case=case, meta=d["meta"], r=d["result"], v=v, **explain_ctx)


@app.route("/case/<case>/json")
def case_json(case):
    d = load_case(case)
    return Response(json.dumps(d, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": f"attachment; filename=report_{case[:8]}.json"})


@app.route("/files/<case>/<path:path>")
def case_file(case, path):
    d = valid_case(case)
    return send_from_directory(d, path)  # send_from_directory blocks traversal


@app.route("/history")
def history():
    return render(HISTORY, title="History", cases=list_cases())


@app.route("/models")
def models_page():
    return render(MODELS, title="ML Models", models=mlclassifier.list_models(), mdir=mlclassifier.MODELS_DIR)


@app.route("/models/reload")
def models_reload():
    mlclassifier.reload_models()
    return redirect(url_for("models_page"))


@app.route("/compare")
def compare():
    cases = [c for c in list_cases() if c["kind"] == "image"]  # perceptual-hash compare is image-only
    a, b = request.args.get("a"), request.args.get("b")
    result = None
    if a and b and CASE_RE.match(a) and CASE_RE.match(b):
        da, db = load_case(a), load_case(b)
        if da["meta"].get("kind", "image") != "image" or db["meta"].get("kind", "image") != "image":
            return render(COMPARE, title="Compare", cases=cases, a=a, b=b,
                          result={"error": "Perceptual-hash comparison only works between two images, not video."})
        ra, rb = da["result"], db["result"]
        dist = {k: imagehash.hex_to_hash(ra["img_hashes"][k]) - imagehash.hex_to_hash(rb["img_hashes"][k])
                for k in ra["img_hashes"]}
        avg = sum(dist.values()) / len(dist)
        verdict = ("Same image (resized/recompressed)" if avg <= 6 else
                   "Similar image (possibly edited version)" if avg <= 16 else "Different images")
        result = {"dist": dist, "same": ra["hashes"]["sha256"] == rb["hashes"]["sha256"], "verdict": verdict}
    return render(COMPARE, title="Compare", cases=cases, a=a, b=b, result=result)


if __name__ == "__main__":
    # Debug is OFF by default: Flask's debugger allows remote code execution.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1",
            host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5001)))
