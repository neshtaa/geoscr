"""
Lightweight Web Server and Interactive UI for GeoGuessr Locator
Runs natively using Python's standard http.server with zero external dependencies.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import urllib.request
import os
import tempfile
import base64
import math
from engine.rules_matcher import GeoKnowledgeBase
from engine.vlm_engine import analyze_image_with_gemini
from engine.offline_engine import OfflineGeoLocator, COUNTRY_CENTERS
from engine.geo_clusters import is_correct_or_neighbor, NEIGHBORS
from engine.deterministic_locator import DeterministicGeoLocator


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GeoGuessr Vision Locator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1120;
      --card: #1e293b;
      --accent: #10b981;
      --accent-hover: #059669;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
      --gold: #f59e0b;
      --tag-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 28px; }
    h1 { font-size: 2.2rem; font-weight: 800; color: #fff; display: flex; align-items: center; justify-content: center; gap: 12px; }
    h1 span { color: var(--accent); }
    p.subtitle { color: var(--muted); margin-top: 6px; font-size: 0.95rem; }
    
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
    
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    .card-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }
    
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 10px;
      padding: 28px 16px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
      background: #0f172a55;
    }
    .dropzone:hover, .dropzone.dragover { border-color: var(--accent); background: #10b98110; }
    .dropzone p { color: var(--muted); font-size: 0.95rem; }
    .dropzone span { color: var(--accent); font-weight: 600; }
    #preview-img { max-width: 100%; max-height: 240px; border-radius: 8px; margin-top: 14px; display: none; object-fit: contain; }

    .form-group { margin-top: 14px; }
    label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; }
    input[type="text"], input[type="password"] {
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 8px;
      color: #fff;
      font-size: 0.9rem;
    }
    input[type="text"]:focus, input[type="password"]:focus { outline: none; border-color: var(--accent); }

    .btn {
      width: 100%;
      background: var(--accent);
      color: #0b1120;
      font-weight: 800;
      padding: 12px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1rem;
      margin-top: 18px;
      transition: background 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn:hover { background: var(--accent-hover); color: #fff; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* Results */
    .engine-badge {
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 12px;
      background: #0f172a;
      color: var(--accent);
      border: 1px solid var(--accent);
    }
    .prediction-box {
      background: linear-gradient(135deg, #1e293b, #0f172a);
      border: 1px solid var(--accent);
      border-radius: 10px;
      padding: 18px;
      margin-bottom: 14px;
    }
    .pred-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .pred-country { font-size: 1.6rem; font-weight: 800; color: #fff; }
    .pred-conf { background: var(--accent); color: #0b1120; font-weight: 800; padding: 4px 10px; border-radius: 20px; font-size: 0.9rem; }
    .pred-region { color: var(--gold); font-weight: 600; font-size: 1.05rem; }

    /* Official metadata bar */
    .meta-bar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
      font-size: 0.8rem;
    }
    .meta-chip {
      background: var(--tag-bg);
      padding: 4px 8px;
      border-radius: 6px;
      color: var(--muted);
      border: 1px solid var(--border);
    }
    .meta-chip strong { color: var(--text); }

    .detected-params {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 12px;
    }
    .param-tag {
      font-size: 0.75rem;
      background: #33415555;
      padding: 3px 8px;
      border-radius: 4px;
      color: #94a3b8;
    }

    .alt-list { margin-top: 14px; }
    .alt-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      background: #0f172a;
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 0.88rem;
    }
    .alt-item span.prob { color: var(--accent); font-weight: 700; }

    .clues-list { margin-top: 12px; list-style: none; }
    .clues-list li {
      padding: 6px 0;
      color: #cbd5e1;
      font-size: 0.88rem;
      border-bottom: 1px solid #33415533;
      display: flex;
      gap: 8px;
    }
    .clues-list li::before { content: "✓"; color: var(--accent); font-weight: bold; }

    .reasoning-box {
      margin-top: 12px;
      padding: 12px;
      background: #0f172a;
      border-left: 3px solid var(--accent);
      border-radius: 4px;
      font-size: 0.88rem;
      color: #cbd5e1;
      line-height: 1.4;
    }

    .loader {
      display: none;
      text-align: center;
      padding: 40px 0;
    }
    .spinner {
      width: 40px; height: 40px;
      border: 4px solid #334155;
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 1s infinite linear;
      margin: 0 auto 12px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🌍 GeoGuessr <span>Vision Locator</span></h1>
      <p class="subtitle">Pixel-Level Computer Vision + Plonk It & GeoGuessr Knowledge Bases (136 Countries)</p>
    </header>

    <div class="grid">
      <!-- Upload Card -->
      <div class="card">
        <div class="card-title">📷 Panorama / Screenshot Input</div>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <p>Drag & drop panorama image here, paste <span>(Ctrl + V)</span>, or <span>Browse</span></p>
          <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFileSelect(event)">
          <img id="preview-img" alt="Preview">
        </div>

        <div class="form-group">
          <label>Or Image URL</label>
          <input type="text" id="url-input" placeholder="https://example.com/streetview.jpg">
        </div>

        <div class="form-group">
          <label>Observed Clues (Optional, e.g. 'snorkel, yellow plates')</label>
          <input type="text" id="clues-input" placeholder="e.g. red car, ladder pole, left drive">
        </div>

        <div class="form-group">
          <label>Gemini API Key (Optional — runs local Computer Vision if omitted)</label>
          <input type="password" id="api-key-input" placeholder="AIzaSy... (leave blank to run local computer vision)">
        </div>

        <button class="btn" id="predict-btn" onclick="predictLocation()">
          ⚡ Geolocate Panorama
        </button>
      </div>

      <!-- Results Card -->
      <div class="card">
        <div class="card-title">
          <span>🎯 Geolocation Results</span>
          <span class="engine-badge" id="engine-badge" style="display:none;"></span>
        </div>
        
        <div class="loader" id="loader">
          <div class="spinner"></div>
          <p style="color:var(--muted)">Inspecting image pixels, sky, road lines, soil & meta rules...</p>
        </div>

        <div id="results-content">
          <p style="color: var(--muted); text-align: center; padding: 60px 0;">
            Upload any GeoGuessr panorama or enter an image URL to see instant country, region, and Plonk It clues.
          </p>
        </div>
      </div>
    </div>
  </div>

  <script>
    let selectedFile = null;

    const dropzone = document.getElementById('dropzone');
    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        processFile(e.dataTransfer.files[0]);
      }
    });

    function handleFileSelect(e) {
      if (e.target.files.length) {
        processFile(e.target.files[0]);
      }
    }

    // Direct clipboard paste handler (Ctrl + V anywhere on page)
    window.addEventListener('paste', async (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            processFile(file);
            // Flash dropzone border green
            dropzone.style.borderColor = 'var(--accent)';
            setTimeout(() => { dropzone.style.borderColor = ''; }, 800);
            
            // Show toast notification
            const toast = document.createElement('div');
            toast.textContent = '📋 Screenshot pasted from clipboard! Analyzing...';
            toast.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#10b981;color:#0b1120;padding:10px 18px;border-radius:8px;font-weight:700;box-shadow:0 4px 12px rgba(0,0,0,0.5);z-index:9999;transition:opacity 0.3s;';
            document.body.appendChild(toast);
            setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 2500);

            // Auto-trigger prediction
            predictLocation();
            break;
          }
        }
      }
    });

    function processFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        const preview = document.getElementById('preview-img');
        preview.src = e.target.result;
        preview.style.display = 'block';
      };
      reader.readAsDataURL(file);
    }

    async function predictLocation() {
      const urlInput = document.getElementById('url-input').value.trim();
      const clues = document.getElementById('clues-input').value.trim();
      const apiKey = document.getElementById('api-key-input').value.trim();
      const loader = document.getElementById('loader');
      const results = document.getElementById('results-content');
      const btn = document.getElementById('predict-btn');
      const badge = document.getElementById('engine-badge');

      if (!selectedFile && !urlInput) {
        alert('Please upload an image or provide an image URL');
        return;
      }

      loader.style.display = 'block';
      results.style.display = 'none';
      badge.style.display = 'none';
      btn.disabled = true;

      try {
        let payload = { clues: clues, apiKey: apiKey };
        if (selectedFile) {
          payload.image_b64 = await toBase64(selectedFile);
        } else {
          payload.url = urlInput;
        }

        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        renderResults(data);
      } catch (err) {
        results.innerHTML = `<p style="color:#ef4444;text-align:center;">Error: ${err.message}</p>`;
        results.style.display = 'block';
      } finally {
        loader.style.display = 'none';
        results.style.display = 'block';
        btn.disabled = false;
      }
    }

    function toBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = error => reject(error);
      });
    }

    function renderResults(data) {
      const results = document.getElementById('results-content');
      const badge = document.getElementById('engine-badge');

      if (data.error) {
        results.innerHTML = `<p style="color:#ef4444;text-align:center;padding:20px;">${data.error}</p>`;
        return;
      }

      if (!data.top_prediction) {
        results.innerHTML = `<p style="color:#ef4444;text-align:center;">Could not determine location.</p>`;
        return;
      }

      if (data.engine_used) {
        badge.innerText = data.engine_used;
        badge.style.display = 'inline-block';
      }

      const top = data.top_prediction;
      
      // Metadata chips (from GeoGuessr Learning Hub)
      let metaChipsHtml = '';
      if (top.matched_features) {
        metaChipsHtml = top.matched_features.map(f => `<span class="meta-chip">${f}</span>`).join(' ');
      }

      // Detected pixel parameters
      let paramsHtml = '';
      if (data.detected_parameters && Object.keys(data.detected_parameters).length) {
        paramsHtml = `<div class="detected-params">
          ${Object.entries(data.detected_parameters).map(([k, v]) => `
            <span class="param-tag">${k.replace('_', ' ')}: <strong>${typeof v === 'object' ? JSON.stringify(v) : v}</strong></span>
          `).join('')}
        </div>`;
      }

      let altsHtml = '';
      if (data.alternative_candidates && data.alternative_candidates.length) {
        altsHtml = `<div class="alt-list">
          <label>Alternative Candidates</label>
          ${data.alternative_candidates.map(a => `
            <div class="alt-item">
              <span><strong>${a.country}</strong> (${a.country_code || ''}) - <em>${a.region || 'General'}</em></span>
              <span class="prob">${a.confidence_percent || 0}%</span>
            </div>
          `).join('')}
        </div>`;
      }

      let cluesHtml = '';
      if (data.identified_clues && data.identified_clues.length) {
        cluesHtml = `
          <label style="margin-top:16px;">Identified Visual Clues</label>
          <ul class="clues-list">
            ${data.identified_clues.map(c => `<li>${c}</li>`).join('')}
          </ul>
        `;
      }

      let officialCluesHtml = '';
      if (top.official_clues && top.official_clues.length) {
        officialCluesHtml = `
          <label style="margin-top:16px;color:#f59e0b;display:flex;align-items:center;gap:6px;">
            <span>🏆</span> Official GeoGuessr Post-Match Clues (${top.country})
          </label>
          <div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(210px, 1fr));gap:10px;margin-top:10px;">
            ${top.official_clues.map(c => `
              <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;overflow:hidden;padding:10px;display:flex;flex-direction:column;gap:6px;">
                ${c.image_url ? `<img src="${c.image_url}" style="width:100%;height:115px;object-fit:cover;border-radius:6px;" alt="${c.title}" onerror="this.style.display='none'">` : ''}
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span style="font-weight:700;font-size:0.85rem;color:#f8fafc;">${c.title}</span>
                  <span style="font-size:0.65rem;background:#1e293b;color:#10b981;padding:2px 6px;border-radius:4px;text-transform:uppercase;">${c.type || c.category || 'clue'}</span>
                </div>
                <p style="font-size:0.75rem;color:#94a3b8;line-height:1.3;">${c.description}</p>
              </div>
            `).join('')}
          </div>
        `;
      }

      let reasoningHtml = '';
      if (data.plonkit_meta_reasoning) {
        reasoningHtml = `
          <label style="margin-top:16px;">Plonk It & GeoGuessr Deduction</label>
          <div class="reasoning-box">${data.plonkit_meta_reasoning}</div>
        `;
      }

      results.innerHTML = `
        <div class="prediction-box">
          <div class="pred-header">
            <span class="pred-country">${top.country} ${top.country_code ? `(${top.country_code})` : ''}</span>
            <span class="pred-conf">${top.confidence_percent}%</span>
          </div>
          <div class="pred-region">📍 ${top.region || 'General Region'}</div>
          ${top.gps_estimate ? `<div style="font-size:0.8rem; color:var(--muted); margin-top:4px;">GPS: ${top.gps_estimate.lat}, ${top.gps_estimate.lng}</div>` : ''}
          <div class="meta-bar">${metaChipsHtml}</div>
          ${paramsHtml}
        </div>
        ${altsHtml}
        ${officialCluesHtml}
        ${cluesHtml}
        ${reasoningHtml}
      `;
    }
  </script>
</body>
</html>
"""

def reverse_geocode(lat, lng):
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lng}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "GeoScript-Evaluator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            addr = data.get("address", {})
            return {
                "country_code": addr.get("country_code", "").upper(),
                "country": addr.get("country", ""),
                "state": addr.get("state", "") or addr.get("region", "")
            }
    except Exception:
        min_d = float("inf")
        best = "US"
        for cc, (clat, clng) in COUNTRY_CENTERS.items():
            d = (clat - lat)**2 + (clng - lng)**2
            if d < min_d:
                min_d = d
                best = cc
        return {"country_code": best, "country": best, "state": ""}

BENCHMARK_ROUNDS = [
    {
        "round": 1,
        "true_country": "Senegal",
        "true_code": "SN",
        "lat": 15.7495,
        "lng": -13.4062,
        "north_image": "/scratch/round_1_8_north.jpg",
        "south_image": "/scratch/round_1_8_south.jpg",
        "north_file": "scratch/round_1_8_north.jpg",
        "south_file": "scratch/round_1_8_south.jpg"
    },
    {
        "round": 2,
        "true_country": "Denmark",
        "true_code": "DK",
        "lat": 55.9996,
        "lng": 10.0082,
        "north_image": "/scratch/round_1_2_north.jpg",
        "south_image": "/scratch/round_1_2_south.jpg",
        "north_file": "scratch/round_1_2_north.jpg",
        "south_file": "scratch/round_1_2_south.jpg"
    },
    {
        "round": 3,
        "true_country": "Kenya",
        "true_code": "KE",
        "lat": -4.2383,
        "lng": 39.4155,
        "north_image": "/scratch/round_1_4_north.jpg",
        "south_image": "/scratch/round_1_4_south.jpg",
        "north_file": "scratch/round_1_4_north.jpg",
        "south_file": "scratch/round_1_4_south.jpg"
    },
    {
        "round": 4,
        "true_country": "Philippines",
        "true_code": "PH",
        "lat": 10.2883,
        "lng": 123.6944,
        "north_image": "/scratch/round_1_5_north.jpg",
        "south_image": "/scratch/round_1_5_south.jpg",
        "north_file": "scratch/round_1_5_north.jpg",
        "south_file": "scratch/round_1_5_south.jpg"
    },
    {
        "round": 5,
        "true_country": "Estonia",
        "true_code": "EE",
        "lat": 58.2654,
        "lng": 22.5124,
        "north_image": "/scratch/round_1_6_north.jpg",
        "south_image": "/scratch/round_1_6_south.jpg",
        "north_file": "scratch/round_1_6_north.jpg",
        "south_file": "scratch/round_1_6_south.jpg"
    },
    {
        "round": 6,
        "true_country": "Mongolia",
        "true_code": "MN",
        "lat": 48.6255,
        "lng": 97.6180,
        "north_image": "/scratch/round_1_7_north.jpg",
        "south_image": "/scratch/round_1_7_south.jpg",
        "north_file": "scratch/round_1_7_north.jpg",
        "south_file": "scratch/round_1_7_south.jpg"
    },
    {
        "round": 7,
        "true_country": "Malta",
        "true_code": "MT",
        "lat": 35.8947,
        "lng": 14.5067,
        "north_image": "/scratch/round_1_9_north.jpg",
        "south_image": "/scratch/round_1_9_south.jpg",
        "north_file": "scratch/round_1_9_north.jpg",
        "south_file": "scratch/round_1_9_south.jpg"
    },
    {
        "round": 8,
        "true_country": "Poland",
        "true_code": "PL",
        "lat": 53.1300,
        "lng": 17.9828,
        "north_image": "/scratch/round_1_10_north.jpg",
        "south_image": "/scratch/round_1_10_south.jpg",
        "north_file": "scratch/round_1_10_north.jpg",
        "south_file": "scratch/round_1_10_south.jpg"
    },
    {
        "round": 9,
        "true_country": "Argentina",
        "true_code": "AR",
        "lat": -51.4339,
        "lng": -69.5721,
        "north_image": "/scratch/round_3_8_north.jpg",
        "south_image": "/scratch/round_3_8_south.jpg",
        "north_file": "scratch/round_3_8_north.jpg",
        "south_file": "scratch/round_3_8_south.jpg"
    },
    {
        "round": 10,
        "true_country": "Germany",
        "true_code": "DE",
        "lat": 48.1783,
        "lng": 11.3946,
        "north_image": "/scratch/round_3_2_north.jpg",
        "south_image": "/scratch/round_3_2_south.jpg",
        "north_file": "scratch/round_3_2_north.jpg",
        "south_file": "scratch/round_3_2_south.jpg"
    }
]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

class GeoRequestHandler(BaseHTTPRequestHandler):
    kb = GeoKnowledgeBase()
    offline = OfflineGeoLocator()
    deterministic_locator = DeterministicGeoLocator()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        if path == "/calibration" or path == "/calibration.html":
            cal_path = os.path.join(os.path.dirname(__file__), "calibration_dashboard.html")
            if os.path.exists(cal_path):
                with open(cal_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
                return

        if path.startswith("/scratch/"):
            fname = os.path.basename(path)
            fpath = os.path.join("scratch", fname)
            if os.path.exists(fpath):
                self.send_response(200)
                if fname.endswith(".png"):
                    self.send_header("Content-Type", "image/png")
                else:
                    self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
                return

        if path == "/api/calibration/rounds":
            rounds_summary = [
                {
                    "round": r["round"],
                    "true_country": r["true_country"],
                    "true_code": r["true_code"],
                    "lat": r["lat"],
                    "lng": r["lng"],
                    "north_image": r["north_image"],
                    "south_image": r["south_image"]
                }
                for r in BENCHMARK_ROUNDS
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(rounds_summary).encode("utf-8"))
            return

        if path == "/api/countries":
            countries = [{"title": v["title"], "code": k, "slug": v["slug"]} for k, v in self.kb.kb.items()]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(countries).encode("utf-8"))
            return

        if path == "/api/search":
            q = query.get("q", [""])[0]
            results = self.kb.search_clues(q, top_k=10)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(results).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/evaluate_round":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            lat = float(data.get("lat", 0))
            lng = float(data.get("lng", 0))
            pred_code = data.get("pred_code", "")

            geo = reverse_geocode(lat, lng)
            true_code = geo["country_code"]
            is_match, match_type = is_correct_or_neighbor(true_code, pred_code)

            result = {
                "is_match": is_match,
                "match_type": match_type,
                "true_code": true_code,
                "true_country": geo["country"],
                "true_state": geo["state"],
                "pred_code": pred_code
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        if self.path == "/api/calibration/analyze":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            round_id = int(data.get("round_id", 1))
            matched_r = next((r for r in BENCHMARK_ROUNDS if r["round"] == round_id), None)
            if not matched_r:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error": "Round not found"}')
                return

            features = self.deterministic_locator.extract_features(matched_r["north_file"], matched_r["south_file"])
            pred = self.deterministic_locator.predict(features)
            top = pred["top_prediction"]
            pred_code = top["country_code"]
            true_code = matched_r["true_code"]

            dist_km = haversine_km(
                matched_r["lat"], matched_r["lng"],
                top["gps_estimate"]["lat"], top["gps_estimate"]["lng"]
            )

            is_exact = (pred_code == true_code)
            is_neighbor = False
            match_type = "EXACT" if is_exact else "MISMATCH"
            if not is_exact:
                borders = NEIGHBORS.get(true_code, [])
                if pred_code in borders:
                    is_neighbor = True
                    match_type = f"IMMEDIATE_NEIGHBOR ({pred_code} borders {true_code})"

            response_payload = {
                **pred,
                "evaluation": {
                    "is_exact": is_exact,
                    "is_neighbor": is_neighbor,
                    "match_type": match_type,
                    "distance_km": dist_km
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))
            return

        if self.path == "/api/predict":

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid JSON"}')
                return

            img_b64 = data.get("image_b64")
            img_south_b64 = data.get("image_south_b64")
            url = data.get("url")
            clues = data.get("clues", "")
            api_key = data.get("apiKey", "") or os.environ.get("GEMINI_API_KEY")

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name

            tmp_south_path = None
            if img_south_b64:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_s:
                    tmp_south_path = tmp_s.name
                    tmp_s.write(base64.b64decode(img_south_b64))

            try:
                if img_b64:
                    with open(tmp_path, "wb") as f:
                        f.write(base64.b64decode(img_b64))
                elif url:
                    urllib.request.urlretrieve(url, tmp_path)
                else:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "No image or URL provided"}')
                    return

                # Pure Deterministic Mathematical Computer Vision & Clue Database Matching (0% AI / LLM)
                features = self.deterministic_locator.extract_features(tmp_path, tmp_south_path or tmp_path)
                res = self.deterministic_locator.predict(features)
                top_code = res["top_prediction"]["country_code"]
                res["top_prediction"]["official_clues"] = self.deterministic_locator.master_clues.get(top_code, [])[:3]
                res["engine_used"] = "⚡ Pure Deterministic CV + GeoGuessr Clue Engine (0% AI/LLM)"

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if tmp_south_path and os.path.exists(tmp_south_path):
                    os.remove(tmp_south_path)
            return


        self.send_response(404)
        self.end_headers()

def run_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), GeoRequestHandler)
    print(f"GeoGuessr Locator Server running at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    run_server()
