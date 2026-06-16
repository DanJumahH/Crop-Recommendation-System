/* =====================================================
   CROP RECOMMENDATION SYSTEM — MAIN JS
   Group 18 | Kwara State University
   ===================================================== */

const DEFAULT_LAT = 9.0054;
const DEFAULT_LON = 38.7636;

let map, marker;
let envData = {};

// ── Init map ──────────────────────────────────────────
window.addEventListener('load', function () {
  map = L.map('map').setView([DEFAULT_LAT, DEFAULT_LON], 6);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  map.on('click', function (e) {
    placeMarker(e.latlng.lat, e.latlng.lng, null);
  });

  document.getElementById('addrInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); searchAddr(); }
  });
  document.getElementById('mapSearch').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); searchOnMap(); }
  });
});

// ── Tabs ──────────────────────────────────────────────
function switchTab(btn, tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
  if (tab === 'map' && map) setTimeout(() => map.invalidateSize(), 100);
}

// ── Place marker ──────────────────────────────────────
function placeMarker(lat, lon, name) {
  if (!map) return;
  if (marker) map.removeLayer(marker);

  const icon = L.divIcon({
    className: '',
    html: `<div style="
      width:26px;height:26px;
      background:#2D6A4F;
      border-radius:50% 50% 50% 0;
      transform:rotate(-45deg);
      border:3px solid white;
      box-shadow:0 2px 6px rgba(0,0,0,0.3);">
    </div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 26]
  });

  marker = L.marker([lat, lon], { icon, draggable: true }).addTo(map);
  marker.on('dragend', function () {
    const ll = marker.getLatLng();
    reverseGeocode(ll.lat, ll.lng);
  });

  map.setView([lat, lon], 11);
  setLocation(lat, lon, name);
}

// ── Geocode ───────────────────────────────────────────
async function geocode(query) {
  const res  = await fetch(
    `https://nominatim.openstreetmap.org/search` +
    `?q=${encodeURIComponent(query)}&format=json&limit=1`,
    { headers: { 'Accept-Language': 'en' } }
  );
  const data = await res.json();
  if (!data.length) throw new Error(
    'Location not found. Try adding "Ethiopia" e.g. "Jimma, Ethiopia"'
  );
  return {
    lat:  parseFloat(data[0].lat),
    lon:  parseFloat(data[0].lon),
    name: data[0].display_name.split(',').slice(0, 3).join(',')
  };
}

// ── Reverse geocode ───────────────────────────────────
async function reverseGeocode(lat, lon) {
  try {
    const res  = await fetch(
      `https://nominatim.openstreetmap.org/reverse` +
      `?lat=${lat}&lon=${lon}&format=json`,
      { headers: { 'Accept-Language': 'en' } }
    );
    const data = await res.json();
    const name = data.display_name
      ? data.display_name.split(',').slice(0, 3).join(',')
      : `Lat ${lat.toFixed(4)}, Lon ${lon.toFixed(4)}`;
    setLocation(lat, lon, name);
  } catch {
    setLocation(lat, lon, null);
  }
}

// ── Address search ────────────────────────────────────
async function searchAddr() {
  const q   = document.getElementById('addrInput').value.trim();
  if (!q) return;
  const btn = document.querySelector('#tab-addr .btn-action');
  btn.textContent = '...';
  btn.disabled    = true;
  try {
    const r = await geocode(q);
    if (map) placeMarker(r.lat, r.lon, r.name);
    else setLocation(r.lat, r.lon, r.name);
  } catch (e) { alert(e.message); }
  btn.textContent = 'Search';
  btn.disabled    = false;
}

// ── Map search ────────────────────────────────────────
async function searchOnMap() {
  const q   = document.getElementById('mapSearch').value.trim();
  if (!q) return;
  const btn = document.querySelector('#tab-map .btn-action');
  btn.textContent = '...';
  btn.disabled    = true;
  try {
    const r = await geocode(q);
    placeMarker(r.lat, r.lon, r.name);
  } catch (e) { alert(e.message); }
  btn.textContent = 'Search';
  btn.disabled    = false;
}

// ── Set location ──────────────────────────────────────
function setLocation(lat, lon, name) {
  document.getElementById('hLat').value     = lat;
  document.getElementById('hLon').value     = lon;
  document.getElementById('hLocName').value = name || `${lat.toFixed(4)}, ${lon.toFixed(4)}`;

  const locName   = name || `Lat ${parseFloat(lat).toFixed(4)}, Lon ${parseFloat(lon).toFixed(4)}`;
  const locCoords = `Lat: ${parseFloat(lat).toFixed(4)}, Lon: ${parseFloat(lon).toFixed(4)}`;
  document.getElementById('locName').textContent   = locName;
  document.getElementById('locCoords').textContent = locCoords;
  document.getElementById('locConfirmed').classList.remove('hidden');

  document.getElementById('step-ind-1').classList.add('done');
  document.getElementById('step-ind-2').classList.add('active');

  fetchEnvData(lat, lon);
}

// ── Fetch climate + topography ────────────────────────
async function fetchEnvData(lat, lon) {
  const afBar  = document.getElementById('afBar');
  const afDone = document.getElementById('afDone');

  afBar.classList.remove('hidden');
  afDone.classList.add('hidden');
  document.getElementById('climateSummary').classList.add('hidden');
  document.getElementById('afMsg').textContent =
    'Fetching climate & terrain data from Open-Meteo & NASADEM...';

  try {
    const res = await fetch('/get_env_data', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ lat, lon })
    });
    envData = await res.json();

    // Update topography fields
    if (envData.elevation) document.getElementById('fElev').value   = Math.round(envData.elevation);
    if (envData.slope)     document.getElementById('fSlope').value  = envData.slope;
    if (envData.aspect)    document.getElementById('fAspect').value = envData.aspect;
    if (envData.TWI)       document.getElementById('fTWI').value    = envData.TWI;

    // Show climate summary
    updateClimateSummary(envData);

    afBar.classList.add('hidden');
    afDone.classList.remove('hidden');

  } catch (e) {
    afBar.classList.add('hidden');
    document.getElementById('afMsg').textContent =
      '⚠️ Could not fetch data. Default values will be used.';
    afBar.classList.remove('hidden');
  }
}

// ── Update climate summary box ────────────────────────
function updateClimateSummary(data) {
  const avgTemp  = data['avg_temp']    || data['T2M_MAX-Su'] || null;
  const rain     = data['annual_prec'] || null;
  const humidity = data['QV2M-Su']     || null;
  const elevation= data['elevation']   || null;
  const slope    = data['slope']       || null;
  const cloud    = data['CLOUD_AMT']   || null;

  document.getElementById('cs-temp').textContent  =
    avgTemp   ? `${parseFloat(avgTemp).toFixed(1)}°C`    : '—';
  document.getElementById('cs-rain').textContent  =
    rain      ? `${parseFloat(rain).toFixed(0)} mm/yr`   : '—';
  document.getElementById('cs-hum').textContent   =
    humidity  ? `${parseFloat(humidity).toFixed(1)} g/kg`: '—';
  document.getElementById('cs-elev').textContent  =
    elevation ? `${Math.round(elevation)} m`              : '—';
  document.getElementById('cs-slope').textContent =
    slope     ? `${parseFloat(slope).toFixed(1)}°`        : '—';
  document.getElementById('cs-cloud').textContent =
    cloud     ? `${parseFloat(cloud).toFixed(0)}%`        : '—';

  document.getElementById('climateSummary').classList.remove('hidden');

  // Update seasonal climate in advanced panel
  if (data['T2M_MAX-W'])      document.getElementById('fT2M_MAX_W').value  = data['T2M_MAX-W'];
  if (data['T2M_MAX-Sp'])     document.getElementById('fT2M_MAX_Sp').value = data['T2M_MAX-Sp'];
  if (data['T2M_MAX-Su'])     document.getElementById('fT2M_MAX_Su').value = data['T2M_MAX-Su'];
  if (data['T2M_MAX-Au'])     document.getElementById('fT2M_MAX_Au').value = data['T2M_MAX-Au'];
  if (data['PRECTOTCORR-W'])  document.getElementById('fPREC_W').value     = data['PRECTOTCORR-W'];
  if (data['PRECTOTCORR-Sp']) document.getElementById('fPREC_Sp').value    = data['PRECTOTCORR-Sp'];
  if (data['PRECTOTCORR-Su']) document.getElementById('fPREC_Su').value    = data['PRECTOTCORR-Su'];
  if (data['PRECTOTCORR-Au']) document.getElementById('fPREC_Au').value    = data['PRECTOTCORR-Au'];
}

// ── Soil colour ───────────────────────────────────────
function selectSoil(card, name) {
  document.querySelectorAll('.soil-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');
  document.getElementById('hSoil').value = name;
  document.getElementById('soilErr').classList.add('hidden');
  document.getElementById('step-ind-2').classList.add('done');
  document.getElementById('step-ind-3').classList.add('active');
}

// ── Advanced toggle ───────────────────────────────────
function toggleAdv(btn) {
  const panel = document.getElementById('advPanel');
  panel.classList.toggle('open');
  btn.textContent = panel.classList.contains('open')
    ? 'Hide Advanced Parameters ▲'
    : 'Show Advanced Parameters ▼';
}

// ── Submit prediction ──────────────────────────────────
async function submitPrediction() {
  if (!document.getElementById('hLat').value) {
    showErr('Please select a farm location first.'); return;
  }
  if (!document.getElementById('hSoil').value) {
    document.getElementById('soilErr').classList.remove('hidden');
    document.getElementById('soilErr').scrollIntoView({ behavior: 'smooth' });
    return;
  }

  hideErr();
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('results').classList.add('hidden');
  document.querySelector('.btn-submit').disabled = true;

  const payload = {
    lat:           document.getElementById('hLat').value,
    lon:           document.getElementById('hLon').value,
    location_name: document.getElementById('hLocName').value,
    soil_color:    document.getElementById('hSoil').value,
    Ph:   document.getElementById('fPh').value,
    N:    document.getElementById('fN').value,
    P:    document.getElementById('fP').value,
    K:    document.getElementById('fK').value,
    Zn:   document.getElementById('fZn').value,
    S:    document.getElementById('fS').value,
    elevation: document.getElementById('fElev').value,
    slope:     document.getElementById('fSlope').value,
    aspect:    document.getElementById('fAspect').value,
    TWI:       document.getElementById('fTWI').value,
    ...envData
  };

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) showResults(data);
    else showErr(data.error || 'Prediction failed. Please try again.');
  } catch (e) {
    showErr('Connection error. Please try again.');
  }

  document.getElementById('loading').classList.add('hidden');
  document.querySelector('.btn-submit').disabled = false;
}

// ── Show results ──────────────────────────────────────
function showResults(data) {
  document.getElementById('resultsMeta').innerHTML =
    `Location: <span>${data.location}</span> &nbsp;|&nbsp; ` +
    `Elevation: <span>${data.elevation}m</span> &nbsp;|&nbsp; ` +
    `Avg Temp: <span>${data.avg_temp}°C</span> &nbsp;|&nbsp; ` +
    `Rainfall: <span>${data.annual_prec}mm/yr</span>`;

  const ranks = ['1st — Best Match', '2nd Choice', '3rd Choice'];

  // Normalize top 3 to sum to 100%
  const total = data.top3.reduce((sum, r) => sum + r.prob, 0);
  const normalized = data.top3.map(r => ({
    ...r,
    normProb: total > 0 ? Math.round((r.prob / total) * 1000) / 10 : r.prob
  }));

  document.getElementById('resultsGrid').innerHTML = normalized.map((r, i) => {
    const prob      = r.normProb;
    const confColor = prob >= 40 ? '#276749' : prob >= 25 ? '#D69E2E' : '#C53030';
    const confLabel = prob >= 40 ? 'Strongly suits your farm'
                    : prob >= 25 ? 'Moderately suits your farm'
                    : 'May suit your farm';
    return `
    <div class="result-card">
      <div class="r-rank">${ranks[i]}</div>
      <div class="r-emoji">${r.emoji}</div>
      <div class="r-crop">${r.crop}</div>
      <div class="r-prob" style="color:${confColor}">${prob}%</div>
      <div class="r-conf-label">Relative Suitability Score</div>
      <div class="r-suit-label" style="color:${confColor}">${confLabel}</div>
      <div class="conf-bar">
        <div class="conf-fill" style="width:${prob}%;background:${confColor}"></div>
      </div>
      <div class="r-season">Planting Season: ${r.season}</div>
      <div class="r-desc">${r.desc}</div>
    </div>`;
  }).join('');

  document.getElementById('results').classList.remove('hidden');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

// ── Error helpers ──────────────────────────────────────
function showErr(msg) {
  const el = document.getElementById('errBox');
  el.textContent = msg;
  el.classList.remove('hidden');
  el.scrollIntoView({ behavior: 'smooth' });
}
function hideErr() {
  document.getElementById('errBox').classList.add('hidden');
}
