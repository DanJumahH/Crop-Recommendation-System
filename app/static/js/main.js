// ── Soil definitions ──────────────────────────────────────────────────
const SOILS = [
  {name:'black',      label:'Black',       hex:'#1C1C1C', ph:7.2,N:1.8,P:18,K:320,Zn:1.2,S:12},
  {name:'brown',      label:'Brown',       hex:'#7B4F2E', ph:6.5,N:1.2,P:22,K:280,Zn:0.9,S:10},
  {name:'dark brown', label:'Dark Brown',  hex:'#4A2C1A', ph:6.2,N:1.5,P:20,K:300,Zn:1.0,S:11},
  {name:'dark gray',  label:'Dark Gray',   hex:'#4A4A4A', ph:7.0,N:1.0,P:15,K:260,Zn:0.7,S:8 },
  {name:'gray',       label:'Gray',        hex:'#9A9A8A', ph:6.8,N:0.9,P:14,K:240,Zn:0.6,S:7 },
  {name:'red',        label:'Red',         hex:'#A83232', ph:5.2,N:0.6,P:8, K:160,Zn:0.4,S:5 },
  {name:'reddish brown',label:'Reddish Brown',hex:'#8B3A2A',ph:5.8,N:0.7,P:10,K:175,Zn:0.5,S:6},
  {name:'yellow',     label:'Yellow',      hex:'#C9A84C', ph:6.0,N:0.5,P:9, K:150,Zn:0.3,S:4 },
];

// ── State ─────────────────────────────────────────────────────────────
let lat=null, lon=null, envData={}, selectedSoil=null, map, marker;

// ── Keep-alive ping every 10 min ──────────────────────────────────────
setInterval(()=>fetch('/ping'),600000);

// ── Map init ──────────────────────────────────────────────────────────
function initMap(){
  map = L.map('map').setView([9.0,38.7],5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
    attribution:'© OpenStreetMap contributors', maxZoom:18
  }).addTo(map);

  const icon = L.divIcon({
    html:'<div style="width:14px;height:14px;background:#6B7C3F;border:2.5px solid #fff;border-radius:50%;box-shadow:0 1px 4px rgba(0,0,0,.3)"></div>',
    iconSize:[14,14],iconAnchor:[7,7],className:''
  });

  map.on('click',e=>pinLocation(e.latlng.lat,e.latlng.lng,icon));

  // Address search
  document.getElementById('search-btn').addEventListener('click',()=>{
    const q = document.getElementById('address-input').value.trim();
    if(!q) return;
    fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}`)
      .then(r=>r.json()).then(res=>{
        if(!res.length){alert('Location not found. Try a different search term.');return;}
        const r0 = res[0];
        map.setView([r0.lat,r0.lon],10);
        pinLocation(parseFloat(r0.lat),parseFloat(r0.lon),icon);
      }).catch(()=>alert('Search failed — try clicking the map instead.'));
  });
  document.getElementById('address-input').addEventListener('keydown',e=>{
    if(e.key==='Enter') document.getElementById('search-btn').click();
  });

  // Geolocation
  document.getElementById('locate-btn').addEventListener('click',()=>{
    if(!navigator.geolocation){alert('Geolocation not supported by your browser.');return;}
    navigator.geolocation.getCurrentPosition(
      pos=>{ map.setView([pos.coords.latitude,pos.coords.longitude],12);
             pinLocation(pos.coords.latitude,pos.coords.longitude,icon); },
      ()=>alert('Could not get your location. Please allow location access.')
    );
  });
}

async function pinLocation(la,lo,icon){
  lat=parseFloat(la.toFixed(6));
  lon=parseFloat(lo.toFixed(6));
  if(marker) map.removeLayer(marker);
  marker = L.marker([lat,lon],{icon,draggable:true}).addTo(map);
  marker.on('dragend',e=>{
    const p=e.target.getLatLng();
    pinLocation(p.lat,p.lng,icon);
  });
  document.getElementById('status-text').textContent=`Pinned: ${lat}, ${lon}`;
  document.getElementById('fetch-indicator').hidden=false;
  document.getElementById('fetch-done').hidden=true;
  document.getElementById('env-chips').hidden=true;
  await fetchEnvData(lat,lon);
}

async function fetchEnvData(la,lo){
  try{
    const res  = await fetch('/get_env_data',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat:la,lon:lo})});
    envData    = await res.json();
    fillClimateFields(envData);
    setField('elevation',envData.elevation); setField('slope',envData.slope);
    setField('aspect',envData.aspect);       setField('TWI',envData.TWI);
    document.getElementById('fetch-indicator').hidden=true;
    document.getElementById('fetch-done').hidden=false;
    document.getElementById('env-chips').hidden=false;
    document.getElementById('chip-loc').textContent=`📍 ${la.toFixed(4)}, ${lo.toFixed(4)}`;
    document.getElementById('chip-elev').textContent=`⛰ ${envData.elevation}m`;
    document.getElementById('chip-temp').textContent=`🌡 ${envData.avg_temp}°C avg`;
    document.getElementById('chip-rain').textContent=`🌧 ${envData.annual_prec}mm/yr`;
  }catch(e){
    document.getElementById('fetch-indicator').hidden=true;
    document.getElementById('status-text').textContent='Could not fetch data — fill climate fields manually.';
  }
}

function fillClimateFields(d){
  const map={
    'T2M_MAX-W':d['T2M_MAX-W'],'T2M_MIN-W':d['T2M_MIN-W'],'PRECTOTCORR-W':d['PRECTOTCORR-W'],'QV2M-W':d['QV2M-W'],
    'T2M_MAX-Sp':d['T2M_MAX-Sp'],'T2M_MIN-Sp':d['T2M_MIN-Sp'],'PRECTOTCORR-Sp':d['PRECTOTCORR-Sp'],'QV2M-Sp':d['QV2M-Sp'],
    'T2M_MAX-Su':d['T2M_MAX-Su'],'T2M_MIN-Su':d['T2M_MIN-Su'],'PRECTOTCORR-Su':d['PRECTOTCORR-Su'],'QV2M-Su':d['QV2M-Su'],
    'T2M_MAX-Au':d['T2M_MAX-Au'],'T2M_MIN-Au':d['T2M_MIN-Au'],'PRECTOTCORR-Au':d['PRECTOTCORR-Au'],'QV2M-Au':d['QV2M-Au'],
    'WD10M':d['WD10M'],'WS2M_RANGE':d['WS2M_RANGE'],'GWETTOP':d['GWETTOP'],'CLOUD_AMT':d['CLOUD_AMT'],'PS':d['PS'],
  };
  Object.entries(map).forEach(([id,val])=>setField(id,val,true));
}

function setField(id,val,autofill=false){
  const el=document.getElementById(id);
  if(!el) return;
  el.value=val;
  if(autofill) el.classList.add('autofilled'); else el.classList.remove('autofilled');
}

// ── Soil grid ─────────────────────────────────────────────────────────
function initSoilGrid(){
  const grid=document.getElementById('soil-grid');
  SOILS.forEach(s=>{
    const card=document.createElement('div');
    card.className='soil-card';
    card.innerHTML=`<div class="soil-swatch" style="background:${s.hex}"></div>
      <div class="soil-name">${s.label}</div>
      <div class="soil-npk">pH ${s.ph} · N ${s.N} · P ${s.P}</div>`;
    card.addEventListener('click',()=>selectSoil(s,card));
    grid.appendChild(card);
  });
}

function selectSoil(s,card){
  document.querySelectorAll('.soil-card').forEach(c=>c.classList.remove('selected'));
  card.classList.add('selected');
  selectedSoil=s.name;
  document.getElementById('soil-color-input').value=s.name;
  setField('Ph',s.ph); setField('N',s.N); setField('P',s.P);
  setField('K',s.K);   setField('Zn',s.Zn); setField('S',s.S);
}

// ── Collapsible panels ────────────────────────────────────────────────
function initToggles(){
  [['climate-toggle','climate-fields'],['terrain-toggle','terrain-fields']].forEach(([btnId,panelId])=>{
    const btn=document.getElementById(btnId);
    const panel=document.getElementById(panelId);
    if(!btn||!panel) return;
    btn.addEventListener('click',()=>{
      const open=!panel.hidden;
      panel.hidden=open;
      btn.textContent=open?'Show ⌄':'Hide ⌃';
    });
  });
}

// ── Submit ────────────────────────────────────────────────────────────
document.getElementById('submit-btn').addEventListener('click',async()=>{
  const warn=document.getElementById('submit-warn');
  if(!lat){warn.hidden=false;warn.textContent='Please pin a location on the map first.';return;}
  if(!selectedSoil){warn.hidden=false;warn.textContent='Please select a soil type.';return;}
  warn.hidden=true;

  const btn=document.getElementById('submit-btn');
  btn.querySelector('.btn-text').hidden=true;
  btn.querySelector('.btn-loading').hidden=false;
  btn.disabled=true;
  document.getElementById('results').hidden=true;
  document.getElementById('error-box').hidden=true;

  const keys=['Ph','N','P','K','Zn','S',
    'T2M_MAX-W','T2M_MIN-W','PRECTOTCORR-W','QV2M-W',
    'T2M_MAX-Sp','T2M_MIN-Sp','PRECTOTCORR-Sp','QV2M-Sp',
    'T2M_MAX-Su','T2M_MIN-Su','PRECTOTCORR-Su','QV2M-Su',
    'T2M_MAX-Au','T2M_MIN-Au','PRECTOTCORR-Au','QV2M-Au',
    'WD10M','WS2M_RANGE','GWETTOP','CLOUD_AMT','PS',
    'elevation','slope','aspect','TWI'];
  const payload={soil_color:selectedSoil,
    location_name:document.getElementById('address-input').value||`${lat}, ${lon}`,
    avg_temp:envData.avg_temp||22.1, annual_prec:envData.annual_prec||850};
  keys.forEach(k=>{const el=document.getElementById(k);payload[k]=el?parseFloat(el.value)||0:0;});

  try{
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await res.json();
    if(!data.success) throw new Error(data.error);
    renderResults(data);
  }catch(e){
    document.getElementById('error-msg').textContent='Something went wrong: '+e.message;
    document.getElementById('error-box').hidden=false;
  }finally{
    btn.querySelector('.btn-text').hidden=false;
    btn.querySelector('.btn-loading').hidden=true;
    btn.disabled=false;
  }
});

function renderResults(data){
  document.getElementById('results-location').innerHTML=
    `📍 ${data.location} &nbsp;·&nbsp; ⛰ ${data.elevation}m &nbsp;·&nbsp; 🌡 ${data.avg_temp}°C &nbsp;·&nbsp; 🌧 ${data.annual_prec}mm/yr`;
  const cards=document.getElementById('results-cards');
  cards.innerHTML='';
  const ranks=['1st choice','2nd choice','3rd choice'];
  data.top3.forEach((r,i)=>{
    const card=document.createElement('div');
    card.className=`result-card rank-${i+1}`;
    card.innerHTML=`
      <div class="result-rank">${ranks[i]}</div>
      <div class="result-emoji">${r.emoji}</div>
      <div class="result-crop">${r.crop}</div>
      <div class="result-season">Best season: ${r.season}</div>
      <div class="result-desc">${r.desc}</div>
      <div class="result-prob">${r.prob}% confidence</div>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>`;
    cards.appendChild(card);
    setTimeout(()=>card.querySelector('.bar-fill').style.width=r.prob+'%',80*(i+1));
  });
  document.getElementById('results').hidden=false;
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});
}

document.getElementById('reset-btn').addEventListener('click',()=>{
  document.getElementById('results').hidden=true;
  window.scrollTo({top:0,behavior:'smooth'});
});

// ── Boot ──────────────────────────────────────────────────────────────
initMap();
initSoilGrid();
initToggles();
