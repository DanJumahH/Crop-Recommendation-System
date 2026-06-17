"""
CropIQ — Crop Recommendation System
Group 18 | Kwara State University
"""
from flask import Flask, render_template, request, jsonify
import pickle, os, numpy as np, pandas as pd, requests

app = Flask(__name__)

BASE              = os.path.join(os.path.dirname(__file__), 'model')
xgb               = pickle.load(open(os.path.join(BASE, 'xgb_model.pkl'),        'rb'))
scaler            = pickle.load(open(os.path.join(BASE, 'scaler.pkl'),            'rb'))
le_label          = pickle.load(open(os.path.join(BASE, 'le_label.pkl'),          'rb'))
feat_cols         = pickle.load(open(os.path.join(BASE, 'feat_cols.pkl'),         'rb'))
numeric_feat_cols = pickle.load(open(os.path.join(BASE, 'numeric_feat_cols.pkl'), 'rb'))
onehot_feat_cols  = pickle.load(open(os.path.join(BASE, 'onehot_feat_cols.pkl'),  'rb'))
soil_cols         = pickle.load(open(os.path.join(BASE, 'soil_cols.pkl'),         'rb'))

CROP_INFO = {
    "Teff":       {"emoji":"🌾","season":"Mar – Sep","desc":"Staple Ethiopian grain, highly nutritious and drought-tolerant. Grows best in loamy soils with moderate rainfall."},
    "Maize":      {"emoji":"🌽","season":"Apr – Oct","desc":"High-yield cereal suited to warm climates with moderate to high rainfall and well-drained fertile soils."},
    "Wheat":      {"emoji":"🌿","season":"Nov – Apr","desc":"Cool-season cereal widely grown in Ethiopian highlands. Prefers well-drained clay-loam soils."},
    "Barley":     {"emoji":"🌱","season":"Oct – Mar","desc":"Hardy cereal tolerant of poor soils and cold highland conditions. Adaptable to a wide pH range."},
    "Bean":       {"emoji":"🫘","season":"Sep – Jan","desc":"Legume that fixes nitrogen, improving soil fertility. Grows well in well-drained loam with moderate rainfall."},
    "Pea":        {"emoji":"🟢","season":"Sep – Jan","desc":"Cool-season legume suited to well-drained highland soils with low to moderate rainfall."},
    "Sorghum":    {"emoji":"🌾","season":"Apr – Oct","desc":"Drought-resistant cereal ideal for semi-arid lowlands with sandy to loamy soils."},
    "Dagussa":    {"emoji":"🌾","season":"Jun – Nov","desc":"Ethiopian finger millet, resilient in acidic and marginal soils with low rainfall."},
    "Niger seed": {"emoji":"🌻","season":"Jun – Oct","desc":"Oilseed crop tolerant of waterlogged or acidic soils. Grows well in clay-rich highland areas."},
    "Potato":     {"emoji":"🥔","season":"Oct – Feb","desc":"Tuber crop suited to cool highland conditions with well-drained, loose, fertile soils."},
    "Red Pepper": {"emoji":"🌶️","season":"Mar – Aug","desc":"Spice crop requiring warm temperatures, good drainage, and fertile loamy soils."},
}

def get_soil_onehot(soil_color):
    return {col: (1 if col.replace('soil_','') == soil_color else 0) for col in soil_cols}

def engineer_features(row):
    row['K_P_ratio']        = row['K']  / (row['P']  + 1e-6)
    row['N_K_ratio']        = row['N']  / (row['K']  + 1e-6)
    row['NPK_sum']          = row['N']  +  row['P']  + row['K']
    row['Ph_N_inter']       = row['Ph'] *  row['N']
    row['T_range_W']        = row['T2M_MAX-W']  - row['T2M_MIN-W']
    row['T_range_Sp']       = row['T2M_MAX-Sp'] - row['T2M_MIN-Sp']
    row['T_range_Su']       = row['T2M_MAX-Su'] - row['T2M_MIN-Su']
    row['T_range_Au']       = row['T2M_MAX-Au'] - row['T2M_MIN-Au']
    row['T_MAX_annual']     = np.mean([row['T2M_MAX-W'],row['T2M_MAX-Sp'],row['T2M_MAX-Su'],row['T2M_MAX-Au']])
    row['T_MIN_annual']     = np.mean([row['T2M_MIN-W'],row['T2M_MIN-Sp'],row['T2M_MIN-Su'],row['T2M_MIN-Au']])
    row['PREC_annual']      = row['PRECTOTCORR-W']+row['PRECTOTCORR-Sp']+row['PRECTOTCORR-Su']+row['PRECTOTCORR-Au']
    row['QV2M_annual']      = np.mean([row['QV2M-W'],row['QV2M-Sp'],row['QV2M-Su'],row['QV2M-Au']])
    row['wet_ratio']        = (row['PRECTOTCORR-Su']+row['PRECTOTCORR-Sp'])/(row['PREC_annual']+1e-6)
    row['elev_temp_inter']  = row['elevation'] * row['T_MAX_annual']
    row['slope_rain_inter'] = row['slope']     * row['PREC_annual']
    return row

def get_climate(lat, lon):
    try:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}"
               "&start_date=2023-01-01&end_date=2023-12-31"
               "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
               "&timezone=Africa%2FAddis_Ababa")
        data = requests.get(url, timeout=15).json().get('daily', {})
        tmax = data.get('temperature_2m_max', [])
        tmin = data.get('temperature_2m_min', [])
        prec = data.get('precipitation_sum',  [])
        def smean(arr, months):
            vals = [v for i,v in enumerate(arr) if v is not None and ((i//30)+1) in months]
            return round(float(np.mean(vals)),3) if vals else 0.0
        W,Sp,Su,Au = [12,1,2],[3,4,5],[6,7,8],[9,10,11]
        annual_prec = sum([smean(prec,W),smean(prec,Sp),smean(prec,Su),smean(prec,Au)])
        return {
            "T2M_MAX-W":smean(tmax,W),"T2M_MAX-Sp":smean(tmax,Sp),"T2M_MAX-Su":smean(tmax,Su),"T2M_MAX-Au":smean(tmax,Au),
            "T2M_MIN-W":smean(tmin,W),"T2M_MIN-Sp":smean(tmin,Sp),"T2M_MIN-Su":smean(tmin,Su),"T2M_MIN-Au":smean(tmin,Au),
            "PRECTOTCORR-W":smean(prec,W),"PRECTOTCORR-Sp":smean(prec,Sp),"PRECTOTCORR-Su":smean(prec,Su),"PRECTOTCORR-Au":smean(prec,Au),
            "QV2M-W":8.3,"QV2M-Sp":9.1,"QV2M-Su":12.2,"QV2M-Au":11.0,
            "WD10M":71.6,"GWETTOP":0.71,"CLOUD_AMT":51.6,"WS2M_RANGE":6.2,
            "PS":round(77.0+(9.5-abs(float(lat)-9.0))*0.3,2),
            "annual_prec":round(annual_prec,1),"avg_temp":round(smean(tmax,Su),1),
        }
    except:
        return {"T2M_MAX-W":23.5,"T2M_MAX-Sp":24.8,"T2M_MAX-Su":22.1,"T2M_MAX-Au":23.9,
                "T2M_MIN-W":10.2,"T2M_MIN-Sp":11.5,"T2M_MIN-Su":12.8,"T2M_MIN-Au":11.0,
                "PRECTOTCORR-W":1.2,"PRECTOTCORR-Sp":2.8,"PRECTOTCORR-Su":8.5,"PRECTOTCORR-Au":3.1,
                "QV2M-W":8.3,"QV2M-Sp":9.1,"QV2M-Su":12.2,"QV2M-Au":11.0,
                "WD10M":71.6,"GWETTOP":0.71,"CLOUD_AMT":51.6,"WS2M_RANGE":6.2,"PS":78.4,
                "annual_prec":850.0,"avg_temp":22.1}

def get_topography(lat, lon):
    try:
        offset = 0.0045
        locs = [{"latitude":lat,"longitude":lon},{"latitude":lat+offset,"longitude":lon},
                {"latitude":lat-offset,"longitude":lon},{"latitude":lat,"longitude":lon+offset},
                {"latitude":lat,"longitude":lon-offset}]
        res   = requests.post("https://api.open-elevation.com/api/v1/lookup",json={"locations":locs},timeout=15)
        elevs = [r["elevation"] for r in res.json()["results"]]
        elev_c,elev_n,elev_s,elev_e,elev_w = elevs
        dz_dx = (elev_e-elev_w)/(2*500); dz_dy = (elev_n-elev_s)/(2*500)
        slope  = round(np.degrees(np.arctan(np.sqrt(dz_dx**2+dz_dy**2))),3)
        aspect = round(np.degrees(np.arctan2(dz_dx,dz_dy))%360,3)
        twi    = round(float(np.log(1000/np.tan(np.radians(max(slope,0.1))))),4)
        return {"elevation":elev_c,"slope":slope,"aspect":aspect,"TWI":max(4.0,min(15.0,twi))}
    except:
        return {"elevation":2300,"slope":5.0,"aspect":180.0,"TWI":8.5}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping')
def ping():
    return 'ok', 200

@app.route('/get_env_data', methods=['POST'])
def get_env_data():
    data = request.get_json()
    climate = get_climate(float(data['lat']), float(data['lon']))
    topo    = get_topography(float(data['lat']), float(data['lon']))
    climate.update(topo)
    return jsonify(climate)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        base_keys = ['Ph','K','P','N','Zn','S',
            'QV2M-W','QV2M-Sp','QV2M-Su','QV2M-Au',
            'T2M_MAX-W','T2M_MAX-Sp','T2M_MAX-Su','T2M_MAX-Au',
            'T2M_MIN-W','T2M_MIN-Sp','T2M_MIN-Su','T2M_MIN-Au',
            'PRECTOTCORR-W','PRECTOTCORR-Sp','PRECTOTCORR-Su','PRECTOTCORR-Au',
            'WD10M','GWETTOP','CLOUD_AMT','WS2M_RANGE','PS',
            'elevation','slope','aspect','TWI']
        row = {k: float(data.get(k,0)) for k in base_keys}
        row = engineer_features(row)
        row.update(get_soil_onehot(data.get('soil_color','brown')))
        X        = np.array([[row.get(c,0) for c in feat_cols]])
        X_df     = pd.DataFrame(X, columns=feat_cols)
        X_num    = scaler.transform(X_df[numeric_feat_cols])
        X_scaled = pd.DataFrame(X_num, columns=numeric_feat_cols)
        X_scaled[onehot_feat_cols] = X_df[onehot_feat_cols].values
        X_scaled = X_scaled[feat_cols]
        prob = xgb.predict_proba(X_scaled)[0]
        top3 = []
        for i in np.argsort(prob)[::-1][:3]:
            crop = le_label.classes_[i]
            info = CROP_INFO.get(crop,{"emoji":"🌿","season":"—","desc":""})
            top3.append({"crop":crop,"prob":round(float(prob[i])*100,1),
                         "emoji":info["emoji"],"season":info["season"],"desc":info["desc"]})
        return jsonify({"success":True,
            "location":data.get('location_name','Selected Location'),
            "elevation":int(float(data.get('elevation',2300))),
            "avg_temp":data.get('avg_temp',22.1),
            "annual_prec":data.get('annual_prec',850),"top3":top3})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)})

if __name__ == '__main__':
    app.run(debug=True)
