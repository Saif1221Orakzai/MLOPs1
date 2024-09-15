from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = pd.DataFrame([[
        data['bedrooms'], data['bathrooms'], data['garage'], data['land_area'], data['floor_area'],
        data['build_year'], data['cbd_dist'], data['nearest_stn_dist'], data['latitude'], data['longitude'],
        data['nearest_sch_dist'], data['nearest_sch_rank']
    ]], columns=['BEDROOMS', 'BATHROOMS', 'GARAGE', 'LAND_AREA', 'FLOOR_AREA', 
                 'BUILD_YEAR', 'CBD_DIST', 'NEAREST_STN_DIST', 'LATITUDE', 'LONGITUDE', 
                 'NEAREST_SCH_DIST', 'NEAREST_SCH_RANK'])

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
