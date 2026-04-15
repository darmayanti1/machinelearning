from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model_xgb.pkl")

# Load fitur (urutan kolom)
try:
    fitur = joblib.load("fitur.pkl")
except:
    fitur = None

# ========================
# PREPROCESSING FUNCTION
# ========================
def preprocess_input(data):

    df = pd.DataFrame([data])

    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Feature engineering
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek

    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['temp_diff'] = df['temp'] - df['atemp']
    df['time_category'] = df['hour'].apply(
        lambda x: 0 if 0 <= x <= 5 else
                  1 if 6 <= x <= 9 else
                  2 if 10 <= x <= 15 else
                  3 if 16 <= x <= 19 else
                  4
    )

    # Drop datetime
    df = df.drop(columns=['datetime'])

    # Urutkan fitur (biar sama dengan training)
    if fitur is not None:
        df = df[fitur]

    return df

# ========================
# ROUTES
# ========================
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = {
            "season": float(request.form['season']),
            "holiday": float(request.form['holiday']),
            "workingday": float(request.form['workingday']),
            "weather": float(request.form['weather']),
            "temp": float(request.form['temp']),
            "atemp": float(request.form['atemp']),
            "humidity": float(request.form['humidity']),
            "windspeed": float(request.form['windspeed']),
            "datetime": request.form['datetime']
        }

        # Preprocessing otomatis
        processed = preprocess_input(data)

        # Prediksi
        pred = model.predict(processed)

        return render_template("index.html", prediction=round(pred[0], 2))

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

# ========================
# RUN APP
# ========================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)