
import warnings
warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__,template_folder='E:/AI_ML_Final/templates')
model = joblib.load('XGBoost_Regressor_model.pkl')  # loading the saved XGBoost_regressor model

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        # ['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag']
        f_list = [request.form.get('age'), request.form.get('cement'), request.form.get('water'),
                  request.form.get('fa'),
                  request.form.get('sp'), request.form.get('bfs')]  # list of inputs

        final_features = np.array(f_list).reshape(-1, 6)
        df = pd.DataFrame(final_features)

        prediction = model.predict(df)
        result = "%.2f" % round(prediction[0], 2)

        return render_template('index.html',
                               prediction_text=f"The Concrete compressive strength is {result} MPa")


if __name__ == "__main__":
    app.run(debug=True)
