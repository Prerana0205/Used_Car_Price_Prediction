import pandas as pd
import numpy as np
from flask import Flask,render_template,request
import pickle


app = Flask(__name__)

def process_data(data, scaler):
    encoder=pickle.load(open("models/encoder.pkl","rb"))
    data = {  "price":[12],
              "year":[data['year']],
              "mileage":[data['mileage']],
              "city": [data['city']],
              "state":[data['state']],
              "make":[data['make']],
              "model":[data['model']], }
    data = pd.DataFrame(data)
    data_en = encoder.transform(data)
    data_en[['price','mileage']] = scaler.transform(
        data_en[['price','mileage']])
    data_en.pop('price')
    return data_en

@app.route('/')
def home():    
    form_data =request.form  
    return render_template('index.html',form_data = form_data)

@app.route('/api', methods=['POST'])
def predict():
    form_data =request.form  
    model=pickle.load(open("models/model.pkl","rb"))
    scaler =pickle.load(open("models/scaler.pkl","rb"))
    encoder=pickle.load(open("models/encoder.pkl","rb"))
    data_processed = process_data(data=form_data, scaler=scaler)
    res=model.predict(data_processed)
    pred_price = scaler.inverse_transform([[res[0],0]])
    Price= pred_price[0][0]
    return render_template('index.html', output= f"Your Used Car price is {((np.round(Price,2)))} $")
  
if __name__ == "__main__":
    app.run()
   