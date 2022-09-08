import numpy as np 
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import lightgbm

Data_1= pd.read_csv("data/tc20171021.csv",on_bad_lines='skip')
Data_2= pd.read_csv("data/true_car_listings.csv",on_bad_lines='skip')

df = pd.concat([Data_1,Data_2])
df = df.drop_duplicates()
df.drop(columns = ['Id','Vin'],axis=1,inplace = True)
df.columns = df.columns.str.lower()

df_train, df_test = train_test_split(df,train_size=0.5, random_state=100)   

encoder = ce.BinaryEncoder(cols = [  'year',  'city', 'state',  'make', 'model'],return_df=True)
df_train = encoder.fit_transform(df_train)
df_test = encoder.transform(df_test)
    
scaler = StandardScaler()
df_train[['price','mileage']] = scaler.fit_transform(df_train[['price','mileage']])
df_test[['price','mileage']] = scaler.transform(df_test[['price','mileage']])

y_train = df_train.pop('price')
X_train = df_train
y_test = df_test.pop('price')
X_test = df_test

train_data = lightgbm.Dataset(X_train, label=y_train)
test_data =  lightgbm.Dataset(X_test, label=y_test)

model = lightgbm.train({},train_data, valid_sets=[test_data], num_boost_round=5000,early_stopping_rounds=50)   
    
pickle.dump(model,open("models/model.pkl","wb"))
pickle.dump(encoder,open("models/encoder.pkl","wb"))
pickle.dump(scaler,open("models/scaler.pkl","wb"))