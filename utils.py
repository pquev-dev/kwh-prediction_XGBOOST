import pandas as pd
import numpy as np
from hampel import hampel

from models import Models

import joblib

class Utils:    
    
#**********LOAD CSV****************
    def load_csv(self,path,id=439950):
        df = pd.read_csv(path,sep=';')
        df = df.loc[df['id_medidor']==id,].reset_index()
        return df[['id_medidor','date','Consumo']]
    
    def drop_na_values(self,data):
        data = data.dropna().reset_index()
        return data
    
    def change_format_date(self,data,type='H'):
        if type=='H':
            data['new_date'] = data['date'].dt.floor('H')
            new_frecuency = data.groupby(['new_date']).agg(
                kWh = ('Consumo','sum'),
                kW = ('Consumo','max')
            ).reset_index()
        return new_frecuency
    
    def load_df(self,path):
        df = self.load_csv(path)
        df = self.drop_na_values(df)
        #
        df['date'] = pd.to_datetime(df['date'])
        #chanche 15 minutes to hour 
        df = self.change_format_date(df)
        return df
    
##*********ADD PREV DAY******************
    def add_prev_day(self,data,colum):
        l = len(data)
        for i in range(l-1):
            q = data.iloc[(i-1),][colum]
            data.loc[data.index==i,'prev_kwh'] = q
        data = data.drop([np.min(data.index),np.max(data.index)],axis=0)
        return data.reset_index()
    
    def create_date_features(self,data,date_column):
        ##date to index 
        data['hour'] = data[date_column].dt.hour 
        data['day'] = data[date_column].dt.day
        data['dayofweek'] = data[date_column].dt.dayofweek
        data['dayofyear'] = data[date_column].dt.dayofyear
        data['quarter'] = data[date_column].dt.quarter
        data['weekofyear'] = data[date_column].dt.isocalendar().week
        
        data = data.set_index(date_column)
        return data
    
    def generate_predict(self,):
        pass
    
    def pipeline_process_data(self,data,time_predict=365):
        #1. Hampel filter 
        new_kwh = hampel(data['kWh'],window_size=3,imputation=True)
        data['kWh_2'] = new_kwh
        #2. add prev day 
        df = self.add_prev_day(data,'kWh_2')
        #3. create new features per date
        df = self.create_date_features(df,'new_date')
        
        print(df)
        return df
    
    def pipeline_training_model(self,data,features,target):
        X = data[features]
        y = data[target]
                
        training_data = tuple((X,y))
        Models().split_time_series(data,features,target)
        Models().training_model(data,features=features,target=target)
        
    def generate_date_range(self,start_date,end_date):
        data_range = pd.date_range(start=start_date,end=end_date,freq='H')
        df = pd.DataFrame({'fecha' : data_range})
        return df
    
    def pipeline_generate_predicts(self,start_date,end_date,last_kwh,features):
        #load model 
        model = joblib.load('./models_folder/forecast.pkl')
        
        df = self.generate_date_range(start_date,end_date)
        df = self.create_date_features(df,'fecha')
        
        df['prev_kwh'] = 0
        df.loc[df.index==start_date,'prev_kwh'] = last_kwh
        aux=last_kwh
        all_predicts = {}
        
        for i in df.index:
            row = df.loc[df.index==i,]
            row = row[features]
            row['prev_kwh'] = aux
            y_hat = model.predict(row)
            aux = y_hat
            all_predicts[i] = y_hat
            
        return all_predicts
        
        
        
        
        
        