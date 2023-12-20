import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import os

class Models:
    
    def __init__(self,):
        self.models = {
            'xgboost' : xgb.XGBRegressor()
        }
        self.params = {
            'xgboost' : {
                'max_dept' : 3,
                'n_estimaros' : 1000,
                'objective' : 'reg:linear',
                'booster' : 'gbtree',
                'base_score' : 0.5,
                'learning_rate' : 0.01
            }
        }
        
    def split_time_series(self,data,features,target):
        print(f"tamano {data.shape}")
        size_test = int(data.shape[0]*0.20)
        tss = TimeSeriesSplit(n_splits=2,test_size=size_test,gap=24)
        
        fold = 0
        preds = []
        scores = []
        reg = None
        
        for train_idx, val_idx in tss.split(data):
            train = data.iloc[train_idx]
            test = data.iloc[val_idx]
            
            X_train = train[features]
            X_test = test[features]
            y_train = train[target]
            y_test = test[target]
            
            model = self.models['xgboost']
            print(model)
            reg = model
            reg.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test,y_test)],verbose=100)
            
            y_pred = reg.predict(X_test)
            preds.append(y_pred)
            score = mean_squared_error(y_test,y_pred,squared=False)
            scores.append(score)
            
        print(scores)
    
    def training_model(self,data,features,target): 
        X = data[features]
        y = data[target]
        
        model = self.models['xgboost']
        reg = model.fit(X,y,eval_set=[(X,y)])
        
        xgb.plot_importance(reg)
        plt.savefig('./img/importance_plot.png')
        
        def save_model(model):
            try:
                if not os.path.exists('models_folder'):
                    os.mkdir('models_folder')
                joblib.dump(model,'./models_folder/forecast.pkl')
            except Exception as err: 
                print(f"ERROR! {err}")
        ##guardar modelo
        save_model(reg)
        
            
             
        
        
        
        
        