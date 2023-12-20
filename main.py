from utils import Utils
import pandas as pd
import numpy as np

import os

if __name__ == "__main__":
    path_document = '../../datasets/mediciones.csv'
    utils = Utils()
    df = utils.load_df(path=path_document)
    #1. Pipe data process 
    df = utils.pipeline_process_data(df)
    
    features = ['prev_kwh','hour','day','dayofweek','dayofyear','quarter','weekofyear']
    target = 'kWh_2'
    
    if not os.path.exists('models_folder/forecast.pkl'):
        #2. select features and target 
        model = utils.pipeline_training_model(df,features,target)
    else:
        start_date = np.max(df.index)
        last_kwh = df.loc[df.index==start_date,'prev_kwh']
        end_date = "2023-01-01"
        
        results = utils.pipeline_generate_predicts(start_date=start_date,
                                                   end_date=end_date,
                                                   last_kwh=last_kwh,
                                                   features=features)
        
        final_result = [{"date" : i, "predict" : j[0]} for i,j in results.items()]
        final_df = pd.DataFrame(final_result)
        
        ##save cvs with predicts 
        final_df.to_csv("./results/data_forecasted.csv")
        
        
        
        
        
