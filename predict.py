import pickle
import numpy as np
import pandas as pd
from train import gradient_text_preprocessing_transform



if __name__ == "__main__":
    
    with open('practice/avito_demand/catboost_model.pkl', 'rb') as f:
        catboost = pickle.load(f)
    
    df_test = pd.read_csv("practice/avito_demand/data/test.csv")
    X_test, y_test = gradient_text_preprocessing_transform(df_test)

    final_preds = catboost.predict(X_test)
    final_preds = np.clip(final_preds, 0, 1)

    submission = pd.DataFrame({ 
        'item_id': df_test['item_id'],
        'deal_probability': final_preds
    })
    submission.to_csv('avito_submission.csv', index=False)