import pickle
import pandas as pd
from catboost import CatBoostRegressor

RANDOM_STATE = 42
STUDY_BEST_PARAMS = {
    'learning_rate': 0.08047291588307089, 
    'depth': 10, 
    'l2_leaf_reg': 5.851792917605791, 
    'random_strength': 1.5837729868972885, 
    'bagging_temperature': 0.23593153067360195
}

STATIC_PARAMS = {
    'iterations': 1500,
    'cat_features': ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'activation_day_of_week', 'user_type', 'image_top_1'],
    'text_features': ['title', 'description'],
    'early_stopping_rounds': 20,
    'verbose': False,
    'random_seed': 42,
    'thread_count': -1
}

def gradient_text_preprocessing_fit_transform(df_train):
    df = df_train.copy()

    df['activation_date'] = pd.to_datetime(df['activation_date'])
    df['activation_day_of_week'] = df['activation_date'].dt.day_of_week
    df['activation_day_of_week'] = df['activation_day_of_week'].astype(str)
    df['image_top_1'] = df['image_top_1'].astype(str)
    df['title'] = df['title'].astype(str)
    df['description'] = df['description'].astype(str)

    cat_features = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'activation_day_of_week', 'user_type', 'image_top_1']
    for col in cat_features:
        df[col] = df[col].astype(str)

    feature_names = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'activation_day_of_week', 'user_type', 'image_top_1', 'price', 'item_seq_number', 'title', 'description']
    
    X_train = df[feature_names]
    y_train = df['deal_probability']

    return X_train, y_train


def gradient_text_preprocessing_transform(df_valid):
    df = df_valid.copy()

    df['activation_date'] = pd.to_datetime(df['activation_date'])
    df['activation_day_of_week'] = df['activation_date'].dt.day_of_week
    df['activation_day_of_week'] = df['activation_day_of_week'].astype(str)
    df['image_top_1'] = df['image_top_1'].astype(str)
    df['title'] = df['title'].astype(str)
    df['description'] = df['description'].astype(str)
    
    cat_features = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'activation_day_of_week', 'user_type', 'image_top_1']
    for col in cat_features:
        df[col] = df[col].astype(str)
    feature_names = ['region', 'city', 'parent_category_name', 'category_name', 'param_1', 'activation_day_of_week', 'user_type', 'image_top_1', 'price', 'item_seq_number', 'title', 'description']

    X_valid = df[feature_names]
    
    if 'deal_probability' in df.columns:
        y_valid = df['deal_probability']
    else:
        y_valid = None

    return X_valid, y_valid

if __name__ == "__main__":

    df_train = pd.read_csv("practice/avito_demand/data/train.csv")
    X_train, y_train = gradient_text_preprocessing_fit_transform(df_train)

    print('Подготовили данные')

    best_params = {**STATIC_PARAMS, **STUDY_BEST_PARAMS}
    catbost = CatBoostRegressor(**best_params)

    print('Приступаем к обучению модели')

    catbost.fit(X_train, y_train)
    
    with open('catboost_model.pkl', 'wb') as f:
        pickle.dump(catbost, f)
    
    print('Модель успешно сохранена!')