from datetime import datetime

import dill
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

def new_columns(total_df: pd.DataFrame) -> pd.DataFrame:  # Добавление новых признаков в датасет
    mobile_list = ['mobile', 'tablet']
    weekend_list = [5, 6]

    total_df['mobile'] = total_df.device_category.apply(lambda x: 1 if x in mobile_list else 0)

    total_df['week_day'] = pd.to_datetime(total_df['visit_date'])
    total_df['week_day'] = total_df['week_day'].dt.weekday
    total_df['weekend'] = total_df.week_day.apply(lambda x: 1 if x in weekend_list else 0)

    # city_count_df = total_df.geo_city.value_counts().to_frame()
    # city_count_list = city_count_df[city_count_df['geo_city'] > 4500].index.tolist()
    # total_df['most_visited_city'] = total_df.geo_city.apply(lambda x: 1 if x in city_count_list else 0)

    return total_df

def utm_preparing(total_df: pd.DataFrame) -> pd.DataFrame:  # Заполнение пропущенных значений utm_*
    columns_for_preparing = [
        'utm_source',
        'utm_campaign',
        'utm_adcontent',
        'utm_keyword']

    for item in columns_for_preparing:
        nan_df = pd.DataFrame()
        nan_list = []
        nan_dict = {}

        nan_df[['utm_medium', item]] = total_df[total_df[item].isna()][['utm_medium', item]]
        nan_list = nan_df['utm_medium'].unique().tolist()

        max_utm_index_value = total_df[item].mode().tolist()[0]
        for nan_list_index in nan_list:
            max_value = total_df[total_df['utm_medium'] == nan_list_index][item].mode().tolist()
            if max_value == []:
                nan_dict[nan_list_index] = max_utm_index_value
            else:
                nan_dict[nan_list_index] = max_value[0]

        for j_key, j_value in nan_dict.items():
            nan_index_list = []
            nan_index_list = nan_df[nan_df['utm_medium'] == j_key].index.tolist()
            total_df.loc[nan_index_list, item] = j_value

    return total_df

def device_brand_preparing(total_df: pd.DataFrame) -> pd.DataFrame:  # Заполнение пропущенных значений device_brand
    nan_df = pd.DataFrame()
    device_category_list = []
    brand_dict_browser = {}
    brand_dict = {}

    nan_df[['device_category', 'device_os', 'device_brand', 'device_screen_resolution',
            'device_browser']] = total_df[total_df.device_brand.isna()][
        ['device_category', 'device_os', 'device_brand', 'device_screen_resolution', 'device_browser']]

    device_category_list = nan_df['device_category'].unique().tolist()

    for index in device_category_list:
        brand_dict_browser[index] = nan_df[nan_df['device_category'] == index].device_browser.unique().tolist()

    for j_key_1, j_value_1 in brand_dict_browser.items():
        brand_dict_1 = {}
        df_category = pd.DataFrame()
        df_category = total_df[total_df['device_category'] == j_key_1]
        category_os_list = df_category.device_os.mode().tolist()
        for index in j_value_1:
            df_browser = pd.DataFrame()
            df_browser = df_category[df_category['device_browser'] == index]
            browser_os_list = df_browser.device_os.mode().tolist()
            max_value = df_browser.device_brand.mode().tolist()
            if max_value == []:
                brand_dict_1[index] = category_os_list[0] + '_based'
            elif browser_os_list[0] == 'iOS' or browser_os_list[0] == 'Macintosh':
                brand_dict_1[index] = 'Apple'
            elif browser_os_list[0] == 'Android':
                brand_dict_1[index] = 'Samsung'
            else:
                brand_dict_1[index] = max_value[0]
        brand_dict[j_key_1] = brand_dict_1

    for j_key_1, j_value_1 in brand_dict.items():
        df_category = pd.DataFrame()
        df_category = nan_df[nan_df['device_category'] == j_key_1]
        for j_key_2, j_value_2 in j_value_1.items():
            nan_index_list = []
            nan_index_list = df_category[df_category['device_browser'] == j_key_2].index.tolist()
            total_df.loc[nan_index_list, 'device_brand'] = j_value_2

    return total_df

def device_os_preparing(total_df: pd.DataFrame) -> pd.DataFrame: # Заполнение пропущенных значений device_os
    nan_df = pd.DataFrame()
    device_category_list = []
    category_brand_dict = {}
    brand_os_dict = {}
    # total_df = total_df.copy()

    nan_df[['device_category', 'device_os', 'device_brand', 'device_screen_resolution',
            'device_browser']] = total_df[total_df.device_os.isna()][
        ['device_category', 'device_os', 'device_brand', 'device_screen_resolution', 'device_browser']]

    device_category_list = nan_df['device_category'].unique().tolist()

    for index in device_category_list:
        category_brand_dict[index] = nan_df[nan_df['device_category'] == index].device_brand.unique().tolist()

    for j_key, j_value in category_brand_dict.items():
        brand_os_dict_1 = {}
        category_df = pd.DataFrame()
        category_df = total_df[total_df['device_category'] == j_key]
        max_device_os_value = category_df.device_os.mode().tolist()[0]
        for index in j_value:
            max_value = category_df[category_df['device_brand'] == index].device_os.mode().tolist()
            if max_value == []:
                brand_os_dict_1[index] = max_device_os_value
            else:
                brand_os_dict_1[index] = max_value[0]
        brand_os_dict[j_key] = brand_os_dict_1

    for j_key_1, j_value_1 in brand_os_dict.items():
        df_category = pd.DataFrame()
        df_category = nan_df[nan_df['device_category'] == j_key_1]
        for j_key_2, j_value_2 in j_value_1.items():
            nan_index_list = []
            nan_index_list = df_category[df_category['device_brand'] == j_key_2].index.tolist()
            total_df.loc[nan_index_list, 'device_os'] = j_value_2

    return total_df

def device_count_preparing(total_df: pd.DataFrame) -> pd.DataFrame: # Уменьшение количества уникальных значений
    device_df = pd.DataFrame()
    device_list = ["utm_source", "utm_medium", "utm_campaign", "utm_adcontent", "utm_keyword",
                   "device_os", "device_brand", "device_browser", "geo_country", "geo_city"]

    device_df[device_list] = total_df[device_list]
    for item in device_list:
        item_df = pd.DataFrame()
        item_list = []

        item_df['name'] = device_df.groupby(by=[item], as_index=False)[item].max()
        item_df['count'] = device_df.groupby(by=[item], as_index=False)[item].count()
        item_list = item_df[item_df['count'] < 1000].name.tolist()
        most_popular_item = device_df[item].mode().tolist()[0]

        for index_1 in item_list:
            item_index_list = []
            item_index_list = device_df[device_df[item] == index_1].index.tolist()
            device_df.loc[item_index_list, item] = most_popular_item

        total_df[item] = device_df[item].apply(lambda x: x)

    return total_df

def device_screen_resolution_count_preparing(total_df: pd.DataFrame) -> pd.DataFrame: # Уменьшение количества уникальных значений
    device_df = pd.DataFrame()                                                        # признака device_screen_resolution
    device_brand_list = []

    device_df[['device_brand', 'device_screen_resolution']] = total_df[['device_brand', 'device_screen_resolution']]
    device_brand_list = device_df['device_brand'].unique().tolist()

    for index in device_brand_list:
        df_brand = pd.DataFrame()
        df_dsr_name_count = pd.DataFrame()
        dsr_list = []

        df_brand = device_df[device_df['device_brand'] == index]
        df_dsr_name_count['name'] = df_brand.groupby(by=['device_screen_resolution'], as_index=False)[
            'device_screen_resolution'].max()
        df_dsr_name_count['count'] = df_brand.groupby(by=['device_screen_resolution'], as_index=False)[
            'device_screen_resolution'].count()

        dsr_list = df_dsr_name_count[df_dsr_name_count['count'] < 1000].name.tolist()
        most_popular_dsr = df_brand.device_screen_resolution.mode().tolist()[0]

        for item in dsr_list:
            dsr_index_list = []
            dsr_index_list = df_brand[df_brand['device_screen_resolution'] == item].index.tolist()
            device_df.loc[dsr_index_list, 'device_screen_resolution'] = most_popular_dsr
    total_df['device_screen_resolution'] = device_df['device_screen_resolution'].apply(lambda x: x)

    return total_df

def filter_data(total_df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'week_day',
        'device_model'
    ]

    return total_df.drop(columns_to_drop, axis=1)

def pipeline() -> None:
    action_list = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                   'sub_open_dialog_click', 'sub_custom_question_submit_click',
                   'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                   'sub_car_request_submit_click']

    sessions_df = pd.read_csv('data/train/ga_sessions.csv', low_memory=False)
    ga_hits_df = pd.read_csv('data/train/ga_hits-002.csv', low_memory=False)
    ga_hits_df = ga_hits_df.drop(['hit_type', 'hit_date', 'hit_time', 'hit_number', 'hit_referer',
                                  'hit_page_path', 'event_category', 'event_label', 'event_value'], axis=1)
    ga_hits_df['event_action_result'] = ga_hits_df.event_action.apply(lambda x: 1 if x in action_list else 0)
    ga_hits_df = ga_hits_df.drop('event_action', axis=1)
    ga_hits_df = ga_hits_df.drop_duplicates()

    total_df = pd.merge(left=sessions_df, right=ga_hits_df, on='session_id', how='inner')
    total_df = total_df[~(total_df.session_id.duplicated())]

    # total_df = total_df.loc[0:150000, :]
    sessions_df = pd.DataFrame()
    ga_hits_df = pd.DataFrame()

    X = total_df.drop('event_action_result', axis=1)
    y = total_df['event_action_result']

    categorical_features = make_column_selector(dtype_include=object)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('new_columns', FunctionTransformer(new_columns)),
        ('utm_preparing', FunctionTransformer(utm_preparing)),
        ('device_brand_preparing', FunctionTransformer(device_brand_preparing)),
        ('device_os_preparing', FunctionTransformer(device_os_preparing)),
        ('device_count_preparing', FunctionTransformer(device_count_preparing)),
        ('device_screen_resolution_count_preparing', FunctionTransformer(device_screen_resolution_count_preparing)),
        ('filter', FunctionTransformer(filter_data)),
        ('column_transformer', column_transformer)
    ])

    models = [RandomForestClassifier(max_depth=22, max_features='sqrt', min_samples_split=13, n_estimators=50)]

    best_score = .0
    best_pipe = None
    for model in models:

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        pipe.fit(X_train, y_train)

        pred_rf = pipe.predict_proba(X_test)[:, 1]
        rf_roc = roc_auc_score(y_test, pred_rf)

        print(f'{model}: ROC AUC=%.3f' % (rf_roc))

        if rf_roc > best_score:
            best_score = rf_roc
            best_pipe = pipe
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, ROC AUC: {best_score:.4f}')

    best_pipe.fit(X, y)

    model_filename = f'data/models/sber_avtopodpiska_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'

    with open(model_filename, 'wb') as file:
        dill.dump(best_pipe, file)

    print(f'Model is saved as {model_filename}')

if __name__ == '__main__':
    pipeline()
