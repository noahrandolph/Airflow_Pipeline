import intern_machine_learning_framework.functions.model_db_interface as model_db
import logging
import math
import numpy as np
import pandas as pd
import pickle
from impala.dbapi import connect
from impala.util import as_pandas
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import gen_features

np.random.seed(42)

PICKLE_PATH = '/Users/garb55/airflow/dags/intern_machine_learning_framework/pickled_mock_data/'

MAPPER = {'term': {'60 months': 1, '36 months': 0, '': None},
          'emp_length': {'10': 11, '9 years': 10, '8 years': 9, '7 years': 8, '6 years': 7,
                         '5 years': 6, '4 years': 5, '3 years': 4,
                         '2 years': 3, '1 year': 2, '< 1 year': 1,
                         'n/a': 0, '': None},  # Assume 'n/a' means unemployed
          'home_ownership': {'OWN': 4, 'MORTGAGE': 3, 'OTHER': 2, 'RENT': 1, 'NONE': 0, '': None},
          'loan_status': {'Fully Paid': 6, 'Current': 5, 'In Grace Period': 4, '16': 3,
                          '31': 2, 'Default': 1, 'Charged Off': 0, '': None}}

NUMERICAL_ATTRIBUTES = ['loan_amt', 'funded_amt', 'term', 'int_rate', 'installment', 'emp_length',
                        'home_ownership', 'annual_inc', 'loan_status', 'dti', 'delinq_2yrs',
                        'earliest_cr_line', 'open_acc', 'revol_bal', 'total_acc', 'out_prncp',
                        'total_pymnt', 'total_rec_prncp', 'total_rec_int']

CATEGORICAL_ATTRIBUTES = ['purpose', 'addr_state']


def pickle_object_with_xcom_push(object, file_name, ti):
    file_path = PICKLE_PATH + file_name + '.pkl'
    file_object = open(file_path, 'wb')
    pickle.dump(object, file_object)
    file_object.close()
    ti.xcom_push(key=file_name, value=file_path)


def xcom_pull_and_unpickle_object(key, task_ids, ti):
    file_path = ti.xcom_pull(key=key, task_ids=task_ids)
    file_object = open(file_path, 'rb')
    object = pickle.load(file_object)
    file_object.close()
    return object


def unpickle_object_without_xcom_pull(file_name):
    file_path = PICKLE_PATH + file_name + '.pkl'
    file_object = open(file_path, 'rb')
    object = pickle.load(file_object)
    file_object.close()
    return object


def load_training_data_from_db(**kwargs):
    ti = kwargs['ti']
    file_name = 'raw_training_data'
    ip_address = '0.0.0.0'
    source_dataset_id = 'raw_mock_loans' + str(datetime.now())

    connection = connect(host=ip_address, port=10000, database='default', user='cloudera', auth_mechanism='PLAIN')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM mock_loans')

    if cursor.description is None:
        raw_training_data = None
    else:
        raw_training_data = as_pandas(cursor)

    raw_training_data.columns = list(map(lambda x: x if '.' not in x else x.split('.')[-1],
                                         raw_training_data.columns))
    cursor.close()
    connection.close()

    raw_training_data = {'raw_training_data': raw_training_data,
                         'source_dataset_id': source_dataset_id}

    pickle_object_with_xcom_push(raw_training_data, file_name, ti)
    model_db.store_source_dataset_meta(source_dataset_id=source_dataset_id,
                                       file_name=file_name,
                                       ip_address=ip_address,
                                       created_date='unknown',
                                       is_valid='unknown',
                                       kwargs=kwargs)


def load_unseen_prediction_data_from_db(**kwargs):
    ti = kwargs['ti']
    connection = connect(host='0.0.0.0', port=10000, database='default', user='cloudera', auth_mechanism='PLAIN')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM random_loans')

    if cursor.description is None:
        raw_unseen_prediction_data = None
    else:
        raw_unseen_prediction_data = as_pandas(cursor)

        raw_unseen_prediction_data.columns = list(map(lambda x: x if '.' not in x else x.split('.')[-1],
                                                      raw_unseen_prediction_data.columns))
    cursor.close()
    connection.close()
    pickle_object_with_xcom_push(raw_unseen_prediction_data, 'raw_unseen_prediction_data', ti)


class LogTransformer(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, column=None, log_func=None):  # no *args or **kwargs allowed
        self.column = column
        self.log_func = log_func
        self.feature_names_ = None

    def fit(self, x, y=None):  # fit method required for pipeline functionality
        self.feature_names_ = list(x.columns.values)
        return self

    def transform(self, x, y=None):
        x[self.column] = x[self.column].apply(self.log_func)
        return x

    def get_feature_names(self):
        return self.feature_names_


class DateStringToInt(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, column=None, int_type=None):  # no *args or **kwargs allowed
        self.column = column
        self.int_type = int_type
        self.feature_names_ = None

    def fit(self, x, y=None):  # fit method required for pipeline functionality
        self.feature_names_ = list(x.columns.values)
        return self

    def transform(self, x, y=None):
        x[self.column] = pd.to_datetime(x[self.column])
        x[self.column] = x[x[self.column].notna()][self.column].astype(self.int_type) // 10 ** 9
        return x

    def get_feature_names(self):
        return self.feature_names_


# Return a data frame from imputation rather than a numpy array
class DataFrameImputer(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, num_attribs=None, strategy=None):  # no *args or **kwargs allowed
        self.num_attribs = [[i] for i in num_attribs]
        self.feature_def = gen_features(columns=self.num_attribs, classes=[{'class': Imputer, 'strategy': strategy}])
        self.mapper = DataFrameMapper(self.feature_def, input_df=True, df_out=True, default=None)
        self.feature_names_ = None

    def fit(self, x, y=None):
        self.feature_names_ = list(x.columns.values)
        self.mapper.fit(x)
        return self

    def transform(self, x, y=None):
        x_np = self.mapper.transform(x)
        x_df = pd.DataFrame(x_np, columns=x.columns)
        return x_df

    def get_feature_names(self):
        return self.feature_names_


# Return a data frame from standard scaling rather than a numpy array
class DataFrameStandardScaler(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, num_attribs=None):  # no *args or **kwargs allowed
        self.num_attribs = [[i] for i in num_attribs]
        self.feature_def = gen_features(columns=self.num_attribs, classes=[{'class': StandardScaler}])
        self.mapper = DataFrameMapper(self.feature_def, input_df=True, df_out=True, default=None)
        self.feature_names_ = None

    def fit(self, x, y=None):
        self.feature_names_ = list(x.columns.values)
        self.mapper.fit(x)
        return self

    def transform(self, x, y=None):
        x_np = self.mapper.transform(x)
        x_df = pd.DataFrame(x_np, columns=x.columns)
        return x_df

    def get_feature_names(self):
        return self.feature_names_


class OrdinalAttributeConverter(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, mapper):  # no *args or **kwargs allowed
        self.mapper = mapper
        self.feature_names_ = None

    def fit(self, x, y=None):  # fit method required for pipeline functionality
        self.feature_names_ = list(x.columns.values)
        return self

    def transform(self, x, y=None):
        return x.replace(to_replace=self.mapper, regex=True)

    def get_feature_names(self):
        return self.feature_names_


class NominalAttributeImputer(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, columns=[]):  # no *args or **kwargs allowed
        self.columns = columns
        self.most_frequent = dict.fromkeys(columns)
        self.feature_names_ = None

    def fit(self, x, y=None):
        self.feature_names_ = list(x.columns.values)
        for i in self.columns:
            self.most_frequent[i] = x[i].value_counts().index[0]
        return self

    def transform(self, x, y=None):
        for i in self.columns:
            x[i] = x[i].apply(lambda z: self.most_frequent[i] if z is None else z)
        return x

    def get_feature_names(self):
        return self.feature_names_


class ModifiedLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def fit(self, x, y=None):
        self.label_encoder.fit(x)
        return self

    def transform(self, x, y=None):
        return self.label_encoder.transform(x).reshape(-1, 1)


class ModifiedOneHotEncoder(BaseEstimator, TransformerMixin):  # arguments required for pipeline functionality
    def __init__(self, columns=None, bool_list=None, n_values=None):  # no *args or **kwargs allowed
        self.columns = columns
        self.n_values = n_values
        self.label_encoder = ModifiedLabelEncoder()
        self.one_hot_encoder = OneHotEncoder(categorical_features=bool_list, n_values=n_values)
        self.feature_names_ = None

    def fit(self, x, y=None):
        self.feature_names_ = list(x.columns.values)
        x[self.columns[0]] = self.label_encoder.fit_transform(x[self.columns[0]])
        x[self.columns[1]] = self.label_encoder.fit_transform(x[self.columns[1]])
        self.one_hot_encoder.fit(x)
        return self

    def transform(self, x, y=None):
        logging.warning("Fitting the label encoder here assigns different numbers to different categories, " +
                        "which may affect the consistency of the model with new data when there are new or " +
                        "missing categories compared to the training data. Label encoding should just be a " +
                        "transform in the transform method, but more research is needed to make that work " +
                        "without errors.")
        x[self.columns[0]] = self.label_encoder.fit_transform(x[self.columns[0]])
        x[self.columns[1]] = self.label_encoder.fit_transform(x[self.columns[1]])
        x = self.one_hot_encoder.transform(x)
        return x

    def get_feature_names(self):
        return self.feature_names_


def one_hot_indices(cat_attribs):
    wrangled_training_and_test_data = unpickle_object_without_xcom_pull('wrangled_training_and_test_data')
    train_data_wrangled = wrangled_training_and_test_data['train_data_wrangled']
    columns = np.array(list(train_data_wrangled.columns.values))
    bool_array = (columns == cat_attribs[0]) | (columns == cat_attribs[1])
    return bool_array


def create_pipeline(num_attribs, cat_attribs, estimator):
    pipeline = Pipeline([
        ('logtransformer', LogTransformer('annual_inc', log_func=np.log1p)),
        ('datetimeconv', DateStringToInt('earliest_cr_line', int_type=np.int64)),
        ('ordinalconv', OrdinalAttributeConverter(mapper=MAPPER)),
        ('num_imputer', DataFrameImputer(num_attribs=num_attribs, strategy='median')),
        ('std_scaler', DataFrameStandardScaler(num_attribs=num_attribs)),
        ('cat_imputer', NominalAttributeImputer(cat_attribs)),
        ('one_hot', ModifiedOneHotEncoder(columns=cat_attribs, bool_list=one_hot_indices(cat_attribs),
                                          n_values=[14, 45])),
        ('estimator', estimator)
    ])
    return pipeline


def split_data_into_train_and_test(raw_training_data):
    """Will need to use a unique and immutable identifier from Hyundai Card data for
    updating data sets with new data. This will ensure data previously in training
    set is not randomly allocated to test set in future runs."""
    train_set, test_set = train_test_split(raw_training_data, test_size=0.2, random_state=42)
    return train_set, test_set


def impute_missing_labels(series_of_labels, missing_label_value_to_impute):
    return series_of_labels.apply(lambda x: missing_label_value_to_impute if math.isnan(x) else x)


def separate_features_and_impute_missing_labels(data_set, label, missing_label_value_to_impute):
    features = data_set.drop(label, axis=1)  # axis=1 creates a copy
    labels = impute_missing_labels(data_set[label], missing_label_value_to_impute)
    return features, labels


def wrangle_training_data(**kwargs):
    ti = kwargs['ti']
    file_name = 'wrangled_training_data'
    ip_address = '0.0.0.0'
    wrangled_dataset_id = str(datetime.now())

    dict_object = xcom_pull_and_unpickle_object('raw_training_data', 'load_raw_training_data', ti)
    raw_training_data = dict_object['raw_training_data']
    source_dataset_id = dict_object['source_dataset_id']

    train_set, test_set = split_data_into_train_and_test(raw_training_data)

    train_data, train_data_labels = separate_features_and_impute_missing_labels(train_set, 'mths_since_last_delinq', 0)
    test_data, test_data_labels = separate_features_and_impute_missing_labels(test_set, 'mths_since_last_delinq', 0)

    wrangled_training_and_test_data = {'train_data_wrangled': train_data,
                                       'train_data_labels': train_data_labels,
                                       'test_data_wrangled': test_data,
                                       'test_data_labels': test_data_labels,
                                       'wrangled_dataset_id': wrangled_dataset_id}

    pickle_object_with_xcom_push(wrangled_training_and_test_data, 'wrangled_training_and_test_data', ti)
    model_db.store_wrangled_dataset_meta(wrangled_dataset_id=wrangled_dataset_id,
                                         source_dataset_id=source_dataset_id,
                                         file_name=file_name,
                                         ip_address=ip_address,
                                         created_date='unknown',
                                         is_valid='unknown',
                                         kwargs=kwargs)


def wrangle_unseen_data(**kwargs):
    ti = kwargs['ti']

    raw_unseen_prediction_data = xcom_pull_and_unpickle_object('raw_unseen_prediction_data',
                                                               'load_raw_unseen_prediction_data', ti)

    wrangled_features_for_prediction = raw_unseen_prediction_data.drop('mths_since_last_delinq',
                                                                       axis=1)  # axis=1 creates a copy

    pickle_object_with_xcom_push(wrangled_features_for_prediction, 'wrangled_features_for_prediction', ti)


def train_linear_regression_model(**kwargs):
    ti = kwargs['ti']
    file_name = 'linear_regression_model'
    ip_address = '0.0.0.0'
    predictive_model_id = str(datetime.now())
    hyperparameters = 'default'

    wrangled_data = xcom_pull_and_unpickle_object('wrangled_training_and_test_data', 'wrangle_training_data', ti)

    train_data_wrangled = wrangled_data["train_data_wrangled"]
    train_data_labels = wrangled_data["train_data_labels"]
    wrangled_dataset_id = wrangled_data['wrangled_dataset_id']

    linear_regression_model = create_pipeline(NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES, LinearRegression())

    linear_regression_model.fit(train_data_wrangled, train_data_labels)

    linear_regression_model_and_id = {'model': linear_regression_model,
                                      'predictive_model_id': predictive_model_id,
                                      'file_name': file_name,
                                      'hyperparameters': hyperparameters}

    pickle_object_with_xcom_push(linear_regression_model_and_id, file_name, ti)
    model_db.store_predictive_model_meta(predictive_model_id=predictive_model_id,
                                         wrangled_dataset_id=wrangled_dataset_id,
                                         hyperparameters=hyperparameters,
                                         file_name=file_name,
                                         ip_address=ip_address,
                                         created_date='unknown',
                                         is_valid='unknown',
                                         kwargs=kwargs)
    model_db.store_model_feature_importance(predictive_model_id=predictive_model_id,
                                            feature_importance='not applicable')


def train_decision_tree_model(**kwargs):
    ti = kwargs['ti']
    file_name = 'decision_tree_regression_model'
    ip_address = '0.0.0.0'
    predictive_model_id = str(datetime.now())
    hyperparameters = 'default'

    wrangled_data = xcom_pull_and_unpickle_object('wrangled_training_and_test_data', 'wrangle_training_data', ti)

    train_data_wrangled = wrangled_data["train_data_wrangled"]
    train_data_labels = wrangled_data["train_data_labels"]
    wrangled_dataset_id = wrangled_data['wrangled_dataset_id']

    decision_tree_regression_model = create_pipeline(NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES,
                                                     DecisionTreeRegressor())
    decision_tree_regression_model.fit(train_data_wrangled, train_data_labels)

    decision_tree_regression_model_and_id = {'model': decision_tree_regression_model,
                                             'predictive_model_id': predictive_model_id,
                                             'file_name': file_name,
                                             'hyperparameters': hyperparameters}

    pickle_object_with_xcom_push(decision_tree_regression_model_and_id, file_name, ti)
    model_db.store_predictive_model_meta(predictive_model_id=predictive_model_id,
                                         wrangled_dataset_id=wrangled_dataset_id,
                                         hyperparameters=hyperparameters,
                                         file_name=file_name,
                                         ip_address=ip_address,
                                         created_date='unknown',
                                         is_valid='unknown',
                                         kwargs=kwargs)
    feature_importance = str(decision_tree_regression_model.steps[-1][1].feature_importances_)
    model_db.store_model_feature_importance(predictive_model_id=predictive_model_id,
                                            feature_importance=feature_importance)


def train_random_forest_model(**kwargs):
    ti = kwargs['ti']
    file_name = 'random_forest_regression_model'
    ip_address = '0.0.0.0'
    predictive_model_id = str(datetime.now())
    hyperparameters = {'n_estimators': 14, 'max_features': 60}  # tuned from Jupyter NB grid search

    wrangled_data = xcom_pull_and_unpickle_object('wrangled_training_and_test_data', 'wrangle_training_data', ti)

    train_data_wrangled = wrangled_data["train_data_wrangled"]
    train_data_labels = wrangled_data["train_data_labels"]
    wrangled_dataset_id = wrangled_data['wrangled_dataset_id']

    random_forest_regression_model = create_pipeline(NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES,
                                                     RandomForestRegressor(n_estimators=
                                                                           hyperparameters['n_estimators'],
                                                                           max_features=
                                                                           hyperparameters['max_features']))
    random_forest_regression_model.fit(train_data_wrangled, train_data_labels)

    random_forest_regression_model_and_id = {'model': random_forest_regression_model,
                                             'predictive_model_id': predictive_model_id,
                                             'file_name': file_name,
                                             'hyperparameters': hyperparameters}

    pickle_object_with_xcom_push(random_forest_regression_model_and_id, file_name, ti)
    model_db.store_predictive_model_meta(predictive_model_id=predictive_model_id,
                                         wrangled_dataset_id=wrangled_dataset_id,
                                         hyperparameters=hyperparameters,
                                         file_name=file_name,
                                         ip_address=ip_address,
                                         created_date='unknown',
                                         is_valid='unknown',
                                         kwargs=kwargs)
    feature_importance = str(random_forest_regression_model.steps[-1][1].feature_importances_)
    model_db.store_model_feature_importance(predictive_model_id=predictive_model_id,
                                            feature_importance=feature_importance)


def train_svm_model(**kwargs):
    ti = kwargs['ti']
    file_name = 'svm_regression_model'
    ip_address = '0.0.0.0'
    predictive_model_id = str(datetime.now())
    hyperparameters = 'not applicable'

    wrangled_data = xcom_pull_and_unpickle_object('wrangled_training_and_test_data', 'wrangle_training_data', ti)

    train_data_wrangled = wrangled_data["train_data_wrangled"]
    train_data_labels = wrangled_data["train_data_labels"]
    wrangled_dataset_id = wrangled_data['wrangled_dataset_id']

    svm_regression_model = create_pipeline(NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES, SVR())
    svm_regression_model.fit(train_data_wrangled, train_data_labels)

    svm_regression_model_and_id = {'model': svm_regression_model,
                                   'predictive_model_id': predictive_model_id,
                                   'file_name': file_name,
                                   'hyperparameters': hyperparameters}

    pickle_object_with_xcom_push(svm_regression_model_and_id, file_name, ti)
    model_db.store_predictive_model_meta(predictive_model_id=predictive_model_id,
                                         wrangled_dataset_id=wrangled_dataset_id,
                                         hyperparameters=hyperparameters,
                                         file_name=file_name,
                                         ip_address=ip_address,
                                         created_date='unknown',
                                         is_valid='unknown',
                                         kwargs=kwargs)
    model_db.store_model_feature_importance(predictive_model_id=predictive_model_id,
                                            feature_importance='not applicable')


def select_best_model(**kwargs):
    ti = kwargs['ti']

    linear_regression_model_and_id = xcom_pull_and_unpickle_object('linear_regression_model',
                                                                   'train_linear_regression_model', ti)
    decision_tree_regression_model_and_id = xcom_pull_and_unpickle_object('decision_tree_regression_model',
                                                                          'train_decision_tree_model', ti)
    random_forest_regression_model_and_id = xcom_pull_and_unpickle_object('random_forest_regression_model',
                                                                          'train_random_forest_model', ti)
    svm_regression_model_and_id = xcom_pull_and_unpickle_object('svm_regression_model',
                                                                'train_svm_model', ti)

    wrangled_data = xcom_pull_and_unpickle_object('wrangled_training_and_test_data',
                                                  'wrangle_training_data', ti)

    model_list = [linear_regression_model_and_id,
                  decision_tree_regression_model_and_id,
                  random_forest_regression_model_and_id,
                  svm_regression_model_and_id]

    max_score = 0
    best_model_index = 0

    for i in range(len(model_list)):
        preds = model_list[i]['model'].predict(wrangled_data["test_data_wrangled"])
        score = np.sqrt(mean_squared_error(wrangled_data["test_data_labels"], preds))
        model_db.store_model_evaluation(predictive_model_id=model_list[i]['predictive_model_id'],
                                        metric_name='rmse',
                                        metric_value=score)
        if score > max_score:
            max_score = score
            best_model_index = i

    pickle_object_with_xcom_push(model_list[best_model_index], 'best_model', ti)
    model_db.store_selected_model_meta(predictive_model_id=model_list[best_model_index]['predictive_model_id'],
                                       file_name=model_list[best_model_index]['file_name'],
                                       hyperparameters=model_list[best_model_index]['hyperparameters'],
                                       kwargs=kwargs)


def notify(**kwargs):
    pass  # change DAG task to EmailOperator when ready to send email notifications.


def predict_from_unseen_data(**kwargs):
    ti = kwargs['ti']

    best_model = unpickle_object_without_xcom_pull('best_model')
    wrangled_features_for_prediction = xcom_pull_and_unpickle_object('wrangled_features_for_prediction',
                                                                     'wrangle_unseen_data', ti)

    predictions_from_unseen_data = best_model['model'].predict(wrangled_features_for_prediction)

    pickle_object_with_xcom_push(predictions_from_unseen_data, 'predictions_from_unseen_data', ti)


def store_predictions_as_i_tags(**kwargs):
    ti = kwargs['ti']

    predictions_from_unseen_data = xcom_pull_and_unpickle_object('predictions_from_unseen_data',
                                                                 'predict_from_unseen_data', ti)

    # (future) store as i-tags in database...
    # TODO implement this function
