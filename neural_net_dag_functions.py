import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import pandas.io.sql as pdsql
import pickle
import sqlalchemy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from neural_net_dag.functions.feature_selector import FeatureSelector

DATA_SOURCE_PATH = "/Users/garb55/airflow/dags/neural_net_dag/dataset/home_credit/sources/"
PREDICTION_OUTPUT_PATH = "/Users/garb55/airflow/dags/neural_net_dag/dataset/home_credit/outputs/"
PICKLE_PATH = "/Users/garb55/airflow/dags/neural_net_dag/pickled_nn_objects/"
MODEL_DIRECTORY = "/Users/garb55/airflow/dags/neural_net_dag/models/"
EXPORT_DIRECTORY = "/Users/garb55/airflow/dags/neural_net_dag/exported_saved_models/"


def pickle_object_with_xcom_push(object, file_name, ti):
    file_path = PICKLE_PATH + file_name + '.pkl'
    file_object = open(file_path, 'wb')
    pickle.dump(object, file_object)
    file_object.close()
    ti.xcom_push(key=file_name, value=file_path)


def pickle_object_without_xcom_push(object, file_name):  # Delete this function (just for speedy testing)
    file_path = PICKLE_PATH + file_name + '.pkl'
    file_object = open(file_path, 'wb')
    pickle.dump(object, file_object)
    file_object.close()


def xcom_pull_and_unpickle_object(key, task_ids, ti):
    file_path = ti.xcom_pull(key=key, task_ids=task_ids)
    file_object = open(file_path, 'rb')
    object = pickle.load(file_object)
    file_object.close()
    return object


def unpickle_object_without_xcom_pull(file_name):
    file_path = PICKLE_PATH + file_name + '.pkl'
    try:
        file_object = open(file_path, 'rb')
        object = pickle.load(file_object)
        file_object.close()
        return object
    except FileNotFoundError:
        logging.info('no pickled file present')
        return False


def load_training_data_from_db(**kwargs):
    ti = kwargs['ti']
    # engine = sqlalchemy.create_engine("hive://cloudera@localhost:10000/default")
    # raw_training_data = pdsql.read_sql_table('mock_loans', engine)
    raw_training_data = pd.read_csv(DATA_SOURCE_PATH + 'application_train.csv')
    pickle_object_with_xcom_push(raw_training_data, 'raw_training_data', ti)


def load_unseen_prediction_data_from_db(**kwargs):
    ti = kwargs['ti']
    # engine = sqlalchemy.create_engine("hive://cloudera@localhost:10000/default")
    # raw_training_data = pdsql.read_sql_table('random_loans', engine)
    raw_unseen_prediction_data = pd.read_csv(DATA_SOURCE_PATH + 'application_test.csv')
    pickle_object_with_xcom_push(raw_unseen_prediction_data, 'raw_unseen_prediction_data', ti)


class ModifiedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=None, correlation_threshold=None, task=None, eval_metric=None,
                 cumulative_importance=None):
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.task = task
        self.eval_metric = eval_metric
        self.cumulative_importance = cumulative_importance
        self.selected_features = None

    def fit(self, x, y=None):
        fs = FeatureSelector(data=x, labels=y)
        fs.identify_all(selection_params={'missing_threshold': self.missing_threshold,
                                          'correlation_threshold': self.correlation_threshold,
                                          'task': self.task,
                                          'eval_metric': self.eval_metric,
                                          'cumulative_importance': self.cumulative_importance})
        self.selected_features = fs.remove(methods='all', keep_one_hot=False)
        pickle_object_without_xcom_push(self.selected_features, 'selected_features')
        return self

    def transform(self, x, y=None):
        x = x[self.selected_features.columns].copy()
        return x


class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self, category_object_fill=None, category_integer_fill=None, numeric_fill=None):
        self.category_object_fill = category_object_fill
        self.category_integer_fill = category_integer_fill
        self.numeric_fill = numeric_fill

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        categorical_objects_index = x.dtypes[x.dtypes == 'object'].index
        categorical_integers_index = x.dtypes[x.dtypes == 'int64'].index
        numeric_index = x.dtypes[x.dtypes == 'float64'].index
        x[categorical_objects_index] = x[categorical_objects_index].fillna(self.category_object_fill)
        x[categorical_integers_index] = x[categorical_integers_index].fillna(self.category_integer_fill)
        x[numeric_index] = x[numeric_index].fillna(self.numeric_fill)
        return x


class ModifiedStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.numeric_index = None

    def fit(self, x, y=None):
        self.numeric_index = x.select_dtypes(exclude=['object']).columns
        self.standard_scaler.fit(x[self.numeric_index])
        return self

    def transform(self, x, y=None):
        x.loc[:, self.numeric_index] = self.standard_scaler.transform(x[self.numeric_index])
        return x


def separate_columns(df):
    categorical_columns = []
    numeric_columns = []
    for column in df.columns:
        if column in list(df.select_dtypes(include=['object']).columns):
            categorical_columns.append(column)
        if column in list(df.select_dtypes(exclude=['object']).columns):
            numeric_columns.append(column)
    return categorical_columns, numeric_columns


def convert_feature_columns(df):
    categorical_columns, numeric_columns = separate_columns(df)
    tf_numeric_feature_columns = []
    tf_categorical_feature_columns = []
    for column in numeric_columns:
        column_name = tf.feature_column.numeric_column(column)
        tf_numeric_feature_columns.append(column_name)
    for column in categorical_columns:
        vocabulary_list = df[column].unique().tolist()
        column_name = tf.feature_column.categorical_column_with_vocabulary_list(column, vocabulary_list)
        tf_categorical_feature_columns.append(column_name)
    return tf_numeric_feature_columns, tf_categorical_feature_columns


def indicator_deep_column(tf_categorical_feature_columns):
    tf_categorical_feature_column_indicators = []
    for column in tf_categorical_feature_columns:
        column_indicator = tf.feature_column.indicator_column(column)
        tf_categorical_feature_column_indicators.append(column_indicator)
    return tf_categorical_feature_column_indicators


def wide_deep_columns(df):
    tf_numeric_feature_columns, tf_categorical_feature_columns = convert_feature_columns(df)
    deep_column_indicators = indicator_deep_column(tf_categorical_feature_columns)
    base_columns = tf_categorical_feature_columns
    # categories types with 0.3-0.7 cor
    crossed_columns = []
    wide_columns = base_columns + crossed_columns
    deep_columns = tf_numeric_feature_columns + deep_column_indicators
    return {'wide_columns': wide_columns, 'deep_columns': deep_columns}


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class WideAndDeepNeuralNetEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name=None, save_checkpoints_secs=None, keep_checkpoint_max=None, dnn_hidden_units=None,
                 dnn_activation_fn=None, dnn_dropout=None, test_size=None, train_batch_size=None, num_threads=None,
                 eval_batch_size=None, train_max_steps=None, eval_steps=None, start_delay_secs=None,
                 throttle_secs=None):
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation_fn = dnn_activation_fn
        self.dnn_dropout = dnn_dropout
        self.test_size = test_size
        self.train_batch_size = train_batch_size
        self.num_threads = num_threads
        self.eval_batch_size = eval_batch_size
        self.train_max_steps = train_max_steps
        self.eval_steps = eval_steps
        self.start_delay_secs = start_delay_secs
        self.throttle_secs = throttle_secs
        self.run_config = tf.estimator.RunConfig(model_dir=MODEL_DIRECTORY + model_name,
                                                 save_checkpoints_secs=save_checkpoints_secs,
                                                 keep_checkpoint_max=keep_checkpoint_max)
        self.wide_and_deep_columns = None
        self.evaluation_metrics = None
        self.exported_model_directory = None

    def fit(self, x, y=None):
        self.wide_and_deep_columns = wide_deep_columns(x)
        estimator = tf.estimator.DNNLinearCombinedClassifier(linear_feature_columns=
                                                             self.wide_and_deep_columns['wide_columns'],
                                                             dnn_feature_columns=
                                                             self.wide_and_deep_columns['deep_columns'],
                                                             dnn_hidden_units=self.dnn_hidden_units,
                                                             dnn_activation_fn=self.dnn_activation_fn,
                                                             dnn_dropout=self.dnn_dropout,
                                                             config=self.run_config)
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=self.test_size, random_state=42)
        train_input_fn = tf.estimator.inputs.pandas_input_fn(train_x, train_y, batch_size=self.train_batch_size,
                                                             num_threads=self.num_threads, shuffle=True)
        eval_input_fn = tf.estimator.inputs.pandas_input_fn(val_x, val_y, batch_size=self.eval_batch_size,
                                                            shuffle=False)
        estimator.train(train_input_fn, max_steps=self.train_max_steps)
        self.evaluation_metrics = estimator.evaluate(eval_input_fn, steps=self.eval_steps)

        # export model
        feature_columns = self.wide_and_deep_columns['deep_columns']
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        self.exported_model_directory = estimator.export_savedmodel(export_dir_base=EXPORT_DIRECTORY + self.model_name,
                                                                    serving_input_receiver_fn=export_input_fn)
        logging.info("Model exported to path - %s", self.exported_model_directory)
        return self

    def predict(self, x, y=None):
        prediction_data_ids = unpickle_object_without_xcom_pull('prediction_data_ids')
        with tf.Session() as sess:
            # load the saved model
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                       self.exported_model_directory)

            # get the predictor , refer tf.contrib.predictor
            predictor = tf.contrib.predictor.from_saved_model(self.exported_model_directory)

            prediction_output_file = open(PREDICTION_OUTPUT_PATH + 'output.csv', 'w')

            # Write Header for CSV file
            prediction_output_file.write("SK_ID_CURR, TARGET, probability")
            prediction_output_file.write('\n')

            # Create a feature dictionary for each record
            for row, cust_id in zip(x.itertuples(index=False, name=None), prediction_data_ids):
                feature_dict = dict.fromkeys(x.columns)
                for column, element in zip(x.columns, row):
                    if isinstance(element, str):
                        feature_dict[column] = _bytes_feature(value=element.encode())
                    if isinstance(element, int):
                        feature_dict[column] = _float_feature(value=int(element))
                    if isinstance(element, float):
                        feature_dict[column] = _float_feature(value=float(element))

                # Prepare model input for row
                model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))

                model_input = model_input.SerializeToString()
                output_dict = predictor({"inputs": [model_input]})

                # Positive label = 1
                label_index = np.argmax(output_dict['scores'])
                prediction_output_file.write(str(cust_id))
                prediction_output_file.write(',')
                prediction_output_file.write(str(label_index))
                prediction_output_file.write(',')
                prediction_output_file.write(str(output_dict['scores'][0][label_index]))
                prediction_output_file.write('\n')
        prediction_output_file.close()


def create_pipeline(model_name, dnn_hidden_units, dnn_dropout):
    pipeline = Pipeline([
        ('feature_selector', ModifiedFeatureSelector(missing_threshold=0.6, correlation_threshold=0.95,
                                                     task='classification', eval_metric='auc',
                                                     cumulative_importance=0.99)),
        ('fill_na', FillNA(category_object_fill='etc', category_integer_fill=0, numeric_fill=0)),
        ('standard_scaler', ModifiedStandardScaler()),
        ('estimator',
         WideAndDeepNeuralNetEstimator(model_name=model_name, save_checkpoints_secs=300, keep_checkpoint_max=3,
                                       dnn_hidden_units=dnn_hidden_units, dnn_activation_fn=tf.nn.relu,
                                       dnn_dropout=dnn_dropout, test_size=0.25, train_batch_size=128,
                                       num_threads=1, eval_batch_size=5000, train_max_steps=10000,
                                       eval_steps=10, start_delay_secs=240, throttle_secs=600))
    ])
    return pipeline


def wrangle_training_data(**kwargs):
    ti = kwargs['ti']

    raw_training_data = xcom_pull_and_unpickle_object('raw_training_data', 'load_raw_training_data', ti)

    training_labels = raw_training_data['TARGET']
    training_data = raw_training_data.drop(columns=['TARGET', 'SK_ID_CURR'])

    wrangled_training_data = {'training_data_wrangled': training_data,
                              'training_data_labels': training_labels}

    pickle_object_with_xcom_push(wrangled_training_data, 'wrangled_training_data', ti)


def wrangle_unseen_data(**kwargs):
    ti = kwargs['ti']

    raw_unseen_prediction_data = xcom_pull_and_unpickle_object('raw_unseen_prediction_data',
                                                               'load_raw_unseen_prediction_data', ti)
    prediction_data_ids = raw_unseen_prediction_data['SK_ID_CURR']
    wrangled_prediction_data = raw_unseen_prediction_data.drop(columns='SK_ID_CURR')

    pickle_object_with_xcom_push(wrangled_prediction_data, 'wrangled_data_for_prediction', ti)
    pickle_object_without_xcom_push(prediction_data_ids, 'prediction_data_ids')


def train_wide_and_deep_neural_net_model_1(**kwargs):
    ti = kwargs['ti']
    wrangled_training_data = xcom_pull_and_unpickle_object('wrangled_training_data', 'wrangle_training_data', ti)

    training_data_wrangled = wrangled_training_data["training_data_wrangled"]
    training_data_labels = wrangled_training_data["training_data_labels"]

    wide_and_deep_neural_net_model_1 = create_pipeline(model_name='model_1', dnn_hidden_units=[500, 150, 50],
                                                       dnn_dropout=0.5)
    wide_and_deep_neural_net_model_1.fit(training_data_wrangled, training_data_labels)

    pickle_object_with_xcom_push(wide_and_deep_neural_net_model_1, 'wide_and_deep_neural_net_model_1', ti)


def train_wide_and_deep_neural_net_model_2(**kwargs):
    ti = kwargs['ti']
    wrangled_training_data = xcom_pull_and_unpickle_object('wrangled_training_data', 'wrangle_training_data', ti)

    training_data_wrangled = wrangled_training_data["training_data_wrangled"]
    training_data_labels = wrangled_training_data["training_data_labels"]

    wide_and_deep_neural_net_model_2 = create_pipeline(model_name='model_2', dnn_hidden_units=[400, 100, 40],
                                                       dnn_dropout=0.6)
    wide_and_deep_neural_net_model_2.fit(training_data_wrangled, training_data_labels)

    pickle_object_with_xcom_push(wide_and_deep_neural_net_model_2, 'wide_and_deep_neural_net_model_2', ti)


def select_best_model(**kwargs):
    ti = kwargs['ti']

    wide_and_deep_neural_net_model_1 = xcom_pull_and_unpickle_object('wide_and_deep_neural_net_model_1',
                                                                     'train_wide_and_deep_neural_net_model_1', ti)
    wide_and_deep_neural_net_model_2 = xcom_pull_and_unpickle_object('wide_and_deep_neural_net_model_2',
                                                                     'train_wide_and_deep_neural_net_model_2', ti)

    model_list = [
        wide_and_deep_neural_net_model_1,
        wide_and_deep_neural_net_model_2
    ]

    max_score = 0
    best_model_index = 0

    for i in range(len(model_list)):
        auc = model_list[i].named_steps['estimator'].evaluation_metrics['auc']
        if auc > max_score:
            max_score = auc
            best_model_index = i

    pickle_object_with_xcom_push(model_list[best_model_index], 'best_model', ti)


def predict_from_unseen_data(**kwargs):
    ti = kwargs['ti']

    best_model = unpickle_object_without_xcom_pull('best_model')
    wrangled_data_for_prediction = xcom_pull_and_unpickle_object('wrangled_data_for_prediction',
                                                                 'wrangle_unseen_data', ti)

    predictions_from_unseen_data = best_model.predict(wrangled_data_for_prediction)

    pickle_object_with_xcom_push(predictions_from_unseen_data, 'predictions_from_unseen_data', ti)


def notify(**kwargs):
    pass  # change DAG task to EmailOperator when ready to send email notifications.


def store_predictions_as_i_tags(**kwargs):
    ti = kwargs['ti']

    predictions_from_unseen_data = xcom_pull_and_unpickle_object('predictions_from_unseen_data',
                                                                 'predict_from_unseen_data', ti)

    # (future) store as i-tags in database...
    # TODO implement this function
