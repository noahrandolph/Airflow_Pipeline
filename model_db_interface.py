import logging
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, func, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Uncomment the below lines and execute this file to create tables in model_db database
# engine = create_engine("postgresql://postgres@localhost:5432/model_db")
# Session = sessionmaker()
# Session.configure(bind=engine)
# Base.metadata.create_all(engine)


PICKLE_PATH = '/Users/garb55/airflow/dags/intern_machine_learning_framework/pickled_mock_data/'


def create_session():
    engine = create_engine("postgresql://postgres@localhost:5432/model_db")
    Session = sessionmaker()
    Session.configure(bind=engine)
    return Session()


class InferredTagMeta(Base):
    __tablename__ = 'inferred_tag_meta'
    tag_id = Column(String, primary_key=True)
    tag_name = Column(String)
    problem_type = Column(String)
    selection_criterion = Column(String)


class InferredTagDagInfo(Base):
    __tablename__ = 'inferred_tag_dag_info'
    dag_id = Column(String, primary_key=True)
    tag_id = Column(String, ForeignKey('inferred_tag_meta.tag_id'))
    dag_args = Column(String)
    dag_purpose = Column(String)


class SourceDataSet(Base):
    __tablename__ = 'source_dataset'
    source_dataset_id = Column(String, primary_key=True)
    target_yrmn = Column(String, unique=True)
    dag_id = Column(String, ForeignKey('inferred_tag_dag_info.dag_id'))
    run_id = Column(String, unique=True)
    execution_date = Column(String, unique=True)
    task_id = Column(String)
    file_path = Column(String)
    ip_address = Column(String)
    created_date = Column(String)
    is_valid = Column(String)


class WrangledDataSet(Base):
    __tablename__ = 'wrangled_dataset'
    wrangled_dataset_id = Column(String, primary_key=True)
    target_yrmn = Column(String, unique=True)
    dag_id = Column(String)
    run_id = Column(String)
    execution_date = Column(String)
    task_id = Column(String)
    source_dataset_id = Column(String, ForeignKey('source_dataset.source_dataset_id'))
    file_path = Column(String)
    ip_address = Column(String)
    created_date = Column(String)
    is_valid = Column(String)


class InferredTagDagRunState(Base):
    __tablename__ = 'inferred_tag_dag_run_state'
    target_yrmn = Column(String, primary_key=True)
    dag_id = Column(String)
    run_id = Column(String)
    execution_date = Column(String)
    state = Column(String)
    created_date = Column(String)
    message = Column(String)
    channel = Column(String)
    recipients = Column(String)


class ModelFeatureImportance(Base):
    __tablename__ = 'model_feature_importance'
    predictive_model_id = Column(String, ForeignKey('predictive_model.predictive_model_id'), primary_key=True)
    feature_importance = Column(String)


class ModelEvaluation(Base):
    __tablename__ = 'model_evaluation'
    predictive_model_id = Column(String, ForeignKey('predictive_model.predictive_model_id'), primary_key=True)
    metric_name = Column(String, primary_key=True)
    metric_value = Column(String)


class PredictiveModel(Base):
    __tablename__ = 'predictive_model'
    predictive_model_id = Column(String, primary_key=True)
    target_yrmn = Column(String, unique=True)
    dag_id = Column(String)
    run_id = Column(String)
    execution_date = Column(String)
    task_id = Column(String)
    wrangled_dataset_id = Column(String, ForeignKey('wrangled_dataset.wrangled_dataset_id'))
    model_info = Column(String)
    hyperparameters = Column(String)
    file_path = Column(String)
    ip_address = Column(String)
    created_date = Column(String)
    is_valid = Column(String)


class SelectedModel(Base):
    __tablename__ = 'selected_model'
    predictive_model_id = Column(String, ForeignKey('predictive_model.predictive_model_id'), primary_key=True)
    target_yrmn = Column(String, unique=True)
    dag_id = Column(String)
    run_id = Column(String)
    execution_date = Column(String)
    task_id = Column(String)
    model_info = Column(String)
    hyperparameters = Column(String)


def initialize_model_db(**kwargs):
    session = create_session()

    inferred_tag_dag_run_state = InferredTagDagRunState(target_yrmn=kwargs['ts'],
                                                        dag_id=str(kwargs['dag']),
                                                        run_id=kwargs['run_id'],
                                                        execution_date=str(kwargs['execution_date']),
                                                        state='unknown',
                                                        created_date=kwargs['ds'],
                                                        message='no_message',
                                                        channel='no_channel',
                                                        recipients='no_recipients')
    session.add(inferred_tag_dag_run_state)
    session.commit()


def store_source_dataset_meta(source_dataset_id, file_name, ip_address, created_date, is_valid, kwargs):
    session = create_session()

    file_path = PICKLE_PATH + file_name + '.pkl'

    source_dataset = SourceDataSet(source_dataset_id=source_dataset_id,
                                   target_yrmn=kwargs['ts'],
                                   dag_id=str(kwargs['dag']),
                                   run_id=kwargs['run_id'],
                                   execution_date=str(kwargs['execution_date']),
                                   task_id=kwargs['task_instance_key_str'],
                                   file_path=file_path,
                                   ip_address=ip_address,
                                   created_date=created_date,
                                   is_valid=is_valid)
    session.add(source_dataset)
    session.commit()


def store_wrangled_dataset_meta(wrangled_dataset_id, source_dataset_id, file_name, ip_address, created_date, is_valid,
                                kwargs):
    session = create_session()

    file_path = PICKLE_PATH + file_name + '.pkl'

    wrangled_dataset = WrangledDataSet(wrangled_dataset_id=wrangled_dataset_id,
                                       target_yrmn=kwargs['ts'],
                                       dag_id=str(kwargs['dag']),
                                       run_id=kwargs['run_id'],
                                       execution_date=str(kwargs['execution_date']),
                                       task_id=kwargs['task_instance_key_str'],
                                       source_dataset_id=source_dataset_id,
                                       file_path=file_path,
                                       ip_address=ip_address,
                                       created_date=created_date,
                                       is_valid=is_valid)
    session.add(wrangled_dataset)
    session.commit()


def store_predictive_model_meta(predictive_model_id, wrangled_dataset_id, hyperparameters,
                                file_name, ip_address, created_date, is_valid, kwargs):
    session = create_session()

    file_path = PICKLE_PATH + file_name + '.pkl'

    predictive_model = PredictiveModel(predictive_model_id=predictive_model_id,
                                       target_yrmn=file_name + kwargs['ts'],
                                       dag_id=str(kwargs['dag']),
                                       run_id=kwargs['run_id'],
                                       execution_date=str(kwargs['execution_date']),
                                       task_id=kwargs['task_instance_key_str'],
                                       wrangled_dataset_id=wrangled_dataset_id,
                                       model_info=file_name,
                                       hyperparameters=str(hyperparameters),
                                       file_path=file_path,
                                       ip_address=ip_address,
                                       created_date=created_date,
                                       is_valid=is_valid)
    session.add(predictive_model)
    session.commit()


def store_model_feature_importance(predictive_model_id, feature_importance):
    session = create_session()

    model_feature_importance = ModelFeatureImportance(predictive_model_id=predictive_model_id,
                                                      feature_importance=feature_importance)
    session.add(model_feature_importance)
    session.commit()


def store_model_evaluation(predictive_model_id, metric_name, metric_value):
    session = create_session()

    model_evaluation = ModelEvaluation(predictive_model_id=predictive_model_id,
                                       metric_name=metric_name,
                                       metric_value=metric_value)
    session.add(model_evaluation)
    session.commit()


def store_selected_model_meta(predictive_model_id, file_name, hyperparameters, kwargs):
    session = create_session()

    model_meta = SelectedModel(predictive_model_id=predictive_model_id,
                               target_yrmn=file_name + kwargs['ts'],
                               dag_id=str(kwargs['dag']),
                               run_id=kwargs['run_id'],
                               execution_date=str(kwargs['execution_date']),
                               task_id=kwargs['task_instance_key_str'],
                               model_info=file_name,
                               hyperparameters=str(hyperparameters))
    session.add(model_meta)
    session.commit()