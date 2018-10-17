from sqlalchemy import Column, DateTime, String, Integer, ForeignKey, func, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


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


engine = create_engine("postgresql://postgres@localhost:5432/model_db")
Session = sessionmaker()
Session.configure(bind=engine)
Base.metadata.create_all(engine)


def create_session():
    engine = create_engine("postgresql://postgres@localhost:5432/model_db")
    Session = sessionmaker()
    Session.configure(bind=engine)
    return Session()


session = create_session()
inferred_tag_meta = InferredTagMeta(tag_id="000000",
                                    tag_name="missed_payment_prediction",
                                    problem_type="customer_delinquency",
                                    selection_criterion='rmse')
session.add(inferred_tag_meta)
session.commit()

session = create_session()
inferred_tag_dag_info = InferredTagDagInfo(dag_id="<DAG: mock_loan_training>",
                                           tag_id="000000",
                                           dag_args="()",
                                           dag_purpose='train_and_select_best_model')
session.add(inferred_tag_dag_info)
session.commit()