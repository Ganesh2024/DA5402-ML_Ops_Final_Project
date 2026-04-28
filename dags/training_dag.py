"""
DA5402 MLOps Final Project — Vehicle Insurance Cross-Sell Prediction
Airflow Training Pipeline DAG

Pipeline stages (in order):
    data_ingestion → data_validation → data_transformation
    → model_training → model_evaluation → model_pusher

Failure behaviour:
    If any task fails, all downstream tasks are automatically marked
    'upstream_failed' (Airflow default trigger_rule='all_success').
    The DAG turns red and stops — no stage runs after a failure.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Project root must be on sys.path so src.* imports work ───────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

# ── Default DAG arguments ─────────────────────────────────────────────────────
default_args = {
    "owner":            "Ganesh Mula",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry":   False,
}
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# Task functions
# All imports happen INSIDE functions so each task gets config_entity freshly
# with PIPELINE_TIMESTAMP already set in the environment.
# ═══════════════════════════════════════════════════════════════════════════════

def task_data_ingestion(**context):
    """
    Stage 1: Pull data from MongoDB Atlas, split into train/test CSV files.
    Generates the shared PIPELINE_TIMESTAMP and pushes it to XCom so all
    downstream tasks use the same artifact directory.
    """
    import os
    from datetime import datetime

    ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.environ["PIPELINE_TIMESTAMP"] = ts
    context["ti"].xcom_push(key="pipeline_timestamp", value=ts)

    import importlib
    import src.entity.config_entity as ce
    importlib.reload(ce)

    from src.components.data_ingestion import DataIngestion
    from src.logger import logging

    logging.info(f"[Airflow] Data Ingestion started | timestamp: {ts}")

    ingestion = DataIngestion(data_ingestion_config=ce.DataIngestionConfig())
    artifact  = ingestion.initiate_data_ingestion()

    logging.info(f"[Airflow] Data Ingestion complete: {artifact}")

    return {
        "trained_file_path": artifact.trained_file_path,
        "test_file_path":    artifact.test_file_path,
    }


def task_data_validation(**context):
    """
    Stage 2: Validate schema — column names, types, counts.
    Fails loudly if validation fails so downstream tasks are blocked.
    """
    import os
    import importlib

    ts = context["ti"].xcom_pull(key="pipeline_timestamp", task_ids="data_ingestion")
    os.environ["PIPELINE_TIMESTAMP"] = ts

    import src.entity.config_entity as ce
    importlib.reload(ce)

    from src.entity.artifact_entity import DataIngestionArtifact
    from src.components.data_validation import DataValidation
    from src.logger import logging

    ingestion_result = context["ti"].xcom_pull(task_ids="data_ingestion")
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=ingestion_result["trained_file_path"],
        test_file_path=ingestion_result["test_file_path"],
    )

    logging.info("[Airflow] Data Validation started")

    validation = DataValidation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_config=ce.DataValidationConfig()
    )
    artifact = validation.initiate_data_validation()

    if not artifact.validation_status:
        raise ValueError(f"Data validation failed: {artifact.message}")

    logging.info(f"[Airflow] Data Validation complete: {artifact}")

    return {
        "validation_status":           artifact.validation_status,
        "message":                     artifact.message,
        "validation_report_file_path": artifact.validation_report_file_path,
    }


def task_data_transformation(**context):
    """
    Stage 3: Feature engineering — gender encoding, id drop, dummies,
             scaling (StandardScaler + MinMaxScaler), SMOTEENN.
    """
    import os
    import importlib

    ts = context["ti"].xcom_pull(key="pipeline_timestamp", task_ids="data_ingestion")
    os.environ["PIPELINE_TIMESTAMP"] = ts

    import src.entity.config_entity as ce
    importlib.reload(ce)

    from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
    from src.components.data_transformation import DataTransformation
    from src.logger import logging

    ingestion_result  = context["ti"].xcom_pull(task_ids="data_ingestion")
    validation_result = context["ti"].xcom_pull(task_ids="data_validation")

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=ingestion_result["trained_file_path"],
        test_file_path=ingestion_result["test_file_path"],
    )
    validation_artifact = DataValidationArtifact(
        validation_status=validation_result["validation_status"],
        message=validation_result["message"],
        validation_report_file_path=validation_result["validation_report_file_path"],
    )

    logging.info("[Airflow] Data Transformation started")

    transformation = DataTransformation(
        data_ingestion_artifact=ingestion_artifact,
        data_transformation_config=ce.DataTransformationConfig(),
        data_validation_artifact=validation_artifact,
    )
    artifact = transformation.initiate_data_transformation()

    logging.info(f"[Airflow] Data Transformation complete: {artifact}")

    return {
        "transformed_object_file_path": artifact.transformed_object_file_path,
        "transformed_train_file_path":  artifact.transformed_train_file_path,
        "transformed_test_file_path":   artifact.transformed_test_file_path,
    }


def task_model_training(**context):
    """
    Stage 4: Train 3 candidate models (RandomForest, ExtraTrees, GradientBoosting),
             log each to MLflow, select best by F1, save as model.pkl.
    """
    import os
    import importlib

    ts = context["ti"].xcom_pull(key="pipeline_timestamp", task_ids="data_ingestion")
    os.environ["PIPELINE_TIMESTAMP"] = ts

    import src.entity.config_entity as ce
    importlib.reload(ce)

    from src.entity.artifact_entity import DataTransformationArtifact
    from src.components.model_trainer import ModelTrainer
    from src.logger import logging

    transformation_result = context["ti"].xcom_pull(task_ids="data_transformation")
    transformation_artifact = DataTransformationArtifact(
        transformed_object_file_path=transformation_result["transformed_object_file_path"],
        transformed_train_file_path=transformation_result["transformed_train_file_path"],
        transformed_test_file_path=transformation_result["transformed_test_file_path"],
    )

    logging.info("[Airflow] Model Training started")

    trainer  = ModelTrainer(
        data_transformation_artifact=transformation_artifact,
        model_trainer_config=ce.ModelTrainerConfig()
    )
    artifact = trainer.initiate_model_trainer()

    logging.info(f"[Airflow] Model Training complete: {artifact}")

    return {
        "trained_model_file_path": artifact.trained_model_file_path,
        "f1_score":                artifact.metric_artifact.f1_score,
        "precision_score":         artifact.metric_artifact.precision_score,
        "recall_score":            artifact.metric_artifact.recall_score,
    }


def task_model_evaluation(**context):
    """
    Stage 5: Compare new model vs production model in S3.
             Raises exception if new model is not accepted — stops model pusher.
    """
    import os
    import importlib

    ts = context["ti"].xcom_pull(key="pipeline_timestamp", task_ids="data_ingestion")
    os.environ["PIPELINE_TIMESTAMP"] = ts

    import src.entity.config_entity as ce
    importlib.reload(ce)

    from src.entity.artifact_entity import (
        DataIngestionArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
    )
    from src.components.model_evaluation import ModelEvaluation
    from src.logger import logging

    ingestion_result = context["ti"].xcom_pull(task_ids="data_ingestion")
    training_result  = context["ti"].xcom_pull(task_ids="model_training")

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=ingestion_result["trained_file_path"],
        test_file_path=ingestion_result["test_file_path"],
    )
    metric_artifact = ClassificationMetricArtifact(
        f1_score=training_result["f1_score"],
        precision_score=training_result["precision_score"],
        recall_score=training_result["recall_score"],
    )
    trainer_artifact = ModelTrainerArtifact(
        trained_model_file_path=training_result["trained_model_file_path"],
        metric_artifact=metric_artifact,
    )

    logging.info("[Airflow] Model Evaluation started")

    evaluator = ModelEvaluation(
        model_eval_config=ce.ModelEvaluationConfig(),
        data_ingestion_artifact=ingestion_artifact,
        model_trainer_artifact=trainer_artifact,
    )
    artifact = evaluator.initiate_model_evaluation()

    if not artifact.is_model_accepted:
        raise ValueError(
            f"Model not accepted. Improvement ({artifact.changed_accuracy:.4f}) "
            f"below threshold. Model pusher will not run."
        )

    logging.info(f"[Airflow] Model Evaluation complete: {artifact}")

    return {
        "is_model_accepted":  artifact.is_model_accepted,
        "changed_accuracy":   artifact.changed_accuracy,
        "s3_model_path":      artifact.s3_model_path,
        "trained_model_path": artifact.trained_model_path,
    }


def task_model_pusher(**context):
    """
    Stage 6: Upload accepted model to AWS S3 bucket for production serving.
    Only runs if model_evaluation task succeeded.
    """
    import os
    import importlib

    ts = context["ti"].xcom_pull(key="pipeline_timestamp", task_ids="data_ingestion")
    os.environ["PIPELINE_TIMESTAMP"] = ts

    import src.entity.config_entity as ce
    importlib.reload(ce)

    from src.entity.artifact_entity import ModelEvaluationArtifact
    from src.components.model_pusher import ModelPusher
    from src.logger import logging

    evaluation_result = context["ti"].xcom_pull(task_ids="model_evaluation")
    evaluation_artifact = ModelEvaluationArtifact(
        is_model_accepted=evaluation_result["is_model_accepted"],
        changed_accuracy=evaluation_result["changed_accuracy"],
        s3_model_path=evaluation_result["s3_model_path"],
        trained_model_path=evaluation_result["trained_model_path"],
    )

    logging.info("[Airflow] Model Pusher started")

    pusher   = ModelPusher(
        model_evaluation_artifact=evaluation_artifact,
        model_pusher_config=ce.ModelPusherConfig()
    )
    artifact = pusher.initiate_model_pusher()

    logging.info(f"[Airflow] Model Pusher complete: {artifact}")

    return {
        "bucket_name":   artifact.bucket_name,
        "s3_model_path": artifact.s3_model_path,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DAG definition
# ═══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="vehicle_insurance_training_pipeline",
    description="End-to-end ML training pipeline for vehicle insurance cross-sell prediction",
    default_args=default_args,
    schedule=None,                   # manual trigger only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "vehicle-insurance", "training"],
) as dag:

    t1_data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=task_data_ingestion,
    )

    t2_data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=task_data_validation,
    )

    t3_data_transformation = PythonOperator(
        task_id="data_transformation",
        python_callable=task_data_transformation,
    )

    t4_model_training = PythonOperator(
        task_id="model_training",
        python_callable=task_model_training,
    )

    t5_model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=task_model_evaluation,
    )

    t6_model_pusher = PythonOperator(
        task_id="model_pusher",
        python_callable=task_model_pusher,
    )

    # ── Pipeline dependency chain ─────────────────────────────────────────────
    # trigger_rule defaults to 'all_success':
    # if any task fails → all downstream tasks = upstream_failed (blocked)
    t1_data_ingestion >> t2_data_validation >> t3_data_transformation >> \
    t4_model_training >> t5_model_evaluation >> t6_model_pusher
