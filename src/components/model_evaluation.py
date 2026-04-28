import sys
import mlflow
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
from src.entity.s3_estimator import Proj1Estimator

# ── Same SQLite backend as model_trainer ─────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config      = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact  = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Fetches the current production model from S3.
        Returns None if no model exists yet.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path  = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)
            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    # ── Data preprocessing helpers (mirror of DataTransformation) ────────────
    def _map_gender_column(self, df):
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year":  "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df

    def _drop_id_column(self, df):
        """Drop id and _id columns if present."""
        logging.info("Dropping 'id' column(s) if present")
        for col in ["_id", "id"]:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Compares new trained model against production model on test set.
        Logs evaluation results to MLflow under the same experiment.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for new trained model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info("Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(
                    f"F1_Score — Production Model: {best_model_f1_score:.4f}, "
                    f"New Trained Model: {trained_model_f1_score:.4f}"
                )

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_accepted = trained_model_f1_score > tmp_best_model_score
            difference  = trained_model_f1_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_accepted,
                difference=difference
            )

            # ── MLflow: log evaluation results in same experiment ─────────────
            try:
                mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
                mlflow.set_experiment("VehicleInsurance")

                with mlflow.start_run(run_name="Model_Evaluation"):
                    mlflow.log_metric("new_model_f1",        trained_model_f1_score)
                    mlflow.log_metric("production_model_f1", tmp_best_model_score)
                    mlflow.log_metric("f1_improvement",      difference)
                    mlflow.log_param("model_accepted",       str(is_accepted))
                    mlflow.log_param("acceptance_threshold", self.model_eval_config.changed_threshold_score)

                logging.info("MLflow evaluation metrics logged successfully.")
            except Exception as mlflow_err:
                # MLflow logging failure must NOT break the main pipeline
                logging.warning(f"MLflow logging skipped during evaluation: {mlflow_err}")
            # ── end MLflow ────────────────────────────────────────────────────

            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates model evaluation and returns ModelEvaluationArtifact.
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
