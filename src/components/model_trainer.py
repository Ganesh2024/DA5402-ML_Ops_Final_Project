import os
import sys
import mlflow
import mlflow.sklearn
from typing import Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

# ── MLflow tracking URI (SQLite backend — reliable in MLflow 3.x UI) ─────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT   = "VehicleInsurance"
# ─────────────────────────────────────────────────────────────────────────────


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config          = model_trainer_config

    # ── Model definitions ─────────────────────────────────────────────────────
    def _get_candidate_models(self) -> Dict[str, object]:
        """
        Returns a dict of candidate models to evaluate.
        All use the same random_state for fair comparison.
        n_estimators kept at 100 for all three to keep training time reasonable.
        """
        rs = self.model_trainer_config._random_state
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=self.model_trainer_config._max_depth,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                criterion=self.model_trainer_config._criterion,
                random_state=rs
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=self.model_trainer_config._max_depth,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                criterion=self.model_trainer_config._criterion,
                random_state=rs
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=self.model_trainer_config._max_depth,
                min_samples_split=self.model_trainer_config._min_samples_split,
                min_samples_leaf=self.model_trainer_config._min_samples_leaf,
                random_state=rs
            ),
        }
    # ─────────────────────────────────────────────────────────────────────────

    def _train_and_evaluate(self, model, x_train, y_train, x_test, y_test):
        """
        Trains a single model and returns metrics.
        """
        model.fit(x_train, y_train)
        y_pred    = model.predict(x_test)
        accuracy  = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        return model, accuracy, f1, precision, recall

    def _log_run_to_mlflow(self, model_name, model, params, accuracy, f1, precision, recall):
        """
        Logs a single model run to MLflow — params, metrics, and model artifact.
        """
        with mlflow.start_run(run_name=model_name):
            # Log hyperparameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metric("accuracy",  accuracy)
            mlflow.log_metric("f1_score",  f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall",    recall)

            # Log model artifact
            mlflow.sklearn.log_model(model, model_name)

            logging.info(
                f"[{model_name}] MLflow logged — "
                f"accuracy: {accuracy:.4f}, f1: {f1:.4f}, "
                f"precision: {precision:.4f}, recall: {recall:.4f}"
            )

    def get_model_object_and_report(self, train: np.array, test: np.array):
        """
        Trains 3 candidate models, logs each to MLflow as a separate run,
        selects the best by F1 score, and returns the winner.
        """
        try:
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test,  y_test  = test[:, :-1],  test[:, -1]
            logging.info("train-test split done.")

            # ── MLflow setup ─────────────────────────────────────────────────
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            # ─────────────────────────────────────────────────────────────────

            candidates = self._get_candidate_models()
            results    = {}  # model_name -> (model, accuracy, f1, precision, recall)

            for model_name, model in candidates.items():
                logging.info(f"Training {model_name}...")
                print(f"  >> Training {model_name}...")

                trained, accuracy, f1, precision, recall = self._train_and_evaluate(
                    model, x_train, y_train, x_test, y_test
                )

                # Build param dict for this model
                params = {
                    "model_type":        model_name,
                    "n_estimators":      100,
                    "max_depth":         self.model_trainer_config._max_depth,
                    "min_samples_split": self.model_trainer_config._min_samples_split,
                    "min_samples_leaf":  self.model_trainer_config._min_samples_leaf,
                    "random_state":      self.model_trainer_config._random_state,
                }
                if model_name != "GradientBoosting":
                    params["criterion"] = self.model_trainer_config._criterion

                self._log_run_to_mlflow(
                    model_name, trained, params,
                    accuracy, f1, precision, recall
                )

                results[model_name] = (trained, accuracy, f1, precision, recall)

            # ── Select best model by F1 score ─────────────────────────────────
            best_name = max(results, key=lambda k: results[k][2])  # index 2 = f1
            best_model, best_accuracy, best_f1, best_precision, best_recall = results[best_name]

            logging.info(f"Best model: {best_name} with F1={best_f1:.4f}")
            print(f"  >> Best model selected: {best_name} (F1={best_f1:.4f})")

            # Log a summary "Best Model" run so it's visually clear in UI
            with mlflow.start_run(run_name=f"BestModel_{best_name}"):
                mlflow.log_param("best_model",       best_name)
                mlflow.log_param("selection_metric", "f1_score")
                mlflow.log_metric("best_accuracy",   best_accuracy)
                mlflow.log_metric("best_f1_score",   best_f1)
                mlflow.log_metric("best_precision",  best_precision)
                mlflow.log_metric("best_recall",     best_recall)
                mlflow.sklearn.log_model(best_model, f"BestModel_{best_name}")
                logging.info(f"Best model summary run logged to MLflow.")
            # ─────────────────────────────────────────────────────────────────

            metric_artifact = ClassificationMetricArtifact(
                f1_score=best_f1,
                precision_score=best_precision,
                recall_score=best_recall
            )
            return best_model, metric_artifact, best_accuracy

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates model training: trains 3 models, selects best, saves to artifact.
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component — Training 3 candidate models")

            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            logging.info("train-test data loaded")

            # Train all 3 models, log to MLflow, get best
            best_model, metric_artifact, best_accuracy = self.get_model_object_and_report(
                train=train_arr, test=test_arr
            )
            logging.info("Model selection complete. Best model and metrics ready.")

            # Load preprocessor
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            logging.info("Preprocessing obj loaded.")

            # Check best model meets minimum accuracy threshold on train set
            train_accuracy = accuracy_score(
                train_arr[:, -1],
                best_model.predict(train_arr[:, :-1])
            )
            if train_accuracy < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save best model wrapped with preprocessor
            logging.info("Saving best model with preprocessor.")
            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object (preprocessing + best trained model)")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
