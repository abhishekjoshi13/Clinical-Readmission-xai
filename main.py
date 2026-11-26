
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PatientData


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_schema_columns():
    return [
        field.alias or name
        for name, field in PatientData.model_fields.items()
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(
        os.getenv("MODEL_PATH", "readmission_model_v1.joblib")
    )

    if not model_path.is_file():
        raise RuntimeError(
            f"Model artifact not found: {model_path}"
        )

    try:
        bundle = joblib.load(model_path)

        if not isinstance(bundle, dict):
            raise ValueError("Invalid model artifact format")

        model = bundle["model"]
        threshold = float(bundle["threshold"])
        feature_columns = bundle["feature_columns"]
        numeric_features = bundle["numeric_features"]
        model_version = str(bundle.get("model_version", "unknown"))

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid decision threshold")

        if len(feature_columns) != 44:
            raise ValueError("Unexpected model feature count")

        if len(set(feature_columns)) != len(feature_columns):
            raise ValueError("Duplicate model feature names found")

        if set(feature_columns) != set(get_schema_columns()):
            raise ValueError(
                "API schema features do not match model features"
            )

        if not isinstance(numeric_features, (list, tuple)):
            raise ValueError("Invalid numeric feature metadata")

        if not numeric_features:
            raise ValueError("Numeric feature metadata is empty")

        if len(set(numeric_features)) != len(numeric_features):
            raise ValueError("Duplicate numeric features found")

        if not set(numeric_features).issubset(set(feature_columns)):
            raise ValueError(
                "Numeric features do not match model feature columns"
            )

        if not hasattr(model, "named_steps"):
            raise ValueError("Expected a fitted sklearn pipeline")

        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        app.state.model = model
        app.state.threshold = threshold
        app.state.feature_columns = feature_columns
        app.state.numeric_features = numeric_features
        app.state.preprocessor = preprocessor
        app.state.model_version = model_version
        app.state.explainer = None

        try:
            app.state.explainer = shap.TreeExplainer(classifier)
            logger.info("SHAP explainer initialized successfully")
        except Exception:
            logger.exception(
                "SHAP initialization failed; predictions remain available"
            )

        logger.info(
            "Model version %s loaded successfully",
            model_version
        )

    except Exception as exc:
        logger.exception("Failed to load model artifacts")
        raise RuntimeError("Model initialization failed") from exc

    yield

    app.state.model = None
    app.state.explainer = None

    logger.info("Application resources released")


app = FastAPI(
    title="Clinical Readmission Risk API",
    description=(
        "Research API for estimating the probability of "
        "readmission within 30 days."
    ),
    version="1.0.0",
    lifespan=lifespan
)


allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"]
    )


@app.get("/health")
def health_check():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    return {
        "status": "healthy",
        "model_version": app.state.model_version,
        "explainability_available": (
            app.state.explainer is not None
        )
    }


def generate_explanation(processed_features):
    explainer = app.state.explainer
    preprocessor = app.state.preprocessor

    if explainer is None:
        return {
            "available": False,
            "reason": "SHAP explainer is unavailable"
        }

    try:
        if hasattr(processed_features, "toarray"):
            processed_features = processed_features.toarray()

        shap_output = explainer.shap_values(processed_features)

        if isinstance(shap_output, list):
            shap_output = (
                shap_output[1]
                if len(shap_output) > 1
                else shap_output[0]
            )

        shap_values = np.asarray(shap_output)

        if shap_values.ndim == 3:
            if shap_values.shape[-1] > 1:
                shap_values = shap_values[..., 1]
            else:
                shap_values = shap_values[..., 0]

        shap_values = np.squeeze(shap_values)

        if shap_values.ndim != 1:
            raise ValueError("Unexpected SHAP output dimensions")

        feature_names = preprocessor.get_feature_names_out()

        if len(feature_names) != len(shap_values):
            raise ValueError(
                "SHAP values and feature names do not match"
            )

        contributions = [
            {
                "feature": str(name),
                "contribution": round(float(value), 6)
            }
            for name, value in zip(feature_names, shap_values)
        ]

        top_factors = sorted(
            contributions,
            key=lambda item: abs(item["contribution"]),
            reverse=True
        )[:5]

        return {
            "available": True,
            "method": "SHAP",
            "contribution_scale": "model_output_margin",
            "top_factors": top_factors
        }

    except Exception:
        logger.exception("SHAP explanation generation failed")

        return {
            "available": False,
            "reason": "Explanation could not be generated"
        }


@app.post("/predict_risk")
def predict_risk(data: PatientData):
    model = getattr(app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available"
        )

    try:
        patient_data = data.model_dump(by_alias=True)

        input_df = pd.DataFrame(
            [patient_data],
            columns=app.state.feature_columns
        )

        input_df = input_df.map(
            lambda value: (
                np.nan
                if value is None or (
                    isinstance(value, str) and value == "?"
                )
                else value
            )
        )

        for column in app.state.numeric_features:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="raise"
            )

        probabilities = model.predict_proba(input_df)

        classes = list(model.classes_)
        positive_class_index = classes.index(1)

        probability = float(
            probabilities[0][positive_class_index]
        )

        threshold = app.state.threshold
        prediction = int(probability >= threshold)

        processed_features = (
            app.state.preprocessor.transform(input_df)
        )

        explanation = generate_explanation(processed_features)

        return {
            "model_version": app.state.model_version,
            "readmission_probability": round(probability, 4),
            "decision_threshold": round(threshold, 6),
            "predicted_class": prediction,
            "predicted_label": (
                "readmission_within_30_days"
                if prediction == 1
                else "not_readmitted_within_30_days"
            ),
            "above_threshold": bool(prediction),
            "explanation": explanation,
            "notice": (
                "Research prototype only. This output is not a "
                "clinical diagnosis or treatment recommendation."
            )
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediction request failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed"
        )



       
 
      
