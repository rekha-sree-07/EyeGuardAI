
import json
import math
import logging
from pathlib import Path

import joblib
import pandas as pd

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


logger = logging.getLogger(__name__)

BASE_DIR = Path(settings.BASE_DIR).parent
MODEL_PATH = (
    BASE_DIR / "backend" / "ml_models" / "eyeguard_model.joblib"
)

CLASS_NAMES = {
    0: "Healthy",
    1: "Glaucoma",
    2: "Glaucoma suspect",
}

DEFAULT_THRESHOLD = 0.70
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0


def load_saved_model():
    """Load the trained EyeGuard AI model."""
    return joblib.load(MODEL_PATH)


def parse_json(request):
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, JsonResponse(
            {"error": "Invalid JSON body."}, status=400
        )
    return data, None


def parse_threshold(data):
    raw = data.get("threshold", DEFAULT_THRESHOLD)

    if isinstance(raw, bool):
        raise ValueError("Threshold must be a number from 0 to 1.")

    try:
        threshold = float(raw)
    except (TypeError, ValueError):
        raise ValueError("Threshold must be a number from 0 to 1.")

    if not math.isfinite(threshold) or not (
        MIN_THRESHOLD <= threshold <= MAX_THRESHOLD
    ):
        raise ValueError("Threshold must be between 0 and 1.")

    return threshold


def prepare_sample(record, feature_names):
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object.")

    missing = [name for name in feature_names if name not in record]
    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    values = {}
    for name in feature_names:
        raw = record[name]

        if raw is None:
            values[name] = float("nan")
            continue

        if isinstance(raw, bool):
            raise ValueError(f"{name} must be a number or null.")

        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number or null.")

        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number.")

        values[name] = number

    return values


def predict_record(model, feature_names, record, threshold):
    values = prepare_sample(record, feature_names)
    sample = pd.DataFrame([values], columns=feature_names)

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    probability_map = {
        str(int(label)): round(float(probability), 4)
        for label, probability in zip(model.classes_, probabilities)
    }

    confidence = float(max(probabilities))

    return {
        "predicted_class": prediction,
        "class_name": CLASS_NAMES.get(prediction, "Unknown"),
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "probabilities": probability_map,
        "threshold": threshold,
        "threshold_percent": round(threshold * 100, 2),
        "review_recommended": confidence < threshold,
        "notice": (
            "Research prototype only; not a medical diagnosis. "
            "Confidence is not verified prediction accuracy."
        ),
    }


@csrf_exempt
@require_POST
def predict_eye_disease(request):
    """Existing single-record prediction endpoint."""
    data, error = parse_json(request)
    if error:
        return error

    if not isinstance(data, dict):
        return JsonResponse(
            {"error": "Send a JSON object."}, status=400
        )

    try:
        threshold = parse_threshold(data)
        saved = load_saved_model()
        result = predict_record(
            saved["model"], saved["feature_names"], data, threshold
        )
        return JsonResponse(result)

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except FileNotFoundError:
        return JsonResponse(
            {"error": "Saved ML model file was not found."},
            status=500,
        )
    except Exception:
        logger.exception("EyeGuard AI prediction failed.")
        return JsonResponse(
            {"error": "Prediction failed. Check the server terminal."},
            status=500,
        )


@csrf_exempt
@require_POST
def predict_batch(request):
    """Predict multiple records and optionally evaluate verified labels."""
    data, error = parse_json(request)
    if error:
        return error

    if not isinstance(data, dict):
        return JsonResponse(
            {"error": "Send a JSON object."}, status=400
        )

    records = data.get("records")
    if not isinstance(records, list) or len(records) < 2:
        return JsonResponse(
            {"error": "Provide at least two records in 'records'."},
            status=400,
        )

    if len(records) > 100:
        return JsonResponse(
            {"error": "A maximum of 100 records is allowed per request."},
            status=400,
        )

    try:
        threshold = parse_threshold(data)
        saved = load_saved_model()
        model = saved["model"]
        feature_names = saved["feature_names"]

        results = []
        for index, record in enumerate(records):
            result = predict_record(
                model, feature_names, record, threshold
            )
            result["record_id"] = (
                record.get("record_id", f"Record {index + 1}")
                if isinstance(record, dict)
                else f"Record {index + 1}"
            )
            results.append(result)

        response = {
            "results": results,
            "record_count": len(results),
            "threshold": threshold,
            "threshold_percent": round(threshold * 100, 2),
            "low_confidence_count": sum(
                item["review_recommended"] for item in results
            ),
            "notice": (
                "Research prototype only; not a medical diagnosis. "
                "Confidence is not verified prediction accuracy."
            ),
        }

        # Calculate accuracy only when verified labels are provided.
        actual_labels = data.get("actual_labels")
        if actual_labels is not None:
            if (
                not isinstance(actual_labels, list)
                or len(actual_labels) != len(results)
            ):
                return JsonResponse(
                    {
                        "error": (
                            "'actual_labels' must contain one verified "
                            "class code per record."
                        )
                    },
                    status=400,
                )

            normalized_labels = []
            for label in actual_labels:
                if isinstance(label, bool) or label not in (0, 1, 2):
                    return JsonResponse(
                        {
                            "error": (
                                "Each actual label must be 0, 1, or 2."
                            )
                        },
                        status=400,
                    )
                normalized_labels.append(int(label))

            correct = sum(
                item["predicted_class"] == actual
                for item, actual in zip(results, normalized_labels)
            )

            response["evaluation"] = {
                "correct_predictions": correct,
                "total_predictions": len(results),
                "accuracy_percent": round(
                    correct / len(results) * 100, 2
                ),
                "note": (
                    "Accuracy is meaningful only when actual labels "
                    "are independently verified and the records form "
                    "an appropriate evaluation set."
                ),
            }

        return JsonResponse(response)

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except FileNotFoundError:
        return JsonResponse(
            {"error": "Saved ML model file was not found."},
            status=500,
        )
    except Exception:
        logger.exception("EyeGuard AI batch prediction failed.")
        return JsonResponse(
            {"error": "Batch prediction failed. Check the server terminal."},
            status=500,
        )
