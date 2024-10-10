from flask import Blueprint, request
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons.logger import Logger
from deepface.modules.verification import (
    find_cosine_distance,
    find_distance,
    l2_normalize,
    find_euclidean_distance,
    find_threshold,
)
import time

logger = Logger()

blueprint = Blueprint("routes", __name__)


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    obj = service.represent(
        img_path=img_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification


@blueprint.route("/verify-embeddings", methods=["POST"])
def verify_embeddings():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    embedding1 = input_args.get("embedding1")
    embedding2 = input_args.get("embedding2")

    if embedding1 is None:
        return {"message": "you must pass embedding1 input"}

    if embedding2 is None:
        return {"message": "you must pass embedding2 input"}

    tic = time.time()

    # --------------------------------

    distance_metric = input_args.get("distance_metric", "cosine")

    if distance_metric == "cosine":
        distance = find_cosine_distance(embedding1, embedding2)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(embedding1, embedding2)
    elif distance_metric == "euclidean_l2":
        distance = find_euclidean_distance(l2_normalize(embedding1), l2_normalize(embedding2))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    # -------------------------------

    model_name = input_args.get("model_name", "VGG-Face")
    threshold = find_threshold(model_name, distance_metric)

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "similarity_metric": input_args.get("distance_metric", "cosine"),
        "time": round(toc - tic, 2),
    }

    logger.debug(resp_obj)

    return {"success": bool(resp_obj)}


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    demographies = service.analyze(
        img_path=img_path,
        actions=input_args.get("actions", ["age", "gender", "emotion", "race"]),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(demographies)

    return demographies
