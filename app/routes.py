from flask import jsonify, request
from app.utils import WineSample
from app import app
from flask import request, jsonify
from app.controller import wine_controller, image_controller


@app.route("/api/wine/predict_from_image", methods=["POST"])
def convert_image_to_string():
    if "image" not in request.files:
        return jsonify({"success": False, "error": {"file": "No image provided"}}), 400
    try:
        # decode the image into text
        text = image_controller.decode(request.files["image"])
    except Exception as e:
        return (
            jsonify({"success": False, "error": {"Error": "Internal server error"}}),
            500,
        )

    response = wine_controller.predict_from_image(text)
    return jsonify(response), 200 if response["success"] else 500


@app.route("/api/wine/predict", methods=["POST"])
def wine_predict():
    # validate user data
    validated_schema = WineSample(dict(request.form))
    if validated_schema.validated == False:
        return jsonify({"success": False, "error": validated_schema.errors}), 400
    try:
        # predict outcome
        prediction = wine_controller.wine_predict(validated_schema)
        return jsonify({"success": True, "prediction": prediction, "error": None}), 200
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "prediction": None,
                    "error": {"Server error": "Error"},
                }
            ),
            500,
        )


@app.route("/api/wine/bulk_predict", methods=["POST"])
def bulk_wine_predict():
    # check if given file
    if "csv_file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
        # Check if the file is a CSV file
    if not request.files["csv_file"].filename.endswith(".csv"):
        return jsonify({"error": "File must be a CSV file"}), 400
    # return predictions
    try:
        response = wine_controller.bulk_wine_predict(request.files["csv_file"])
        return jsonify(response), 200 if response["success"] else 400
    except:
        return (
            jsonify({"success": False, "error": {"file-type": "could not parse file"}}),
            500,
        )


if __name__ == "__main__":
    app.run(debug=True)
