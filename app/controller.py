from app.utils import WineSample
from app.services import model
import pandas as pd
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import os
import re


class WineController:
    def __init__(
        self,
    ):
        return

    def wine_predict(self, wine_sample: WineSample):
        prediction = model.predict(wine_sample.get_dataframe())
        return int(prediction)

    def bulk_wine_predict(self, file):
        df = pd.read_csv(file)
        predictions = []
        # Reading the user csv file

        for i, row in df.iterrows():
            # Create a new dataframe for the current row
            current_df = pd.DataFrame(row).T
            # Convert the dataframe to a dictionary
            current_dictionary = current_df.to_dict(orient="records")[0]
            # validate user data
            validated_schema = WineSample(current_dictionary, True)
            if validated_schema.validated == False:
                return {
                    "success": False,
                    "error": validated_schema.errors,
                    "row": i + 1,
                }
            predictions.append(
                {
                    "name": current_dictionary["name"]
                    if "name" in current_dictionary
                    else i + 1,
                    "quality": float(
                        model.predict(
                            current_df.drop(["name"], axis=1)
                            if "name" in current_df.columns and False
                            else current_df,
                            True,
                        )
                    ),
                }
            )
            predictions.sort(key=lambda pred: pred.get("quality"), reverse=True)

        return {"predictions": predictions, "success": True, "error": None, "row": None}

    def predict_from_image(self, text):
        try:
            paragraph = """The wine has a fixed Acidity : 8.2, volatile acidity of 0.57, citric acid of 0.24, residual sugar of 2.1 g/L, chlorides at 0.081 g/L, free sulfur dioxide of 15 mg/L, density of 0.9968 g/mL, pH of 3.29, sulphates at 0.62 g/L, and an alcohol content of 9.9%."""
            keywords = [
                "fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol",
            ]
            values = {}

            # matching the keywords
            for keyword in keywords:
                pattern = r"{}(?:\s*[^0-9]*)([\d.]+)".format(keyword)
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    values[keyword] = float(match.group(1))
            validated_sample = WineSample(values, True)

            missing_columns = 0
            # validate values from image
            if validated_sample.validated:
                # make prediction
                prediction = model.predict(validated_sample.get_dataframe())
            else:
                missing_columns = validated_sample.errors.__len__()
            response = {
                "success": validated_sample.validated,
                "errors": None
                if validated_sample.validated
                else validated_sample.errors,
                "missing_columns": missing_columns,
                "prediction": int(prediction) if validated_sample.validated else None,
                "text": text,
            }
            return response
        except Exception as e:
            print(e)
            response = {
                "success": False,
                "errors": {"Error": "Internal Server Error"},
                "missing_columns": None,
                "prediction": None,
                "text": None,
            }
            return response


class ImageController:
    def __init__(self):
        return

    def decode(self, image):
        image_name = secure_filename(image.filename)
        image_path = os.path.join("temp", image_name)
        image.save(image_path)

        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        os.remove(image_path)
        return text


wine_controller = WineController()
image_controller = ImageController()
