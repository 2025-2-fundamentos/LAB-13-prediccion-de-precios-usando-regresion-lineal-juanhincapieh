#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    median_absolute_error,
)


def loadDatasets():
    trainFrame = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    testFrame = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return trainFrame, testFrame


def addAgeFeature(dataFrame: pd.DataFrame) -> pd.DataFrame:
    df = dataFrame.copy()
    df["Age"] = 2025 - df["Year"]
    df.drop(columns=["Year", "Car_Name"], inplace=True)
    return df


def separateFeaturesTarget(df: pd.DataFrame):
    features = df.drop(columns=["Present_Price"])
    target = df["Present_Price"]
    return features, target


def createPipeline(featureFrame: pd.DataFrame) -> Pipeline:
    categoricalFields = ["Fuel_Type", "Selling_type", "Transmission"]
    numericFields = [col for col in featureFrame.columns if col not in categoricalFields]

    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(), categoricalFields),
            ("numeric", MinMaxScaler(), numericFields),
        ]
    )

    modelPipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessing),
            ("feature_selector", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    return modelPipeline


def tuneAndFitModel(pipe: Pipeline, xTrain, yTrain):
    paramGrid = {
        "feature_selector__k": range(1, 12),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False],
    }

    gridSearch = GridSearchCV(
        estimator=pipe,
        param_grid=paramGrid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    gridSearch.fit(xTrain, yTrain)
    return gridSearch


def persistModel(trainedModel, outputPath: str = "files/models/model.pkl.gz"):
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    with gzip.open(outputPath, "wb") as f:
        pickle.dump(trainedModel, f)


def buildMetrics(yTrue, yPred, datasetLabel: str) -> dict:
    return {
        "type": "metrics",
        "dataset": datasetLabel,
        "r2": float(r2_score(yTrue, yPred)),
        "mse": float(mean_squared_error(yTrue, yPred)),
        "mad": float(median_absolute_error(yTrue, yPred)),
    }


def writeMetricsFile(rows, filePath: str = "files/output/metrics.json"):
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    with open(filePath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run():
    trainData, testData = loadDatasets()

    trainData = addAgeFeature(trainData)
    testData = addAgeFeature(testData)

    xTrain, yTrain = separateFeaturesTarget(trainData)
    xTest, yTest = separateFeaturesTarget(testData)

    pipeline = createPipeline(xTrain)
    fittedModel = tuneAndFitModel(pipeline, xTrain, yTrain)

    persistModel(fittedModel)

    trainPred = fittedModel.predict(xTrain)
    testPred = fittedModel.predict(xTest)

    trainMetrics = buildMetrics(yTrain, trainPred, "train")
    testMetrics = buildMetrics(yTest, testPred, "test")

    writeMetricsFile([trainMetrics, testMetrics])


if __name__ == "__main__":
    run()
