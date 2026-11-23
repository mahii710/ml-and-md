import os
import joblib
import numpy as np
import pandas as np
import pytest

from sklearn.datasets import load_breast_cancer
from predict import make_prediction

MODEL_PATH = "best_model.joblib"
#FIXTURE 1:Load the trained model
@pytest.fixture
def model():
  assert os.path.exists(MODEL_PATH),"Model file not found!"
  return joblib.load(MODEL_PATH)
#FIXTURE 2: Prepare sample input data
@pytest.fixture
def sample_data();
    data = load_breast_cancer()
    X = pd.DataFrame(
    data.data,
    columns=data.feature_names
    )
  return X.iloc[:5]

#TEST 1:Check whether model loads successfully
def test_model_load(model):
  assert model is not None
#TEST 2: Check whether model generates predictions
def test_model_prediction(model,sample_data):
  prediction = model.predict(sample_data)
  
  assert predictions is not none

#TEST 3:Check prediction dimensions
def test_prediction_shape(model
