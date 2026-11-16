#predict.py

import.joblib
import pandas as pd 
from sklearn.datasets import load_breast_cancer
#step 1:Load the saved model
def load_model(model_path="path="best_model.joblib"):
 model=joblib.load(model_path)
 return model
 #step 2: Create a reusable prediction function
 def make_prediction(model,input-data):

 prediction = model.predict(input_data)

 probabilites = model.predict_proba(input_data)

 result = "Maligant" if prediction[0]==0else "benign"
 return{
   "prediction":result,
   "probabilites":{
    "malignant":probabilites[0][0]'
    "benign":probabilites[0][1]
    }
    }

    #step 3:Run a sample prediction when executed directly
    if__name__=="main__":

    model=load_brest_cancer()

    feature_names=data.feature_names

    sample_input=dict(zip(feature_names,data.data[0]))

    #Modify a few values for demonstration
    sample_input["mean radius"]=15.0
    sample_input["mean texture"]=18.0
    sample_input["mean perimeter"]=100.0

    input_df=pd.DataFrame(
      [sample_input],
      columns=feature_names
      )

      output= make_prediction(model,input_df)

      print("Model loaded successfully!")
      print("\nPrediction:",output["prediction"])

      print("Model loaded successfully!")
      print("\nPrediction:",output["prediction])

      print("\nPrediction Probabilites:")
      print("Malignant:",output["probabilites"]["malignant"]
      print("benign:",output["probabilites"]["benign]
