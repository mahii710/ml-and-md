train.py
#STEP 1 : import libraries
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklaern.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import LogistivRegression
from sklearn.neighbours import KNeighboursClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import(
 f1_score,
accuracy-score,
classificatiom_report,
confusion_matix
)

#STEP 2 Load dataset

data=load_breast_cancer()

X=pd.DataFrame(
data,data,
 columns=data.feature_names
)
y=pd.Series(data.target,name="target")

print("Dataset shape:" X.shape)
printt("\nFirst 5 rows.)
print(x.head())

print("\nTarget values:")
print(Y>value_counts())

#STEP 3 split dataset

X_train,X_train,y-test_split(
X,

Y,
test_size=0.20
random_state=42
stratify=y
)

print("\ntrainin samples:",X_train,shape[0])
print("Testing samples:",X_test.shape[0])

#STEP 4:Define classification algorithms

models={

 "Logistic Regression":LogisticRegression(
    max_iter=5000
),
"KNN" : KNeighboursClassifier(),

"Decision Tree": DecisionTreeClassifier(
   random_state=42
),

"SVM":SVC()
}

#STEP 5 : Define hyperparameter grids

param_grids=(

 "Logistic Regression":{
   "model_C":[0.1,1,10],
    "model_solver":["liblinear"]
}

"KNN":{
"model__n_neighbors":[3,5,7,9],
"model__weights":["uniform","distance"]
},

"Decision Tree":{
    "model__max_depth":[None,3,5,10],

"model__min_samples_split":[2,5,10]
},

"Random Forest":{
     "model__n_estimators":[100,200],
     "model__max_depth":[None,5,10]
     "model__min_samples_split":[2,5]
},

"SVM":{
"model__C":[0.1,1,10],
"model__kernel":["linear","rbf"],
"model__gamma":["scale","auto"]
  }
}
#STEP 6: Run GridSearchCV forevery algorithm

results =[]
best_models={}

for model_name,model in models.items():


  print("\n"+"="*60)
  print("Running:",model_name)
  print("="*60)

  pipeline=Pipeline([
   ("scaler",StandardScaler()),
("model",model)
])

grid_search = GridSearchCV(
   estimator=pipeline,
   param_grid=param_grids[model_name],
   scoring="f1",

   cv=5,
   n_jobs=-1
)
grid_search.fit(X_train, Y_train)

best_models[model_name]=grid_search.best_estimator_

results.append({
  "Model":model_name,
  "Best F1 Score":grid_search.best_score_,
  "Best Parameters":grid_search.best_params_
})

print("Best F1 Score:",
     round(grid_search.best_score_,4))

print("Best Parameters:",
    grid_search.best_params_)

#STEP7 Compare all models

results_df=pd.DataFrame(results)

results_df=results_df.sort_values(
 by="Best F1 Score",
 ascending=False
)

print("\n"+"="*60)
print("MODEL COMPARISION")
print("="*60)

print(
  results_df[
  ["Model",Best F1 Score"]
].to_string(index=False)
)

#STEP 8:Select the best model

best_model_name=results_df.iolv[0]["Model"]

best_model=best_models[best_model_name]

best_cv_f1=
