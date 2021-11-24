from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import datasets
import numpy
import time
import os, sys
import argparse

DATASET_NAME = "Sklearn Wine"
FEATURE_ENG_PIPELINE_NAME = "Sklearn Standard Scalar"
PREDICTION_TYPE = "Multiclass"
DATASET_SRC = "sklearn.datasets"
MODEL_NAME = "OneVsRestClassifier(XGBoostClassifier)"

def get_data():

  """
  Load sklearn wine dataset, and scale features to be zero mean, unit variance.
  One hot encode labels (3 classes), to be used by sklearn OneVsRestClassifier.
  """

  data = datasets.load_wine()
  X = data["data"]
  y = data["target"]

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  enc = OneHotEncoder()
  Y = enc.fit_transform(y[:, numpy.newaxis]).toarray()

  return (X_scaled, Y)

def evaluate_xgboost_model(X, y,
                           number_of_cross_val_folds=5,
                           max_depth=6,
                           learning_rate=0.3,
                           min_split_loss=0):
    classifier = OneVsRestClassifier(XGBClassifier(
        objective = "binary:logistic",
        max_depth =    max_depth,
        learning_rate = learning_rate,
        min_split_loss = min_split_loss,
        use_label_encoder=False,
        verbosity = 0
    ))
    cv_accuracies = cross_val_score(classifier, X, y, cv=number_of_cross_val_folds)
    return numpy.mean(cv_accuracies)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--min_split_loss', type=float, required=True)
    parser.add_argument('--saved_path', type=str, required=True)
    return parser.parse_args(args)

def main(params):
    (features, labels) = get_data()
    args = dict(X=features,
                y=labels,
                max_depth=params.max_depth,
                learning_rate=params.learning_rate,
                min_split_loss=params.min_split_loss)

    mean_accuracy = evaluate_xgboost_model(**args)
    model_path = os.path.join(params.saved_path, "chk000000")
    with open(model_path, 'w') as f:
        f.write("Fake Model")
    print(mean_accuracy)

if __name__ == '__main__':
    params = parse_args(sys.argv[1:])
    main(params)
