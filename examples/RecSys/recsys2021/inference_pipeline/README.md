# Useful scripts
* `compress_models.py` - use on pre-computed estimators to optimize them for memory and prepare them for pipeline (this is run locally, not on the GCP instance)
* `run_prediction.py` - run the inference pipeline (this is what should be running on RecSys GCP node)

Each script shows basic usage when invoked without arguments

Environment requirements:
* xgboost >=1.3.3
* pandas
* pyarrow
* scikit-learn
