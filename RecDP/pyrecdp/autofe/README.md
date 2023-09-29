# Auto Feature Engineering Workflow

Auto feature engineering targets to simplify Feature engineering process with enhanced performance via parallel data processing frameworks, automated data processing pipeline and built-in domain-specific feature engineering primitives. This repository provides an end-to-end workflow that automatically analyzes the data based on data type, profiles feature distribution, generates customizable feature engineering pipelines for the data preparation and executes the pipeline parallelly with different backend engines on Intel platform.

![auto feature engineering explained](/RecDP/resources/autofe_pipeline.jpg)

Steps explained:
1. Feature profile: Analyze raw tabular dataset to infer original feature based on data type and generate FeatureList.
2. Feature engineering: Use inferred FeatureList to generate Data Pipeline in Json/Yaml File format.
3. Feature transformation: Convert Data Pipeline to executable operations and transform original features to candidate features with selected engine, currently Pandas and Spark were supported.
4. Feature Importance Estimator: perform feature importance analysis on candidate features to remove un-important features, generate the transfomred dataset that includes all finalize features that will be used for training. 

### Built-In Use Cases

| Workflow Name | Description |
| --- | --- |
| [NYC taxi fare](applications/nyc_taxi_fare/interactive_notebook.ipynb) | Fare prediction based on NYC taxi dataset |
| [Amazon Product Review](applications/amazon_product_review/interactive_notebook.ipynb) | Product recommandation based on reviews from Amazon |
| [IBM Card Transaction Fraud Detect](applications/fraud_detect/interactive_notebook.ipynb) | Recognize fraudulent credit card transactions |
| [Twitter Recsys](applications/twitter_recsys/interactive_notebook.ipynb) | Real-world task of tweet engagement prediction |
| [Outbrain](applications/outbrain_ctr/interactive_notebook.ipynb) | Click prediction for recommendation system |
| [Covid19 TabUtils](applications/covid19_tabutils/interactive_notebook.ipynb) | integration example with Tabular Utils |
| [PredictiveAssetsMaintenance](applications/predictive_assets_maintenance/interactive_notebook.ipynb) | integration example with predictive assets maintenance use case |
