source /etc/profile.d/spark-env.sh
cd /home/vmagent/app/e2eaiok/
python setup.py install
pip install pyrecdp
sh modelzoo/dien/feature_engineering/start_spark_service.sh 
python modelzoo/dien/feature_engineering/preprocessing.py --train
python modelzoo/dien/feature_engineering/preprocessing.py --test
python modelzoo/dien/feature_engineering/preprocessing.py --inference
python -u run_e2eaiok.py --data_path /home/vmagent/app/dataset/amazon_reviews --model_name dien --no_sigopt