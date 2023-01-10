e2eaiok Overview
=================

End-to-End AI democratization
-----------------------------
One approach to resolve those challenges is AI democratization, which trying to make AI accessabile & affordable to every organization and every data scientist. There are a lot of things to be democratized, including:

* Data accessibility & quality, where you make the data access easier and simpler; building a data platform to democratize the data management - simplify data ingestion, data explore, processing and visulaization.
* Storage and compute platforms, instead of running the AI on hi-cost GPU, run it on democratized commodity hardware with auto-scaling.
* Algorithms - Democratize the use, development and sharing of ML & DL algorithms; reduce the engry barrier with automatic model searching, AutoML
* Model development - select the most suitalbe models for users, democratize the end to end model development
* Market place - simply how you access, use, exchange and monetization of data, algorithm, models, and outcomes

Key architecture
----------------
Below firgure showed the architecture diagram of Bluewhale.

.. image:: architecture.jpg
    :alt: e2eaiok overview

Major componments
-----------------
Here are the major componments of Bluewhale:

* RecDP - scalable data processing and feature engineering kit based on Spark and Modin
* Distributed data connector - a distirbuted data connector based on PetaStorm supporting training framework to load data from distributed filesystem, and provide enhanced capabilities like data caching, sharing.
* Smart Democratization Advisor - a human intelligence enhanced toolkit to generate sigopt recipes for Sigopt AutoML. It first generate optimized SigOpt recipes based on user choices and built-in intelligence, including optimized model parameters, optimized training framework parameters and set the training cluster environment, training framework parameters, and final target metrcis. Then it kicks off Siopt AutoML for optimization experiments to generate the best model.
* Model Compression - A model compression toolkit that supports botch train from scratch and pre-trained models, to simply the model tuning process and generate ligher models
* Inference acclerators - enhanced in memory vectors and vector recall utilits to speedup vector recall process.
* Recipes - four proof of concept workloads including DLRM, DIEN, WnD, RecSys, with ready to run notebooks to showcase the benefit of the democratized models.
* deployment kit - a container with deployment kit to have users have a quick try.
* Auto scaling - Bluewhale support autoscaling on the public cloud, users can easiliy scale the work to multiple CPU nodes.

In-Stock-Models
---------------
Currently four recommender system workloads were supported, including:
* DLRM
* DIEN
* WnD
* RecSys

Performance
-----------
ToBeAdded.
The Bluewhale E2E AI democratization kit significantly improved the performance of those models on distributed CPU cluster, reduced the performance gap of CPU to GPU from 100x to < 2x, using the same dataset and the same AUC metrics.