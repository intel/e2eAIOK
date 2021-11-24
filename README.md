# Hydro.ai

Hydro.ai is the project code name to democratize End-to-End AI on CPU. It is initiated by Intel AIA, aiming to provide simple experience for democratized END-TO-END model training leveraging Intel oneAPI aikit, Intel sigopt, and Intel Nerual Compressor.

# Introduction

Modern end to end machine learning or deep learning system usually includes a lifecycle of data processing, feature engineering, training, inference and serving. Each different stage might has different challenges, for example, the data to be processed might be huge, and thus require a signficant amount of time to process the data and the data ETL and pre-processing time often take much more time than training. For feature engineering phase, usually numerous sets of new features need to be created, and then tested for effectiveness. For model training, one of the entry barrier is the model could be quite complex and usually requires lots of expertise to design, understand, tune and deploy the models. What makes things worse is one usually needs to repeate the experiements of the ETL, feature engineering, training and evaluation gprocess for many times on many models, to get the best model, which requires signficant amount of computational resources and is time-consuming. 

# End-to-End AI democratization 

One approach to resolve those challenges is AI democratization, which trying to make AI accessabile & affordable to every organization and every data scientist. There are a lot of things to be democratized, including: 
1. Data accessibility & quality, where you make the data access easier and simpler; building a data platform to democratize the data management - simplify data ingestion, data explore, processing and visulaization. 
2. Storage and compute platforms, instead of running the AI on hi-cost GPU, run it on democratized commodity hardware with auto-scaling. 
3. Algorithms - Democratize the use, development and sharing of ML & DL algorithms; reduce the engry barrier with automatic model searching, AutoML
4. Model development - select the most suitalbe models for users, democratize the end to end model development 
5. Market place - simply how you access, use, exchange and monetization of data, algorithm, models, and outcomes 


# BlueWhale 

Bluewhale is the project to democratize AI on CPU, starting from E2E recommender systems. The strategy is to bring E2E AI to existing CPU installation base with good-enough performance and zero additional cost, it drives the AI portaion on CPU and drive CPU/GPU balance in E2E AI. As a start, Bluewhale democrate recommender systems on distributed CPU clusters through different optimization of each machine learning life cycle phases. 

# Key architecture 

Below firgure showed the architecture diagram of Bluewhale. 

![Architecture](./docs/pictures/architecture.jpg "Bluewhale Architecture")

# Major componments 

Here are the major componments of Bluewhale: 
1. RecDP -  scalable data processing and feature engineering kit based on Spark and Modin 
2. Distributed data connector - a distirbuted data connector based on PetaStorm  supporting training framework to load data from distributed filesystem, and provide enhanced capabilities like data caching, sharing. 
3. Smart Democratization Advisor - a human intelligence enhanced toolkit to generate sigopt recipes for Sigopt AutoML. It first generate optimized SigOpt recipes based on user choices and built-in intelligence, including optimized model parameters, optimized training framework parameters and set the training cluster environment, training framework parameters, and final target metrcis. Then it kicks off Siopt AutoML for optimization experiments to generate the best model. 
4. Model Compression - A model compression toolkit that supports botch train from scratch and pre-trained models, to simply the model tuning process and generate ligher models 
5. Inference acclerators - enhanced in memory vectors and vector recall utilits to speedup vector recall process. 
6. Recipes - four proof of concept workloads including DLRM, DIEN, WnD, RecSys, with ready to run notebooks to showcase the benefit of the democratized models. 
7. deployment kit - a container with deployment kit to have users have a quick try. 
8. Auto scaling - Bluewhale support autoscaling on the public cloud, users can easiliy scale the work to multiple CPU nodes. 

# In-Stock-Models

Currently four recommender system workloads were supported: including DLRM, DIEN, WnD and RecSys. The Bluewhale E2E AI democratization kit significantly improved the performance of those models on distributed CPU cluster, reduced the performance gap of CPU to GPU from 100x to < 2x, using the same dataset and the same AUC metrics. The key optimizations Bluewhale brought can be categorized as: 

# How to Use

[HOW_TO.md](docs/HOW_TO.md)


# Perforamnce 
ToBeAdded. 

# How to Contribute

[Documenting Style](https://www.sphinx-doc.org/en/master/)
[Coding Style](https://www.python.org/dev/peps/pep-0008/)
