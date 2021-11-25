.. Hydro.ai documentation master file, created by
   sphinx-quickstart on Wed Nov 24 23:43:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hydro.ai
========
*Hydro.ai* is the project code name to democratize End-to-End AI on CPU. It is initiated by Intel AIA, aiming to provide simple experience for democratized END-TO-END model training leveraging Intel oneAPI aikit, Intel sigopt, and Intel Nerual Compressor.

.. note::
   This project is under active developing.

Motivation
==========
Modern end to end machine learning or deep learning system usually includes a lifecycle of data processing, feature engineering, training, inference and serving. Each different stage might has different challenges, for example, the data to be processed might be huge, and thus require a signficant amount of time to process the data and the data ETL and pre-processing time often take much more time than training. For feature engineering phase, usually numerous sets of new features need to be created, and then tested for effectiveness. For model training, one of the entry barrier is the model could be quite complex and usually requires lots of expertise to design, understand, tune and deploy the models. What makes things worse is one usually needs to repeate the experiements of the ETL, feature engineering, training and evaluation gprocess for many times on many models, to get the best model, which requires signficant amount of computational resources and is time-consuming.

Contents
========
.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: QuickStart:

   overview
   quickstart

How To Contribute
=================
.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: advanced:

   advanced
   _modules/modules
   codingstyle
   documentingstyle

