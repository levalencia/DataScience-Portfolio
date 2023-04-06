Data Science Projects
===============

## Deep Learning, PyTorch, Transformers or  PHuggingFace Projects

[Building a TinyVGG Model from Scratch to Classify The Simpsons Characters with PyTorch 2.0](https://github.com/levalencia/DataScience-Portfolio/blob/main/SimpsonsClassifier/Pytorch%20-%20Simpsons%20Multi%20Classifier.ipynb)

If you're a fan of The Simpsons and interested in deep learning, you're in for a treat. In this portfolio project, we will be building a TinyVGG model from scratch using PyTorch to classify The Simpsons characters from images. This project will showcase how to build a custom dataset and train a model.

The Simpsons has been entertaining audiences for over 30 years, and with over 700 characters, it's hard to keep track of them all. With the help of deep learning, we can train a model that can recognize each character in a matter of seconds. This project will demonstrate how to preprocess the data, build a custom dataset, create and train the model, and evaluate its performance.

We will start by discussing what deep learning is and why it's becoming increasingly popular in various industries. We will then dive into the details of building a TinyVGG model from scratch, a small version of the VGG model, which is known for its simplicity and effectiveness. We will also learn how to use transfer learning, a popular technique that allows us to use a pre-trained model's weights as a starting point for our model.

Throughout this project, we will use PyTorch, a popular deep learning library known for its simplicity and ease of use. We will use PyTorch's custom dataset functionality to load and preprocess our data, and we will use its built-in optimization and loss functions to train our model.

[Fine-tuning DistilBERT with your own dataset for multi-classification task](https://github.com/levalencia/DataScience-Portfolio/blob/main/FineTuningDistilbert/HateSpeechFineTuningWithDistilberta.ipynb)

HuggingFace and the transformers library has made it very easily for us to avoid training Large Language Models, instead we can re-use existing models, by just donwloading them from the HF Hub and then with Pytorch and Transfomers API Fine tune it for your own specific task with your own specific labels. The end result a great model which used State of The Art Pretrained model but fit to your needs.  In this case I selected DistilBert and fine tuned it for hate speech or offensive tweets detection.  In this notebook I start from the very beginning by introducting NLP Concepts, and to until the end when the model is saved to disk, loaded and then used for inference.

## Data Cleaning and Data Preparation


[Real State Transactions Per Municipality](RealStateTransactionsPerMunicipality/Real%20State%20Analysis.ipynb)

The government has given us a dataset with aggregated data of real state transactions per municipality, per quarter, per semester and year. Our job is to describe the data, clean it, wrangle it and/or prepare it for further processing in an ML forecasting project or for Data StoryTelling.



## Classification
  

[Wine Quality Classification Problem](WineClassification/WineClassificationProject.ipynb)

We have a wine dataset with information about chemical ingredients, ph, and a quality score from 0 to 10. We want to classify a wine as good or bad, the idea is that good wines are scored 7 or above, all the rest are bad wines.



## Recommender Systems


[Recommender system for  internal tranings](RecommenderSystemsTraining/TrainingRecomendations.ipynb)

The idea of this project is easy, we have internal trainings created by our company employees, we have also external trainings that we take via pluralsight, udemy or any other platform like coursera, and we have employees which take those internal or external trainings. Employees have some attributes, like department, language, skills, etc. All those attributes need to be taken into account into our recommender system.

For example: if you are a new employee with only 1 year of experience in Data Science (feature), and in the skills(feature) you have listed Statistics, Machine Learning, but another person in the company, has 10 years of experience in DataScience, with similar skills, and if that person took "Advanced machine learning specialization" in coursera. The recommender system would be able to predict a.k.a recommend this training to the new employee.




## Regression

[Real State Price Prediction](RealStatePricePrediction/Demo.ipynb)

We are required to build a model to predict house prices in the Belgium Real State Market, the idea is that when users want to buy a new house, they can compare the listed price with the model prediction and check if it the prices are similar and take a decision.


## Time Series Forecasting

[Forecasting Energy Inflation in Belgium](ForecastEnergyInflationBelgium/Sarimax.ipynb)

Belgium Gas and Energy Prices skyrocketed last year, and recently we received the news that price has lowered from 300 Euros per MWH to 60 Euros per MWH. With this project I took the last years data for Energy Inflation and forecasted the next 4 months to see if our pockets will finally breathe a bit.


