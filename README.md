Luis Valencia Data Science Portfolio
===============

## Langchain, OpenAI, Vector DB, Pinecone

[Creating a Langchain application with Streamlit, OpenAI to talk to your own text documents using Pinecone as Vector DB](https://github.com/levalencia/DataScience-Portfolio/blob/main/LangChainWithTextFile/LangChainWithTextFile.ipynb)
Discover the power of Langchain applications! In this blog post, I will explore how to create a cutting-edge Langchain application that enables you to interact with your own text documents in a conversational manner. By harnessing the capabilities of  Langchain, OpenAI, and Pinecone as a Vector DB, we'll guide you through the process of building a seamless user experience.

## Deep Learning, PyTorch, Transformers or  HuggingFace Projects

[Transfer Learning for Image Classification with PyTorch](https://github.com/levalencia/DataScience-Portfolio/blob/main/TransferLearningWithPytorch/Transfer_Learning_for_Image_Classification.ipynb)

In my previous [project](https://github.com/levalencia/DataScience-Portfolio/blob/main/SimpsonsClassifier/Pytorch%20-%20Simpsons%20Multi%20Classifier.ipynb) I created a CNN from scratch, but in the real world you would barely do that and instead rely on existing pretrained models, we will take an existing model and modify the classification head for our specific problem.


[Building a TinyVGG Model from Scratch to Classify The Simpsons Characters with PyTorch 2.0](https://github.com/levalencia/DataScience-Portfolio/blob/main/SimpsonsClassifier/Pytorch%20-%20Simpsons%20Multi%20Classifier.ipynb)

In this project I created a CNN from scratch, TinyVGG its a well known CNN architecture, and I create it in order for readers to understand all steps needed to create, train and evaluare a deep learning model from its roots.


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


