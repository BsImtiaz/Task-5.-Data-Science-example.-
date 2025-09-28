Consumer Complaint Classification Project
Project Overview
This project implements a multi-class text classification system to categorize consumer complaints into different product categories. The system uses machine learning and natural language processing techniques to automatically classify complaint narratives, helping organizations streamline their complaint handling processes.

Problem Statement
-The goal is to build a robust classification model that can automatically categorize consumer complaints into one of four main product categories:
-Credit reporting, credit repair services, or other personal consumer reports
-Debt collection
-Consumer Loan
-Mortgage

Dataset
This project uses consumer complaint data containing narrative descriptions and corresponding product categories. The dataset is processed and sampled for efficient model training.

Technical Implementation
Libraries and Dependencies
# Core Data Science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Text Processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D

# Big Data Processing
import dask.dataframe as dd

Project Structure
1. Data Loading & Preprocessing
-Load consumer complaint data using Dask for efficient big data handling
-Convert CSV to Parquet format for faster loading
-Handle missing values and data type conversions
-Create target mapping for classification categories

2. Text Preprocessing
-Text cleaning (lowercasing, punctuation removal, digit removal)
-Stopword removal using NLTK
-Lemmatization for word normalization
-TF-IDF vectorization with unigrams and bigrams

3. Model Training
-Multiple classifier implementations:
-Logistic Regression
-Linear SVM
-Naive Bayes
-Random Forest
-Train-test split with stratification
-Performance evaluation using accuracy and classification reports

4. Model Evaluation & Visualization
-Confusion matrix analysis
-Model comparison charts
-Interactive prediction interface

Installation and setup
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow dask

Data Setup:
-Mount Google Drive
-Place the complaints.csv file in the designated directory
-The script automatically converts CSV to Parquet for optimized performance

NLTK Downloads:
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')

Usage
- Training the Model
- Making Predictions
- Interactive Mode

Results and Visualizations
- Confusion Matrix
- Model Matrix

Key Features
-Big Data Handling: Uses Dask for efficient processing of large datasets
-Multiple Models: Implements and compares 4 different ML algorithms
-Text Preprocessing: Comprehensive NLP pipeline for text cleaning
-Interactive Interface: User-friendly prediction system
-Visual Analytics: Confusion matrices and model comparison charts
-Scalable Architecture: Can handle large volumes of complaint data

Future Enhancements
-Deep learning models (LSTM, Transformers) for improved accuracy
-Real-time prediction API
-Integration with complaint management systems
-Multi-language support
-Sentiment analysis integration
-Automated response generation

Notes
-The current implementation uses a 5% sample of the data for training
-Linear SVM shows the best balance of performance and interpretability
-Text preprocessing significantly impacts model performance
-The system can be easily extended to include more complaint categories

Contributing
-Feel free to contribute to this project by:
-Adding new machine learning models
-Improving text preprocessing techniques
-Enhancing the visualization components
-Optimizing performance for larger datasets
