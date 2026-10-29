# Political Security Threat Prediction Framework

A hybrid lexicon-based and machine learning framework designed to predict political security threats in cyberspace by mining sentiments from online news platforms and social media.

## 📌 Project Overview
Monitoring online sentiments and public opinions is critical for detecting excessive emotional triggers that can lead to threats against national security. This project proposes a comprehensive framework that combines a lexicon-based sentiment analysis approach with machine learning classifiers to achieve optimal accuracy in threat detection.

* **Academic Level:** Major Project submitted in partial fulfillment of the Bachelor of Technology (B.Tech) in Computer Science - Internet of Things.
* **Institution:** Geethanjali College of Engineering and Technology (UGC Autonomous).
* **Guided By:** Ms. B. Nalini (Associate Professor).
* **Developed By:** B. Nikitha, S. Suhas, A. Sanjana (Batch of 2024).

---

## 🛠️ System Requirements & Tech Stack

### Software Requirements
* **Primary Language:** Python
* **Environment Management:** Anaconda
* **IDE/Development Environment:** Jupyter Notebook
* **Web Framework:** Flask (Python)
* **Database:** SQLite3
* **Front-End Stack:** HTML5, CSS3, JavaScript, Bootstrap 4

### Hardware Requirements (Minimum Baseline)
* **Operating System:** Windows Only
* **Processor:** Intel Core i5 and above
* **RAM:** 8 GB or greater
* **Storage Space:** 25 GB available on local disk

---

## ⚙️ Core Architecture & Modules

The system breaks down the prediction pipeline into the following systematic functional modules:
1. **Data Loading:** Imports text datasets (such as the Kaggle Dataset of Indian Politics Tweets and Reactions).
2. **Data Preprocessing & NLP:** Tokenizes sentences, cleans text features, applies stop-word filters, and runs feature extractions (Unigrams, POS Tagging, score-based lexicon features).
3. **Data Splitting:** Dynamically partitions the vector spaces into training and evaluation sets.
4. **Model Generation & Ensemble Execution:** Evaluates single models against state-of-the-art stacking methods:
   * Naive Bayes
   * Support Vector Machine (SVM)
   * Decision Tree
   * Voting Classifier (AdaBoost + Random Forest)
   * Stacking Classifier *(Achieving over 90% accuracy)*
5. **Web Portal Authentication:** Provides multi-user capability via Secure User Registration and Sign-In pages.
6. **Prediction Dashboard:** Interactive UI utilizing Flask routing where users input text blocks to produce live security polarity outputs.

---

## 🚀 Installation & Local Deployment Guide

Follow these steps to deploy and test the application on your local machine:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/political-security-threat-prediction.git
   cd political-security-threat-prediction
   ```

2. **Set Up Python Environment**
   Open your Anaconda Prompt and create a fresh virtual environment:
   ```bash
   conda create -n security-env python=3.9
   conda activate security-env
   ```

3. **Install Dependencies**
   Install the required frameworks and machine learning packages:
   ```bash
   pip install flask pandas numpy scikit-learn nltk workspace
   ```

4. **Initialize Natural Language Corpus**
   Open a Python shell within your terminal and download the required NLTK tokenizers:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   ```

5. **Launch the Application**
   Execute your primary server file:
   ```bash
   python app.py
   ```
   Open your preferred web browser and navigate to `http://127.0.0.1:5000/`.
