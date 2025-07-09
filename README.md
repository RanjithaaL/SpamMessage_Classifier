# Spam Message Classifier

The **Spam Message Classifier** is a machine learning application that predicts whether a given text message is spam or legitimate (ham). This project demonstrates the use of **Natural Language Processing (NLP)** techniques and **classification algorithms** to solve a real-world problem of filtering unwanted messages.

This project highlights practical skills in data cleaning, feature engineering, machine learning modeling, and building user-friendly interfaces for ML solutions.

---

## Features

- Classifies text messages as **spam** or **ham**
- Uses **TF-IDF vectorization** to convert text to numeric features
- Trained with machine learning models like **Naive Bayes**
- Provides **instant predictions** via an interactive web UI
- Implements text preprocessing steps:
  - Lowercasing
  - Removing special characters
  - Stop word removal
  - Tokenization and stemming
- Displays model performance metrics like accuracy, precision, recall, and confusion matrix
- Lightweight and efficient — suitable for quick real-time predictions

---

## 🧠 Technical Overview

### NLP Pipeline

- **Text Cleaning:** Remove punctuation, symbols, and noise
- **Lowercasing:** Ensure uniformity of tokens
- **Stopword Removal:** Reduce noise from frequent words
- **Stemming/Lemmatization:** Normalize words to their root forms
- **Vectorization:** Transform text into numeric vectors using **TF-IDF**

### Model

- Trained primarily with **Multinomial Naive Bayes**, a popular choice for text classification tasks
- Achieves high accuracy in distinguishing spam from ham
- Evaluated using:
  - Confusion matrix
  - Classification report (precision, recall, F1-score)

---

### Dataset

This project uses the popular **SMS Spam Collection Dataset**, containing **5,574 messages** labeled as spam or ham. Each row contains:
- Label (spam/ham)
- Message text

The dataset was sourced from UCI Machine Learning Repository and is widely used in spam detection experiments.

---

### Project Structure

- spam_classifier.ipynb # Jupyter notebook with code and analysis
- app.py # Streamlit app for live predictions
- spam.csv # SMS Spam Collection Dataset
- requirements.txt # Project dependencies
- README.md # Project documentation


---

### Learnings & Contribution

Through this project, I:
- Applied **NLP techniques** for text preprocessing
- Practiced **vectorization methods** like TF-IDF
- Developed a **classification pipeline** for text data
- Built an **interactive ML app** with Streamlit
- Improved understanding of evaluating ML models for real-world deployment

All parts of this project — from data analysis to model building and app deployment — were implemented independently as part of my learning journey in machine learning and NLP.

---

### Future Improvements

- Explore ensemble classifiers for improved accuracy
- Deploy the app to a cloud platform for public use
- Add support for other languages
- Implement visualization for word clouds or feature importances

---

### Author

**RANJITHA LAKSHMINARAYAN**  

---
