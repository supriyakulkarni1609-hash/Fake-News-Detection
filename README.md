# Fake News Detection

A **Fake News Detection system** using Python, NLP techniques, and machine learning algorithms.  
This project classifies news articles as **True** or **False** using preprocessed datasets and trained models.

---

## Project Overview

- Preprocess news text using tokenization, stemming, and other NLP techniques.  
- Extract features with **Bag-of-Words, n-grams, TF-IDF**.  
- Train multiple classifiers: **Logistic Regression, Random Forest, Naive Bayes, SVM**.  
- Evaluate models with **F1 score** and **confusion matrix**, then select the best performing model.  
- Predict news credibility using the trained model via `prediction.py` or `simple_fake_news.py`.

---

## Dataset

- Simplified version of the **LIAR dataset**:  
  | Original Label           | New Label |
  |-------------------------|-----------|
  | True, Mostly-true, Half-true | True      |
  | Barely-true, False, Pants-fire | False     |

- Files included in this repository:  
  `train.csv`, `test.csv`, `valid.csv`  

- Original `.tsv` files are in the `liar` folder (for reference).

---

## Project Structure

```

Fake_News_Detection-master/
├── DataPrep.py            # Data preprocessing and EDA
├── FeatureSelection.py    # Feature extraction & selection
├── classifier.py          # ML classifiers & model tuning
├── prediction.py          # Final prediction script
├── simple_fake_news.py    # Simple prediction interface
├── final_model.sav        # Trained model
├── model.pkl              # Alternative saved model
├── train.csv              # Training dataset
├── test.csv               # Testing dataset
├── valid.csv              # Validation dataset
├── final-fnd.ipynb        # Notebook (optional)
├── README.md              # Project description
└── CSS / front-end files  # Optional UI assets

````

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Internship-Tasks.git
````

2. **Install dependencies**
   Using Python:

```bash
pip install -U scikit-learn numpy scipy pandas
```

Using Anaconda:

```bash
conda install -c scikit-learn
conda install -c anaconda numpy scipy pandas
```

---

## How to Run

**Using Python with PATH set or Anaconda:**

```bash
cd /path/to/Fake_News_Detection-master
python prediction.py
```

**Without PATH:**

```bash
C:/path/to/python.exe /path/to/Fake_News_Detection-master/prediction.py
```

* Enter a news headline when prompted.
* Output: **Predicted class (True/False)** and **probability score**.

---

## Features & Workflow

1. **DataPrep.py** – Tokenization, stemming, missing value checks, EDA.
2. **FeatureSelection.py** – Bag-of-Words, n-grams, TF-IDF features.
3. **classifier.py** – Train multiple classifiers, perform GridSearchCV, evaluate models.
4. **prediction.py** – Uses final trained Logistic Regression model for classifying user input.

---

## Performance

* **Logistic Regression** and **Random Forest** models evaluated using learning curves and F1 score.

<p align="center">
  <img width="500" src="https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/LR_LCurve.PNG">
  <img width="500" src="https://github.com/nishitpatel01/Fake_News_Detection/blob/master/images/RF_LCurve.png">
</p>

---

## Future Improvements

* Include more features: **POS tagging, Word2Vec, topic modeling**
* Increase dataset size
* Explore deep learning models for better accuracy

---
**Author**
Supriya Kulkarni
