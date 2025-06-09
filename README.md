

---

```markdown
#  Sentiment Analysis on IMDB Reviews  
*A symphony of code and emotion*

---

##  Overview

This project unveils a machine learning model that deciphers the emotional tone behind movie reviews from the IMDB dataset. Whether a film evoked tears of joy or sighs of regret, our model listens to every word, learns from sentiment, and predicts with poetic precision.

Using the **Naive Bayes** algorithm and **TF-IDF vectorization**, we transform raw human expressions into quantified insights.

---

##  Project Structure

```

sentiment-analysis-imdb/


├── sentiment\_analysis.py     # Core Python script


├── IMDB Dataset.csv          # Dataset file (not included in repo)


├── requirements.txt          # All required Python packages


└── README.md                 # Project documentation

````

---

##  Features

-  Preprocessing: Cleans text using regex, lowercasing, and NLTK stopwords  
-  Vectorization: Applies TF-IDF to transform text to numeric form  
-  Model Training: Trains a Naive Bayes classifier using scikit-learn  
-  Evaluation: Displays accuracy and classification metrics  
-  Custom Testing: Analyze your own movie review and predict its sentiment

---

##  Getting Started

###  Prerequisites

Ensure you have Python 3.7+ installed.

###  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
````

Install the dependencies:

```bash
pip install -r requirements.txt
```

###  Prepare the Dataset

Download the IMDB dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
Place the CSV file in the same directory as `sentiment_analysis.py` and rename it to:

```
IMDB Dataset.csv
```

---

##  Run the Project

```bash
python sentiment_analysis.py
```

The script will:

* Load and preprocess the data
* Train the model
* Print accuracy and a classification report
* Predict the sentiment of a sample review

---

##  Example Output

```
Accuracy: 0.87
Classification Report:
              precision    recall  f1-score   support
...
New Review: the movie great loved
Predicted Sentiment: Positive
```

---

##  Tech Stack

* Python 3
* Pandas & NumPy
* NLTK (Natural Language Toolkit)
* Scikit-learn

---

##  requirements.txt

```txt
pandas
numpy
nltk
scikit-learn
```

Install using:

```bash
pip install -r requirements.txt
```

---

##  License

This project is open-source and free to use under the [MIT License](LICENSE).

---

##  Acknowledgements

* [IMDB Movie Review Dataset – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* The Python and ML community for open libraries and knowledge

---

##  Author

**Sabarivasan** – Feel free to  star the repo and contribute!

---

