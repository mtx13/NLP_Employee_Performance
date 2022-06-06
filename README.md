# NLP_Employee_Performance
Predicting employee performance rating based on the feedback text that was provided.  

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

---

## Installation <a name="installation"></a>

The following libraries were installed as part of this project:
 - Sentiment_Scoring_Analysis.ipynb
   - NLTK Vader
   - TextBlob
   - Flair
 
- EDA_Prepare_Data.ipynb
  - NLTK tokenize 
  - Flair
  - textstat


- EDA_NLP_Feedback.ipynb
  -  matplotlib
  -  seaborn
  -  re (regex)
  -  contractions
  
  
  
- BuildingPipelines-Final
  -  NLTK (word_tokenize, punkt, WordNetLemmatizer, pos_tag)
  -  sklearn preprocessing (OneHotEncoder,LabelEncoder , OrdinalEncoder, FunctionTransformer, ColumnTransformer)
  -  sklearn pipeline (Pipeline, FeatureUnion)
  -  sklearn TfidfVectorizer
  -  sklearn model ( GridSearchCV, train_test_split)
  -  xgboost (XGBClassifier)
  -  sklearn metrics (confusion_matrix, f1_score, accuracy_score)
  -  matplotlib, seaborn
  -  re (regex)
  -  contractions
  -  Flair
  -  clone 

---
## Project Motivation<a name="motivation"></a>

Writing motivations


---
## File Descriptions <a name="files"></a>

There are 4 Jupyter notebooks associated with this project and 2 data files downloaded from [Kaggle](https://www.kaggle.com/datasets/fiodarryzhykau/employee-review).  

Jupyter Notebooks:
- Sentiment_Scoring_Analysis.ipynb
- EDA_Prepare_Data.ipynb
- EDA_NLP_Feedback.ipynb
- BuildingPipelines-Final

Data Files:
The data files retrieved from Kaggle has 4 columns:
 employee name
 nine-box performance descriptor
 reviewed (was the data reviewed)
 adjusted (was the data adjusted)

- employee_review_mturk_dataset_test_v6_kaggle.csv
- employee_review_mturk_dataset_v10_kaggle.csv


---
## Results<a name="results"></a>

My results are....


---
## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Employee feedback data was downloaded from [Kaggle](https://www.kaggle.com/datasets/fiodarryzhykau/employee-review).  

OrdinalClassifier() by Muhammad Assagaf was dowloaded from [Medium.com](https://medium.com/towards-data-science/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c)

Custom LabelEncoder from [StackOverflow](https://stackoverflow.com/questions/51308994/python-sklearn-determine-the-encoding-order-of-labelencoder)

Other acknowledgements of code leveraged via Stackoverflow are documented within the code itself. 


