# NLP_Employee_Performance
Predicting employee performance rating based on the feedback text that was provided.  

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


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


## Project Motivation<a name="motivation"></a>


## File Descriptions <a name="files"></a>



## Results<a name="results"></a>



## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Employee feedback data was downloaded from [Kaggle] (https://www.kaggle.com/datasets/fiodarryzhykau/employee-review).  

OrdinalClassifier() by Muhammad Assagaf was dowloaded from [Medium.com] (https://medium.com/towards-data-science/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c)

Custom LabelEncoder from [StackOverflow] (https://stackoverflow.com/questions/51308994/python-sklearn-determine-the-encoding-order-of-labelencoder)

Other acknowledgements of code leveraged via Stackoverflow are documented within the code itself. 
