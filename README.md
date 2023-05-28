## VSCode with Jypyter extensions
https://code.visualstudio.com/docs/datascience/overview  
VSCode 에 Jupyter extensions 설치하고,  
해당 프로젝트를 github repo 에 저장  

```$ python3 -m venv .venv``` 을 이용해서 virtual 환경 설치   
```$ source .venv/bin/activate``` 으로 virtual 환경 실행   
```(.venv)$ pip install ipykernel```으로 관련 python 모듈 설치  

> '$ python -m ipykernel install --user' 을 사용하는 경우 jupyter notebook (lab) 에서 ikernel 을.할때,  
설정오류가 발생할 수 있다. 되도록 사용하지 말자 (pip 로 설치)

```
[E 08:57:38.101 NotebookApp] Failed to run command:
    ['/Users/myoungjunesung/pyproject/data/ipynb/.venv/bin/python3', '-m', 'ipykernel_launcher', '-f', '/Users/myoungjunesung/Library/Jupyter/runtime/kernel-5cc8a27b-46fd-45a7-8eee-395c9d2b9f92.json']
```
</br>

## Jupyter lab, Notebook
https://github.com/jupyterlab/jupyterlab
```bash
$ pip install ipykernel
# or $ pip install jupyter (or notebook)
$ pip install jupyterlab
$ jupyter lab  
# or $ jupyter notebook 
```
git extension : ```pip install --upgrade jupyterlab jupyterlab-git```

</br>

## Other Jupyter 
> 1. [Anaconda](https://www.anaconda.com/) 을 설치해서 Jupyter Notebook 사용  (Local)  
> 2. [google colab](https://colab.research.google.com), [Kaggle](https://www.kaggle.com), [dacon](https://dacon.io/) 에서 제공하는 cloud 커널 을 사용  (Cloud)  

</br>

## 참고 사이트

[판다스 시리즈](https://passwd.tistory.com/entry/Python-Pandas-Series-1)  
[데이터스쿨](https://datascienceschool.net/intro.html)  
[데이터매님](https://www.datamanim.com/dataset/99_pandas/pandasMain.html) 
[판다스 10min 영문](https://pandas.pydata.org/docs/user_guide/10min.html#)  
[판단스 10min 한글](https://dandyrilla.github.io/2017-08-12/pandas-10min/)  
[판다스 함수 설명 한글](https://runebook.dev/ko/docs/pandas/-index-#DataFrame)  
[나도코딩 데이터분석](https://nadocoding.tistory.com/90)  
[함수 찾아보기](https://wikidocs.net/book/7188)  
[모의고사 tjd229](http://tjd229.tistory.com/category/Computer%20Science/Data%20Science)

</br>

## ML Model

scipy.stats.norm / ppf 

sklearn.model_selection.train_test_split   
sklearn.model_selection.GridSearchCV  
sklearn.model_selection.KFold  

sklearn.compose.ColumnTransformer(transformers=(['encoder', OneHotEncoder(), [2]]),remainder)  
sklearn.compose.make_column_transformer((OneHotEncoder(),['label']),(),remainder)

statsmodels.stats.outliers_influence.variance_inflation_factor(df,i) for i in df_cols_count

sklearn.preprocessing.OneHotEncoder(drop='first')   
sklearn.preprocessing.StandardScaler, MinMaxScaler(Feature_range(start, stop+1))  
sklearn.preprocessing.PolynomialFeatures   # 다항회귀   

sklearn.linear_model.LinearRegression   
sklearn.linear_model.LogisticRegression  
sklearn.linear_model.Perceptron  
sklearn.linear_model.SGDClassifier(max_iter,eta0)        # 경사하강법   
sklearn.linear_model.SGDRegressor  

sklearn.svm.SVC, LinearSVC   

sklearn.ensemble.RandomForestClassifier   

sklearn.neighbors.KNeighborsClassifier(n_neighbors)    # KNN   
sklearn.neighbors.KNeighborsRegressor  

sklearn.naive_bayes.GaussianNB   

sklearn.tree.DecisionTreeClassifier   
sklearn.tree.DecisionTreeRegressor    

sklearn.cluster.KMeans (n_cluster, init, n_init)   

sklearn.metrics.mean_absolute_error, mean_squared_error, r2_core   

sklearn.metrics.f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, roc_auc_score, classification_report  

sklearn.metrics.silhouette_score (X, lables)   

</br>

## modeling
```
X_train = df[[ , ]]   : 독립변수 학습 데이터  (series.to_frame())
X_test = df[[ , ]]    : 독립변수 테스트 데이터  (series.to_frame())
y_train = df[ ]       : 종속변수 학습 데이터  
y_test = df[ ]        : 종속변수 테스트 데이터  (실제결과)

y_pred                : 종속변수 예측 데이터  
# pred_list = [ ]     : 종속변수 예측 데이터 리스트  


my_model = Given_ML_Model()       # ML 모델 적용  
my_model.fit(X_train, y_train)    # 학습데이터로 학습  - fit_predict(), fit_transform(), inverse_trasform()  
y_pred = my_model.predict(X_test) # 학습된 모델(my_model)으로 테스트 데이터 (독립변수, X_test)의 결과(종속변수, y_pred) 예측값  
                                  # coef_, intercept_, label_, inertia_  
my_model.score(X_test, y_test)    # 학습된 모델(my_model)의 test 평가  

# pred_list.append(y_pred)  
# XXX_score(y_test,y_pred)        # 학습된 모델(my_model)의 실제값 (y_test) 대비 예측(y_pred)결과 평가

```

## for

1. 개별 row,column 별 (row 한개씩) 계산은 : itertuples, apply 사용 (axis = 0,1)
2. row,column 구룹별 (slicing) 계산은 : for i_value in unique_values_list (unique한 값을 공유하는 slicing 그룹)