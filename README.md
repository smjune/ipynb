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
[모의고사 tjd229](http://tjd229.tistory.com/category/Computer%20Science/Data%20Science)  
[파이선 기초](https://wikidocs.net/book/1553)  
[판다스 기초](https://wikidocs.net/book/7188)  

</br>

## ML Model

scipy.stats.binom, poisson, expon / pmf, ppf 
scipy.stats.t, ttest_1samp, ttest_ind, ttest_relf, 
scipy.stats.chi2, chi2_contingency
scipy.stats.f, f_oneway
scipy.stats.levene, bartlett, fligner              # 등분산 검정
scipy.stats.skew

statsmodels.api.GML
statsmodels.api.OLS(y_train,X_train).fit()                    # 절편 미 포함  
statsmodels.formula.api.ols(y_col~X_col_1+X_col2, df).fit()   # 절편 포함되어 있음 , 자동 인코딩
statsmodels.stats.anova.anova_lm
statsmodels.stats.outliers_influence.variance_inflation_factor(df,i) for i in df_cols_count

imblearn.over_sampling.SMOTE               # 샘플링

sklearn.model_selection.train_test_split  
sklearn.model_selection.cross_val_score 
sklearn.model_selection.GridSearchCV       # 하이퍼 파라미터 튜닝
sklearn.model_selection.KFold  

sklearn.compose.ColumnTransformer(transformers=(['encoder', OneHotEncoder(), [2]]),remainder)  
sklearn.compose.make_column_transformer((OneHotEncoder(),['label']),(),remainder)

sklearn.preprocessing.OneHotEncoder(drop='first')   
sklearn.preprocessing.StandardScaler, MinMaxScaler(Feature_range(start, stop+1)), RobustScaler, QuantileTransformer  
sklearn.preprocessing.PowerTransformer, KBinsDiscretizer  # 비선형 (np.log1p, Lasso, Ridge)
sklearn.preprocessing.PolynomialFeatures   # 다항회귀 

sklearn.decomposition.PCA (n_components=)
sklearn.cross_decomposition.PLSRegression(n_components=)

sklearn.manifold.TSNE(n_components= )
sklearn.manifold.MDS(n_components=)  

sklearn.discriminant_analysis.LinearDiscriminantAnalysis     # LDA

sklearn.linear_model.LinearRegression   
sklearn.linear_model.LogisticRegression  
sklearn.linear_model.Perceptron  
sklearn.linear_model.Lasso  
sklearn.linear_model.Ridge  
sklearn.linear_model.SGDClassifier(max_iter,eta0)  / SGDRegressor      # 경사하강법   

sklearn.svm.SVC, SVR, LinearSVC    

sklearn.naive_bayes.GaussianNB, CategoricalNB, MultinomialNB, BernoulliNB, ComplementNB  

sklearn.neighbors.KNeighborsClassifier(n_neighbors) / KNeighborsRegressor   # KNN    

sklearn.tree.DecisionTreeClassifier / ecisionTreeRegressor   
sklearn.tree.BaggingClassifier / BaggingRegressor   

sklearn.ensemble.RandomForestClassifier / RandomForestRegressor   
sklearn.ensemble.GradientBoostingClassifier / GradientBoostingRegressor  
sklearn.ensemble.AdaBoostClassifier / AdaBoostRegressor  
sklearn.ensemble.IsolationForest  

xgboost.XGBClassifier / XGBRegressor

sklearn.cluster.KMeans (n_cluster, init, n_init)   
sklearn.cluster import DBSCAN  
pyclustering.cluster.kmedoids  
scipy.cluster.hierarchy, dendrogram, linkage       # 계층적 군집  

sklearn.metrics.mean_absolute_error, mean_squared_error, r2_core   

sklearn.metrics.f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, roc_auc_score, classification_report  

sklearn.metrics.silhouette_score (X, lables)   

</br>

## preprocessing 

- row : Vertorization, groupby, pivot_table -> select Data (Train, Test)   
1. 개별 (row 한개씩) 계산은 : itertuples, apply (axis = 0,1) 보다 vectorization 사용  
2. 구룹별 (slicing) 계산은 : for i_value in unique_values_list (unique한 값을 공유하는 sliced 그룹)   
* groupby, pivot_table 사용 시, 기존 데이터셋을 업데이트 할때 reset_index, sort_index 로 기존 형태 복구 필요  

- columns : For, apply -> select Feature (X, y)  

## modeling
```
X_train = df[[ , ]]   : 독립변수 학습 데이터  (series.to_frame())
X_test = df[[ , ]]    : 독립변수 테스트 데이터  (series.to_frame())
y_train = df[ ]       : 종속변수 학습 데이터  
y_test = df[ ]        : 종속변수 테스트 데이터  (실제결과)

y_pred                : 종속변수 예측 데이터  
# pred_list = [ ]     : 종속변수 예측 데이터 리스트  


my_model = Given_ML_Model()       # ML 모델 클래스를 인스턴스화(하이퍼파라미터) 
my_model.fit(X_train, y_train)    # 학습데이터로 학습  - fit_predict(), fit_transform(), inverse_trasform()  
y_pred = my_model.predict(X_test) # 학습된 모델(my_model)으로 테스트 데이터 (독립변수, X_test)의 결과(종속변수, y_pred) 예측값  
                                  # coef_, intercept_, label_, inertia_  
my_model.score(X_test, y_test)    # 학습된 모델(my_model)의 test 평가  

# pred_list.append(y_pred)  
# XXX_score(y_test,y_pred)        # 학습된 모델(my_model)의 실제값 (y_test) 대비 예측(y_pred)결과 평가

```


