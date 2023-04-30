### VSCode with Jypyter extensions
https://code.visualstudio.com/docs/datascience/overview  
VSCode 에 Jupyter extensions 설치하고,  
해당 프로젝트를 github repo 에 저장  

```$ python3 -m venv .venv``` 을 이용해서 virtual 환경 설치   
```$ source .venv/bin/activate``` 으로 virtual 환경 실행   
```(.venv)$ pip install ipykernel```으로 관련 python 모듈 설치  

> '$ python -m ipykernel install --user' 을 사용하는 경우 jupyter notebook (lab) 에서 ikernel 을 import 할때,  
설정오류가 발생할 수 있다. 되도록 사용하지 말자 (pip 로 설치)

```
[E 08:57:38.101 NotebookApp] Failed to run command:
    ['/Users/myoungjunesung/pyproject/data/ipynb/.venv/bin/python3', '-m', 'ipykernel_launcher', '-f', '/Users/myoungjunesung/Library/Jupyter/runtime/kernel-5cc8a27b-46fd-45a7-8eee-395c9d2b9f92.json']
```

### Jupyter lab
https://github.com/jupyterlab/jupyterlab
```bash
$ pip install ipykernel
# or $ pip install jupyter (or notebook)
$ pip install jupyterlab
$ jupyter lab  
# or $ jupyter notebook 
```
git extension : ```pip install --upgrade jupyterlab jupyterlab-git```

### Other Jupyter 
> 1. [Anaconda](https://www.anaconda.com/) 을 설치해서 Jupyter Notebook 사용  (Local)  
> 2. [google colab](https://colab.research.google.com), [Kaggle](https://www.kaggle.com), [dacon](https://dacon.io/) 에서 제공하는 cloud 커널 을 사용  (Cloud)  

### 참고 사이트

https://passwd.tistory.com/entry/Python-Pandas-Series-1  
https://datascienceschool.net/intro.html  
