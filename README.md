# BrainAgePrediction
인간의 뇌 MRI영상으로부터 추출된 특징값 데이터로 생물학적 나이를 예측합니다.

### 1. 데이터
데이터는 학부 연구실에서 사용한 데이터로 공개할 수 없습니다.  
- data shape: (2184, 535)   --   2184명의 뇌 MRI영상에서 추출한 535개의 특징값 데이터
- X는 특징값(535개), Y는 age(1개) 입니다.

### 2. 사용한 머신러닝 모델, 성능 비교
* 각 모델의 성능을 비교하기 위해 회귀평가지표인 MAE(mean absolute error)를 사용하였습니다.  
#### ElasticNet
    test MAE: 6.576
    train MAE: 5.043

#### RandomForest
    test MAE: 7.626
    train MAE: 3.200
    
#### LightGBM
    test MAE: 6.792
    train MAE: 1.831
    
#### Gaussian Process Regression
    test MAE: 5.343
    train MAE: 1.049e-06

### 3. 결과 시각화
모델 최적화를 할 때 parameter에 따른 test MAE, train MAE의 변화를 시각화 하였습니다. (visualization.py)  
(단, 본 코드의 시각화는 parameter가 두 개일 때 가능합니다.)


### 4. 주요 Issue
연령에 따른 error plot

### 5. 실행 방법
1) requirements.txt 설치
2) preprocessing.py와 다른 파일들(예-ElasticNet.py)을 같은 경로상에 두기
3) !python ElasticNet.py 실행



* * *
- 참고자료(Reference)  
    https://www.inflearn.com/course/파이썬-머신러닝-완벽가이드
