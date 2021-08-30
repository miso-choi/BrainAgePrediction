# BrainAgePrediction 🧠
인간의 뇌 MRI영상으로부터 추출된 특징값 데이터로 생물학적 나이를 예측합니다.

### 1. 데이터
데이터는 학부 연구실에서 사용한 것이기 때문에 전체 데이터는 공개하지 않습니다.
* Train-set  
    <img src="https://user-images.githubusercontent.com/78155086/131341159-f854a42a-c65f-44e0-bcb4-3f8d4053660e.PNG" width="70%">
    <img src="https://user-images.githubusercontent.com/78155086/131341725-d6cf8231-60df-4ec9-a90e-5642f41abbad.PNG" width="13.8%">  
    - data shape: (2184, 535)     --   2184명의 뇌 MRI영상에서 추출한 535개의 특징값 데이터
    - X: 특징값(535개), Y는 age(1개) 입니다.
    - Overfitting을 방지하기 위해 10-fold Cross Validation을 하였습니다.
* Test-set
    - data shape: (50, 535)
    - X (특징값 535개)만 있습니다.
    - External Validation을 위해 사용합니다.
    

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
    
#### Gaussian Process Regression ✨
    test MAE: 5.343
    train MAE: 1.049e-06

### 3. 결과 시각화
모델 최적화를 할 때 parameter에 따른 test MAE, train MAE의 변화를 시각화 하였습니다. (visualization.py)  
(단, 본 코드의 시각화는 parameter가 두 개일 때 가능합니다.)


### 4. 주요 Issue
연령에 따른 error plot

### 5. 실행 방법
1) 모든 파일들을 같은 경로상에 두기
2) requirements.txt 설치
3) !python 모델명.py 실행  (ex. !python ElasticNet.py)

  
  
* * *
- 참고자료(Reference)  
    https://www.inflearn.com/course/파이썬-머신러닝-완벽가이드
