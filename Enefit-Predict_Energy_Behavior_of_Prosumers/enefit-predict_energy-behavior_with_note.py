import pandas as pd  
import numpy as np 


import random  
np.random.seed(2023)  # numpy 랜덤 시드 고정
random.seed(2023)  # random 모듈 시드 고정
import warnings  
warnings.filterwarnings('ignore')  

train_df=pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/train.csv")  
print(f"len(train_df):{len(train_df)}")  
train_df.dropna(axis=0, how='any', inplace=True)  # 결측값이 있는 행 제거
train_df.head() 

features=['county','is_business','product_type','is_consumption'] 
X=train_df[features].values  
y=train_df['target'].values 

from lightgbm import  LGBMRegressor  # LightGBM 회귀 모델 불러오기
def MAE(y_true,y_pred):  # MAE(평균 절대 오차) 계산 함수 정의
    return np.mean(abs(y_true-y_pred))
model=LGBMRegressor()  # LightGBM 회귀 모델 인스턴스 생성
model.fit(X,y)  # 모델 학습
y_pred=model.predict(X)  # 학습 데이터에 대해 예측 수행
print(f"train_pred:{MAE(y,y_pred)}") 

import enefit  # enefit 환경에서 제공되는 인터페이스 불러오기
env = enefit.make_env()  # 환경 인스턴스 생성
iter_test = env.iter_test()  # 테스트 데이터를 반복자로 받기

# 테스트셋을 순회하며 예측값을 생성
for (test, revealed_targets, client, historical_weather,
        forecast_weather, electricity_prices, gas_prices, sample_prediction) in iter_test:
    sample_prediction['target'] = model.predict(test[features])  # 테스트 데이터에 대한 예측 수행
    env.predict(sample_prediction)  # 예측 결과를 제출