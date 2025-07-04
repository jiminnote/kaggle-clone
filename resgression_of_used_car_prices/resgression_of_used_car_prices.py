## Part 1 : Import Libraries and Configure Environment
import os
import sklearn
import numpy as np # 선형대수 및 수치 계산용
import pandas as pd # 데이터 처리, CSV 파일 입출력 등
import seaborn as sns # 통계적 데이터 시각화
import matplotlib.pyplot as plt # 그래프 그리기용 라이브러리


# 현재 .py 파일이 위치한 디렉토리 경로
base_path = os.path.dirname(os.path.abspath(__file__))

# 파일 경로 구성
train_path = os.path.join(base_path, 'playground-series-s4e9', 'train.csv')
test_path = os.path.join(base_path, 'playground-series-s4e9', 'test.csv')
sample_submission_path = os.path.join(base_path, 'playground-series-s4e9', 'sample_submission.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# 판다스 출력 옵션 설정 :  행,열 각 최대 100개 출력
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None

# 데이터 불러오기
train = pd.read_csv(train_path, index_col='id')
test = pd.read_csv(test_path, index_col='id')

# 전처리를 위해 train 과 test 합치기
df = pd.concat([train, test])

# 상위 5개 행 출력 (0~4번 인덱스 )
df.head()

## Part 3: 변수 식별 및 범주화

# 전체 컬럼 중 'price' 만 제외한 변수 추출
features = df.columns.to_list() # 전체 컬럼명을 리스트 변환
target = 'price' # 타겟 변수 지정 
features.remove(target) # 변수 리스트에 'price' 제거

# 변수를 타입 기준으 나누기
catFeatures = df[features].select_dtypes(include='object').columns.to_list()
numFeatures = df[features].select_dtypes(include='int64').columns.to_list()

print("Categorical Features:", catFeatures)
print("Numerical Features:", numFeatures)

## Part 4: 데이터 정리 및  전처리

# 결측값 및  잘못된 'fuel_type' 데이터 처리
df['fuel_type'].fillna('Electric',inplace=True) # 결측값일 경우 'Electric'을 넣어주기
df['fuel_type'].replace('-', 'Unknown',inplace=True) # 없는 값의 경우 'Unknown'을 넣어주기

# 'engine' 변수에 숫자 정보 추출  
df['horsepower'] = df['engine'].str.extract(r'(\d+).0HP').astype('Int64') # '300.0HP' → 300
df['engine_size'] = df['engine'].str.extract(r'(\d+\.?\d+)\s*(?:L|Liters)').astype(float) # '3.5L' 또는 '3.5 Liters' → 3.5
df['cylinders'] = df['engine'].str.extract(r'(?:V(\d+)|W(\d+)|I(\d+)|H(\d+)|(\d+)\s*Cylinder)').bfill(axis=1).iloc[:, 0].astype('Int64') # V6, 4 Cylinder 등 → 6, 4 등
df['gears'] = df['transmission'].str.extract(r'(\d+)-Speed').astype('Int64') # '8-Speed Automatic' → 8

# 변속기 종류 분류
conditions = [
    df['transmission'].str.contains('AT|Automatic|AT', case = False, na=False),
    df['transmission'].str.contains('Manual|M/T|Mt', case=False, na=False)
]


# 새 컬럼 생성 ('Automatic', 'Manual', 'Other')
choices = ['Automatic','Manual']
df['transmission_type'] = np.select(conditions, choices, default ='Other')
# 'clean_title'과 'accident' 값을 조합한 새로운 변수 생성
df['clean_title*no_accident'] = np.where(((df['clean_title'] == 'Yes') & (df['accident'] == 'None reported')) == True, 'Yes','No')

df.head()


## Part 5 : Handling High Cardinality Features

high_cardinality_features = ['model','engine','ext_col','int_col']
threshold = 100 # 기준 빈도 수 설정

for feature in high_cardinality_features:
    counts = df[features].value_counts() # 각 번주의 등장 횟수
    scarse_categories = counts[counts < threshold].index # 기준 빈도 수 보다 적은 범주 추출
    df[feature] = df[feature].apply(lambda x: 'Other' if x in scarse_categories else x) # 희귀값을 'Other'로 치환

df[high_cardinality_features].nunique() # 컬럼 중 카디널리티가 100개 이상인 컬럼들만 추출
    
## Part 6: Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder # 범주형 변수를 숫자 변환

carFeatures = df.select_dtypes(include="object").columns.to_list() # object타입 컬럼들 리스트 저장 
le = LabelEncoder() 
#   예: “Diesel”, “Electric”, “Gasoline” → 0, 1, 2 로 변환
for feature in catFeatures:
    df[feature] = le.fit_transform(df[feature])

df['cylinders'].fillna(0, inplace=True)
df['gears'].fillna(0, inplace=True)

df.head()

## Part 7: 데이터 결측값 예측

from sklearn.impute import SimpleImputer


train_df = df.iloc[train.index]
test_df = df.iloc[test.index]

imputer = SimpleImputer(strategy='median')
train_df[['horsepower', 'engine_size']] = imputer.fit_transform(train_df[['horsepower', 'engine_size']])
test_df[['horsepower', 'engine_size']] = imputer.transform(test_df[['horsepower', 'engine_size']])

train_df.head()

## Part 8: 데이터 시각화

# 타겟 변수의 분포 시각화
plt.figure(figsize=(8,4)) # 그래프 크기설정 
sns.histplot(train_df['price'], kde=True, color='blue') # 가격 분포 히스토그램 +  밀도 곡선
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 수치형 컬럼만 추출
numeric_df = train_df.select_dtypes(include=['number'])

# 변수들 간의 상관관계 히트맵 시각화
plt.figure(figsize=(12, 8))
corr_matrix = numeric_df.corr() 
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

## Part 9: 모델 학습

from sklearn.ensemble import RandomForestRegressor  
from catboost import CatBoostRegressor  

mod1 = RandomForestRegressor(n_estimators=100, random_state=42)  
# 🌲 100개의 결정 트리를 사용하는 랜덤포레스트 회귀 모델 생성
# random_state=42는 재현성을 위한 랜덤 시드 고정

mod2 = CatBoostRegressor(
    iterations=500, # 500번 반복
    learning_rate=0.1, # 학습률 설정 (한 번에 얼마나 업데이트할지)
    depth=6, # 트리의 최대 깊이
    random_seed=42, # # 결과 일관성을 위한 랜덤 시드 고정
    verbose=0
)


categorical_columns = train_df.select_dtypes(include=['object','category']).columns.tolist() # 범주형 변수 추출 -> brand', 'fuel_type', 'transmission_type'
catFeatures = [feature for feature in categorical_columns if feature in features] # features 리스트와 겹치는 범주형 변수 추출 ->

mod1.fit(train_df[features], train_df[target])
mod2.fit(train_df[features], train_df[target], cat_features=catFeatures)

import lightgbm as lgb
import catboost as ctb

mod1 = lgb.LGBMRegressor(n_estimators=430,
                        reg_alpha=72,
                        reg_lambda=57,
                        colsample_bytree = 0.43,
                        aubsample=0.96,
                        learning_rate = 0.025,
                        max_depth=8,
                        num_leaves=583,
                        min_child_samples=320,
                        verbose=-1)

mod2 = ctb.CatBoostRegressor(iterations=475,
                             subsample=0.55,
                             colsample_bylevel=0.37,
                             min_data_in_leaf=210,
                             learning_rate=0.03,
                             l2_leaf_reg=56,
                             depth=10,
                             verbose=False)

mod1.fit(train_df[features], train_df[target])
mod2.fit(train_df[features], train_df[target], cat_features=catFeatures)

## Part 10: 피처 중요도 시각화

lgb.plot_importance(mod1, importance_type='split', figsize=(10, 6))
plt.title('LightGBM Feature Importance')
plt.show()

## Part 11: 예측 및 제출

submission_lgb = pd.read_csv(sample_submission_path)

submission_lgb['price'] = 0.8 * mod1.predict(test_df[features])
+0.2*mod2.predict(test_df[features])

submission_lgb.to_csv('submission.csv')

