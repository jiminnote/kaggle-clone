import os
import pandas as pd #표 형태 데이터를 다루 위한 필수 라이브러리 
import numpy as np  #수치 연산, 배열 계산에 유용한 수학 라이브러리
import math         #삼각함수, 로그, 제곱근 등 수학 함수 모음 
import seaborn as sns #통계적 데이 시각화를 쉽게 만들어주는 시각화 라이브러리
import matplotlib.pyplot as plt #기본 그래프 (선, 막대 등) 시각화 도구

import warnings #실행 중 발생하는 경고 메세지를 제어할 수 있는 파이썬 기본 모듈
warnings.filterwarnings("ignore") #경고 메세지 무시, 필요 default로 돌려놓음

# 현재 .py 파일이 위치한 디렉토리 경로
base_path = os.path.dirname(os.path.abspath(__file__))

# 파일 경로 구성
train_path = os.path.join(base_path, 'playground-series-s4e10', 'train.csv')
test_path = os.path.join(base_path, 'playground-series-s4e10', 'test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sns.countplot(x='loan_status', data=train) # 'loan_status'컬럼의 각 값 (0,1)별로 몇개인지 막대그래프 시각화

plt.title('Loan Status Distribution') #그래프 제목
plt.xlabel('Loan Status') #x축 레이블 설정 (0=거절,1=승인)
plt.ylabel('Count') #y축 레이블 설정 (각 클래스의 개수)

plt.show() #그래프 출력

negative_samples = train[train['loan_status']==0]
# 대출 '거절'(loan_status == 0)인 행들만 추출 → '불리한 클래스'
positive_samples = train[train['loan_status']==1]
# 대출 '승인'(loan_status == 1)인 행들만 추출 → '유리한 클래스'

negative_samples_under = negative_samples.sample(len(positive_samples), random_state=42)
# 거절 샘플 중 승인 샘플 개수만큼 무작위 추출
# 즉, '승인 클래스 수'에 맞춰 거절 클래스 수를 줄이기 위한 언더샘플링
train = pd.concat([negative_samples_under,positive_samples])
# 두 개의 클래스 수를 같게 맞출 후 다시 합치기 -> 클래스 균형 데이터셋 생성
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
# 데이터 셔플 -> 순서에 의한 편향 제거
# reset_index로 인덱스 초기화 

print(train['loan_status'].value_counts())
# 최종적으로 클래스 비율이 1:1로 잘 맞춰졌는지 확인용 출력

# < 결과 >
# loan_status
# 0    8350
# 1    8350
# Name: count, dtype: int64

print(train.dtypes) # 각 컬럼의 데이터 타입 확인

###

numeric_columns = train.select_dtypes(include=['int','float']).columns
# 숫자형(int,float)타입 컬럼만 선택
num_cols = len(numeric_columns) # 숫자형 컬럼 수 
cols = 3 # 그래프 열 개수 설정
rows = math.ceil(num_cols / cols) # 필요 행 개수 계산 후 올림(ceil)
# subplot 생성
fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
# 2차원 배열을 1차원 배열로 평탄화 (인덱싱 편하게)
axes = axes.reshape(-1)
# 각 범주형 변수별로 countplot 시각화 진행
for i, column in enumerate(numeric_columns): # 숫자형 컬럼 각각에 대해
    # loan_status = 0인 경우의 KDE Plot 그리기
    sns.kdeplot(train[train['loan_status'] == 0][column], label = 'loan_status = 0', shade=True, ax=axes[i])
    # loan_status = 1인 경우의 KDE Plot 그리기
    sns.kdeplot(train[train['loan_status'] == 1][column], label = 'loan_status = 1', shade=True, ax=axes[i])
    
    # 그래프 제목, 축 레이블, 범례 설정
    axes[i].set_title(f'{column} Distribution by Loan Status')  
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Density')
    axes[i].legend()

# 안 쓰는 서브플롯 제거
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
# 그래프 레이아웃 자동 정리
plt.tight_layout()

plt.show()

###  

object_columns = train.select_dtypes(include='object').columns #object 타입컬럼만 선택

num_cols = len(object_columns) # object컬럼 수
cols = 3 
rows = math.ceil(num_cols / cols)
# subplot 생성
fig, axes = plt.subplots(rows, cols, figsize=[18, rows * 5])

axes = axes.reshape(-1)

for i, column in enumerate(object_columns): # 
    # 대출 승인 여부에 따라 색상 구분
    sns.countplot(x=column, hue='loan_status', data=train, ax=axes[i])

    axes[i].set_title(f'{column} Distribution by Loan Status') 
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count') 
    axes[i].legend()

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# 그래프 레이아웃 자동 정리
plt.tight_layout()

plt.show()


sns.set(style="whitegrid") # seaborn 스타일 설정 (배경을 흰색 그리드)
plt.figure(figsize=(15,10)) # 전체 subplot의 사이즈 설정

for i, feature in enumerate(numeric_columns):
    plt.subplot(len(numeric_columns) // 3+1, 3, i+1) # 3열 기준으로 subplot 생성
    sns.boxplot(x=train[feature], color = 'lightblue') # boxplot 시각화로 이상치 탐색
    plt.title(f'Boxplot of {feature}') #subplot 제목 설정
    plt.xlabel(feature)

plt.tight_layout()
plt.show()

# subplot 생성
fig, axes = plt.subplots(rows, cols, figsize=[18, rows * 5])

axes = axes.reshape(-1)

for i,column in enumerate(object_columns):
    sns.countplot(x=column,data=train, ax=axes[i], palette='Set2')
    
    axes[i].set_title(f'Count of {column}') 
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Count')

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout()
plt.show()

print(train.isnull().sum())