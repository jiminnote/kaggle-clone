import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

# 데이터 불러오기
data_path = "/kaggle/input/h-and-m-personalized-fashion-recommendations/"
customers = pd.read_csv(data_path + "customers.csv")
articles = pd.read_csv(data_path + "articles.csv")
transactions = pd.read_csv(data_path + "transactions_train.csv", dtype={'t_dat': str}, nrows=100_000)

# 데이터 개요
print("customers shape:", customers.shape)
print("articles shape:", articles.shape)
print("transactions shape:", transactions.shape)



# -------------------------------
# 1. 고객 데이터 (customers.csv)
# -------------------------------

# 결측치 확인
print("\ncustomers 결측치:\n", customers.isnull().sum())

# 나이 분포
plt.figure(figsize=(10, 4))
sns.histplot(customers['age'].dropna(), bins=50, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 클럽 멤버 상태
sns.countplot(data=customers, x='club_member_status', order=customers['club_member_status'].value_counts().index)
plt.title('Club Member Status Distribution')
plt.show()

# 패션 뉴스 수신 빈도
plt.figure(figsize=(4, 3))
sns.countplot(data=customers, x='fashion_news_frequency', order=customers['fashion_news_frequency'].value_counts().index)
plt.title("Fashion News Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 3. 상품 데이터 (articles.csv)
# -------------------------------

# 고유 상품 수
print("고유 article_id 수:", articles['article_id'].nunique()) # 105,562 개

# product_type_name Top 10
plt.figure(figsize=(10, 4))
articles['product_type_name'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Product Types')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 색상 분포 Top 10
plt.figure(figsize=(10, 4))
articles['colour_group_name'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title('Top 10 Colour Groups')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.show()

# -------------------------------
# 3. 거래 데이터 (transactions.csv)
# -------------------------------

# 날짜 컬럼을 문자열로 지정해서 정확하게 불러오기 
transactions = pd.read_csv(data_path + "transactions_train.csv", dtype={'t_dat': str})

# 날짜 형식으로 변환
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], format='%Y-%m-%d', errors='coerce')

## 위 처럼 안했더니 1970 년도 날짜로 나오면서 데이터가 깨짐

# 일별 거래 수
daily_orders = transactions['t_dat'].value_counts().sort_index()

# 시각화
plt.figure(figsize=(14, 5))
daily_orders.plot()
plt.title("Number of Transactions Over Time")
plt.xlabel("Date")
plt.ylabel("Transactions")
plt.grid(True)
plt.tight_layout()
plt.show()
# 가격 분포 확인
plt.figure(figsize=(8, 4))
sns.histplot(transactions['price'], bins=50, kde=True)
plt.title("Transaction Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 판매 채널 비율
sns.countplot(data=transactions, x='sales_channel_id')
plt.title("Sales Channel Distribution")
plt.xticks([0, 1], ['Online', 'Offline'])
plt.show()