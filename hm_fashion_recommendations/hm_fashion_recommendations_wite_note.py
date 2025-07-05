

## Part 1. 최근 구매 기반 추천 

import cudf # GPU 가속으로 데이터를 처리할 수 있게 해주는 RAPIDS 라이브러리 / pandas랑 문법이 거의 똑같지만, 속도가 훨씬 빠름
print('RAPIDS version',cudf.__version__) 

train = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv') # csv 파일을 GPU 메모리(cudf) 읽어옴
train['customer_id'] = train['customer_id'].str[-16:].str.hex_to_int().astype('int64') # 문자열로 되어 있는 customer_id 뒤 16자리로 16진수 문자열을 정수형으로 변환 -> 숫자 ID를 쓰기 쉽게
train['article_id'] = train.article_id.astype('int32') # 32비트 정수로 변환 -> 메모리 최적화
train.t_dat = cudf.to_datetime(train.t_dat)
train = train[['t_dat','customer_id','article_id']]
train.to_parquet('train.pqt',index=False) # 전처리 결과를 train.pqt에 저장 -> Parquet은 속도 빠르고 용량 작아서 머신러닝에서 자주 씀 
print( train.shape )
train.head()

tmp = train.groupby('customer_id').t_dat.max().reset_index() # customer_id별로 가장 최근 t_dat(구매일)
tmp.columns = ['customer_id','max_dat']
train = train.merge(tmp,on=['customer_id'],how='left') # 최근 구매일(max_dat) 병합
train['diff_dat'] = (train.max_dat - train.t_dat).dt.days
train = train.loc[train['diff_dat']<=6] # 최근 일주일 (6일 이하)의 거래만 필터링 -> 최근 7일동안의 구매이력만 사용하겠다는 전략!
print('Train shape:',train.shape)

tmp = train.groupby(['customer_id','article_id'])['t_dat'].agg('count').reset_index() # customer_id과 article_id의 조합별로 몇 번 구매했는지 t_dat 개수 계산
tmp.columns = ['customer_id','article_id','ct']
train = train.merge(tmp,on=['customer_id','article_id'],how='left') # 구매 횟수(ct)) 정보 병합
train = train.sort_values(['ct','t_dat'],ascending=False) # 구매 횟수(ct)가 많고, 최신 날짜(t_dat) 순으로 정렬
train = train.drop_duplicates(['customer_id','article_id']) # 고객-상품 조합별로 중복 제거
train = train.sort_values(['ct','t_dat'],ascending=False) # 중복 제거 후 순서 정리
train.head()

#  Part 2. 연관 상품 추천
import pandas as pd, numpy as np 
train = train.to_pandas() # 일반적인 연산을 수행하기 위해 cudf에서 pandas로 변환 -> 이후 연산은 GPU가 아닌 CPU에서 처리
pairs = np.load('../input/hmitempairs/pairs_cudf.npy',allow_pickle=True).item() # {article_id: paired_article_id} -> 어떤 상품을 샀을 때 자주 함께 구매된 다른 상품 (쌍 정보)
train['article_id2'] = train.article_id.map(pairs) # article_id를 key로 해서 pairs 딕셔너리에서 연관된 상품 ID(article_id2) 매핑 -> article_id를 key로 해서 pairs 딕셔너리에서 연관된 상품 ID(article_id2) 매핑

# 기존에 매핑된 연관 상품 article_id2를 사용하여 고객별로 추천 리스트를 생성
train2 = train[['customer_id','article_id2']].copy() # 고객 ID와 연관 상품 ID만 복사해서 새 DataFrame 생성
train2 = train2.loc[train2.article_id2.notnull()] # null 값 제거
train2 = train2.drop_duplicates(['customer_id','article_id2']) # 중복 제거 -> 고객 한명이 같은 연관상품을 여러 번 추천받는 것 방지
train2 = train2.rename({'article_id2':'article_id'},axis=1) # 향후 추천 리스트에 합치기 위해 기존 컬럼명과 동일하게 변경


# -------------------------------
# train  : 최근 구매 기반(최근 7일 내)
# train2 : 연관 상품 기반(자주 같이된 상품)
# -------------------------------
# 최근 구매 기반 추천 + 연관 상품 추천을 하나의 추천 리스트로 결합
train = train[['customer_id','article_id']] # 최근 구매 기반 추천 결과만 남기기
train = pd.concat([train,train2],axis=0,ignore_index=True) # train2와 결합
train.article_id = train.article_id.astype('int32') # 일관성 유지 위해 정수형으로 통일
train = train.drop_duplicates(['customer_id','article_id']) # -> 같은 고객에게 같은 상품이 여러 추천 경로로 들어올 수 있으므로 제거

# 추천 리스트 문자열로 변환
train.article_id = ' 0' + train.article_id.astype('str')
preds = cudf.DataFrame( train.groupby('customer_id').article_id.sum().reset_index() )
preds.columns = ['customer_id','prediction']
preds.head()

# 최근 일주일 간 인기 상품 Top 12 뽑기
train = cudf.read_parquet('train.pqt') # 이전에 저장해둔 전처리된 거래 데이터(train.pqt)를 GPU 메모리(cudf)로 불러오기
train.t_dat = cudf.to_datetime(train.t_dat)
train = train.loc[train.t_dat >= cudf.to_datetime('2020-09-16')] # 2020-09-16 이후의 거래만 필터링 
top12 = ' 0' + ' 0'.join(train.article_id.value_counts().to_pandas().index.astype('str')[:12]) # 최근 일주일간 가장 많이 팔린 상품 12개를 추출
print("Last week's top 12 popular items:")
print( top12 )

# 예측 결과를 submission.csv로 저장하기
sub = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
sub = sub[['customer_id']]
sub['customer_id_2'] = sub['customer_id'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.merge(preds.rename({'customer_id':'customer_id_2'},axis=1),\
    on='customer_id_2', how='left').fillna('')
del sub['customer_id_2']
sub.prediction = sub.prediction + top12
sub.prediction = sub.prediction.str.strip()
sub.prediction = sub.prediction.str[:131]
sub.to_csv(f'submission.csv',index=False)
sub.head()