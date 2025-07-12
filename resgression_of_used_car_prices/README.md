# 🚗 Regression of Used Car Prices — 핵심 로직 요약

[[Kaggle 필사 #2] Regression of Used Car Prices - EDA 분석 & 모델링 📊](https://velog.io/@jiminnote/Kaggle-%ED%95%84%EC%82%AC-2-Regression-of-Used-Car-Prices-EDA-%EB%B6%84%EC%84%9D-%EC%A0%95%EB%A6%AC)

[[Kaggle 필사 #2] Regression of Used Car Prices - Lesson Learned ☑️](https://velog.io/@jiminnote/Kaggle-필사-2-Regression-of-Used-Car-Prices-Lesson-Learned)

---

## 1️⃣ 문제 정의 & 평가 방식

- **문제**: 중고차의 다양한 속성 데이터를 기반으로 `price` 예측 (회귀 문제)
- **타겟 변수**: `price` (실수형)
- **평가지표**: RMSE (Root Mean Squared Error)
  > 예측값과 실제값의 차이를 제곱하여 평균낸 뒤 제곱근을 취한 값

---

## 2️⃣ 데이터 구성 요약

| 컬럼명 | 설명 | 유형 |
|---|---|---|
| brand / model | 브랜드 & 모델명 | 범주형 |
| model_year | 제조 연도 | 수치형 |
| mileage | 주행 거리 | 수치형 |
| fuel_type | 연료 유형 | 범주형 |
| engine / transmission | 엔진 / 변속기 | 문자열 |
| exterior_color / interior_color | 외부 / 내부 색상 | 범주형 |
| accident / clean_title | 사고 이력 / 타이틀 상태 | 범주형 |
| price | 차량 가격 (예측 대상) | 수치형 |

---

## 3️⃣ EDA 핵심 요약

### ✅ 결측치 확인
- `fuel_type`, `accident`, `clean_title` 등 일부 컬럼 결측 존재
- → `SimpleImputer` 또는 최빈값/None 처리 진행

### ✅ 이상치 탐색
- `mileage`: 40만 km 이상 → 이상치로 간주 가능
- `price`: $300만 이상 초고가 차량 → 평균을 왜곡

### ✅ 타겟 분포 → 로그 변환 필요

- 로그 변환 전: 오른쪽 꼬리가 긴 비대칭 분포
- 로그 변환 후: 정규성에 가까워짐  
  → `np.log1p(price)` 처리 후 모델 학습

---

## 4️⃣ 전처리 및 특성 엔지니어링

### ✅ 범주형 인코딩

- `get_dummies()`로 one-hot 인코딩 (RF용)
- `LabelEncoder` 또는 `cat_features` (CatBoost용)

### ✅ 특성 가공

- `engine`, `transmission` → 숫자 정보 추출
- 고유값 많은 컬럼 → 'Other'로 정리하여 cardinality 축소

---

## 5️⃣ 모델링 요약

### 🎯 사용 모델
| 모델 | 설명 |
|---|---|
| **RandomForestRegressor** | 트리 기반, 결측/이상치에 강함, 범주형 처리 용이 |
| **CatBoostRegressor** | 범주형 자동 인코딩 + 부스팅 알고리즘 |
| **앙상블** | RF 80% + CatBoost 20% 가중 평균 |

### 📉 모델별 RMSE 비교

```python
# 예시 코드
ensemble_pred = 0.8 * rf_pred + 0.2 * cat_pred
rmse = sqrt(mean_squared_error(y_true, ensemble_pred))
```

| 모델 |RMSE |
|---|---|
| **RandomForest** | 67,879 |
| **CatBoost** | 68,065 |
| **앙상블** | 67,871(최고 성능) |

---

### ✅ 핵심 학습 포인트
### 📌 EDA & 전처리 중요성
* 로그 변환으로 정규성 확보 -> 성능 개선
* 범주형 정리, 이상치 탐색, 결측 처리 등 실무 감각 훈련

### 📌 모델 특성 이해
| 모델 |장점|약점|
|---|---|---|
| **RandomForest** | 안정적, 결측/이상치 강함 | 보수적, 과적합 가능|
| **CatBoost** | 범주형 자동 인코딩 | 예외처리, 튜닝 중요|
| **앙상블** | 균형 잡힌 성능 | 구현 복잡도 증가|

## ✍️ 회고
* 범주형 변수 처리 방식과 로그 변환의 효과를 확인했다.
* CatBoost의 범주형 처리 자동화 기능 덕분에 인코딩 실수가 줄어들 수 있음을 알게 되었다.


> 📌 이 프로젝트는 Kaggle 대회 [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e2/)를 기반으로 분석 및 모델링을 진행한 결과입니다.
