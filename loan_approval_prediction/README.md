
# 🧠 Loan Approval Prediction — 핵심 로직 요약
[**[Kaggle 필사 #1] Loan Approval Prediction - EDA 분석 정리📊**](https://velog.io/@jiminnote/Kaggle-%ED%95%84%EC%82%AC-1-Loan-Approval-Prediction-EDA-%EB%B6%84%EC%84%9D-%EC%A0%95%EB%A6%AC)

[**[Kaggle 필사 #1] Loan Approval Prediction - Lesson Learned ☑️**](https://velog.io/@jiminnote/Kaggle-%ED%95%84%EC%82%AC-1-Loan-Approval-Prediction-Lesson-Learned)

## 1️⃣ 문제 정의 & 목표

- **목표**: 사용자 정보 데이터를 기반으로 **대출 승인 여부 (`loan_status`)** 예측
- **타겟값**: `loan_status` (0: 거절 / 1: 승인)
- **평가지표**: ROC-AUC
→ 불균형 데이터이기 때문에 정확도보다 **AUC**가 적절

---

## 2️⃣ 데이터 구성 요약

| 컬럼명 | 설명 | 특징 |
| --- | --- | --- |
| id | 신청자 ID | 예측과 무관 |
| person_income | 연소득 | 수치형 변수 |
| loan_intent | 대출 목적 | 범주형 |
| loan_int_rate | 대출 이자율 | 타겟과 상관관계 있음 |
| loan_percent_income | 소득 대비 대출 비율 | 파생변수 |
| cb_person_default_on_file | 연체 이력 | 범주형 |
| loan_status | 예측 대상 | 타겟값 (0 / 1) |

---

## 3️⃣ EDA 핵심 흐름

### 1. 타겟 분포 확인

```python
sns.countplot(x='loan_status', data=train)
```

- 승인 / 거절 비율이 불균형하기 때문에 언더샘플링 사용 필요

### **2. 수치형 변수 vs 타겟 (kdeplot)**

```python
sns.kdeplot(data = df, x = 'loan_int_rate', hue = 'loan_status')
```

- `loan_int_rate`, `loan_percent_income` → 분포 차이 명확
- `person_income` , `loan_amnt` → 영향도 낮음

### **3. 범주형 변수 vs 타겟 (countplot)**

```python
sns.countplot(x='loan_intent', hue = 'loan_status', data=train)
```

- `loan_intent`, `loan_grade` , `home_ownership` 등 → 승인 / 거절 간 패턴 존재

### **4. 이상치 탐색 (boxplot)**

```python
sns.boxplot(x='loan_status', y='loan_amnt', data=train)
```

- `loan_amnt`, `income`, `emp_length`일 부 이상치 존재 → 로그 스케일 or 클리핑 고려

---

## **4️⃣ 전처리 로직**

### **✅ 언더샘플링 처리**

```python
# 거절 수가 더 많을 때, 승인 수 만큼 랜덤 추출
approve_df = df[df['loan_status']==1]
reject_df = df[df['loan_status']==0].sample(n=len(approve_df), random_state=42)

df = pd.concat([approve_df, reject_df])
```

---

### **✅ 범주형 인코딩**

```python
df = pd.get_dummies(df, drop_first=True)
```

---

## **5️⃣ 📈 유용했던 시각화 코드 모음**

| **목적** | **시각화 함수** |
| --- | --- |
| 수치형 변수 분포 | sns.kdeplot() |
| 범주형 변수 비교 | sns.countplot() |
| 이상치 탐색 | sns.boxplot() |
| 타겟 불균형 확인 | sns.countplot(x='loan_status') |

---

## **📝 Lesson Learned**

✅ loan_int_rate, loan_percent_income, loan_grade
→ 타겟값 예측에 유의미한 변수

✅ 이진 분류의 불균형 데이터 처리
→ 언더샘플링을 활용하여 균형 맞추는 전처리 중요

✅ kdeplot, countplot, boxplot
→ 각 변수 타입에 맞는 시각화로 이해도 상승

---

## **✨ 회고**

- 캐글을 필사하면서 EDA 분석의 **의도와 흐름**을 처음부터 끝까지 따라가며 익혔다.
- 단순 복붙이 아닌, 각 코드가 **왜 존재하는지** 해석하고 구현해보는 경험을 해봤다.
