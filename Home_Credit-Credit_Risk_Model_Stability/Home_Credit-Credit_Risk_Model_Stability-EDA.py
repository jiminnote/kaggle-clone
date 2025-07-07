

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

base_path = '/kaggle/input/home-credit-credit-risk-model-stability/csv_files'
train_path = f'{base_path}/train'
test_path = f'{base_path}/test'

print("=== 새로운 Home Credit 데이터셋 분석 ===")

# 1. 기본 테이블 로드 (Base tables)
def load_base_tables():
    try:
        train_base = pd.read_csv(f'{train_path}/train_base.csv')
        test_base = pd.read_csv(f'{test_path}/test_base.csv')
        
        print(f"Train base: {train_base.shape}")
        print(f"Test base: {test_base.shape}")
        
        return train_base, test_base
    except Exception as e:
        print(f"Base table 로드 오류: {e}")
        return None, None

train_base, test_base = load_base_tables()

if train_base is not None:
    print(f"\n=== Base Table 정보 ===")
    print(f"Train cases: {len(train_base):,}")
    print(f"Test cases: {len(test_base):,}")
    print(f"Train columns: {list(train_base.columns)}")
    print(f"Test columns: {list(test_base.columns)}")
    
    # 타겟 분포 확인
    if 'target' in train_base.columns:
        target_dist = train_base['target'].value_counts()
        print(f"\nTarget distribution: \n{target_dist}")
        print(f"Default rate: {train_base['target'].mean():.4f}")

# 2. 사용 가능한 모든 테이블 탐색
def explore_available_tables():
    print("\n=== 사용 가능한 테이블 탐색 ===")
    
    train_files = glob.glob(f'{train_path}/*.csv')
    test_files = glob.glob(f'{test_path}/*.csv')
    
    print(f"Train 파일 수: {len(train_files)}")
    print(f"Test 파일 수: {len(test_files)}")
    
    # 파일명 패턴 분석
    train_file_names = [os.path.basename(f) for f in train_files]
    test_file_names = [os.path.basename(f) for f in test_files]
    
    print(f"\nTrain 파일들 (처음 10개):")
    for i, fname in enumerate(train_file_names[:10]):
        print(f"  {i+1}. {fname}")
    
    print(f"\nTest 파일들 (처음 10개):")
    for i, fname in enumerate(test_file_names[:10]):
        print(f"  {i+1}. {fname}")
    
    return train_file_names, test_file_names

train_files, test_files = explore_available_tables()

# 3. 주요 테이블 그룹별 로드 함수
def load_table_group(group_name, max_files=3):
    train_group_files = [f for f in train_files if group_name in f]
    test_group_files = [f for f in test_files if group_name in f]
    
    print(f"\n=== {group_name} 그룹 분석 ===")
    print(f"Train 파일: {len(train_group_files)}개")
    print(f"Test 파일: {len(test_group_files)}개")
    
    train_dfs = []
    test_dfs = []
    
    # 최대 max_files개까지만 로드 (메모리 절약)
    for i, fname in enumerate(train_group_files[:max_files]):
        try:
            df = pd.read_csv(f'{train_path}/{fname}')
            train_dfs.append(df)
            print(f"  ✓ {fname}: {df.shape}")
        except Exception as e:
            print(f"  ✗ {fname}: 로드 실패 - {e}")
    
    for i, fname in enumerate(test_group_files[:max_files]):
        try:
            df = pd.read_csv(f'{test_path}/{fname}')
            test_dfs.append(df)
            print(f"  ✓ {fname}: {df.shape}")
        except Exception as e:
            print(f"  ✗ {fname}: 로드 실패 - {e}")
    
    return train_dfs, test_dfs

# 4. 주요 테이블 그룹들 분석
table_groups = ['static_0', 'static_cb_0', 'applprev_1', 'person_1', 'credit_bureau_a_1']

loaded_tables = {}
for group in table_groups:
    train_dfs, test_dfs = load_table_group(group, max_files=2)
    loaded_tables[group] = {'train': train_dfs, 'test': test_dfs}

# 5. 기본 안정성 분석 - Base Table 중심
def analyze_base_stability(train_base, test_base):
    if train_base is None or test_base is None:
        print("Base table이 없어 분석을 건너뜁니다.")
        return
    
    print("\n=== Base Table 안정성 분석 ===")
    
    # 공통 컬럼 확인
    common_cols = set(train_base.columns) & set(test_base.columns)
    train_only = set(train_base.columns) - set(test_base.columns)
    test_only = set(test_base.columns) - set(train_base.columns)
    
    print(f"공통 컬럼: {len(common_cols)}개")
    print(f"Train에만 있는 컬럼: {len(train_only)}개 - {list(train_only)}")
    print(f"Test에만 있는 컬럼: {len(test_only)}개 - {list(test_only)}")
    
    # WEEK_NUM 분포 분석 (시간 안정성의 핵심)
    if 'WEEK_NUM' in common_cols:
        print(f"\n=== WEEK_NUM 분포 분석 ===")
        train_weeks = train_base['WEEK_NUM'].describe()
        test_weeks = test_base['WEEK_NUM'].describe()
        
        print(f"Train WEEK_NUM: {train_weeks['min']:.0f} ~ {train_weeks['max']:.0f}")
        print(f"Test WEEK_NUM: {test_weeks['min']:.0f} ~ {test_weeks['max']:.0f}")
        
        # WEEK_NUM별 타겟 분포 (시간에 따른 타겟 안정성)
        if 'target' in train_base.columns:
            weekly_target = train_base.groupby('WEEK_NUM')['target'].agg(['count', 'mean']).reset_index()
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # 주별 케이스 수
            axes[0].plot(weekly_target['WEEK_NUM'], weekly_target['count'], marker='o', alpha=0.7)
            axes[0].set_title('Cases per Week')
            axes[0].set_xlabel('Week Number')
            axes[0].set_ylabel('Number of Cases')
            axes[0].grid(True, alpha=0.3)
            
            # 주별 타겟 비율
            axes[1].plot(weekly_target['WEEK_NUM'], weekly_target['mean'], marker='o', alpha=0.7, color='red')
            axes[1].set_title('Default Rate by Week')
            axes[1].set_xlabel('Week Number')
            axes[1].set_ylabel('Default Rate')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # 수치형 변수들의 기본 분포 비교
    numeric_cols = [col for col in common_cols if train_base[col].dtype in ['int64', 'float64']]
    numeric_cols = [col for col in numeric_cols if col not in ['case_id', 'WEEK_NUM']]
    
    if len(numeric_cols) > 0:
        print(f"\n수치형 변수 {len(numeric_cols)}개 발견")
        
        # 처음 6개 변수만 분포 비교
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(numeric_cols[:6]):
            if i < 6:
                train_vals = train_base[col].dropna()
                test_vals = test_base[col].dropna()
                
                if len(train_vals) > 0 and len(test_vals) > 0:
                    axes[i].hist(train_vals, bins=50, alpha=0.6, label='Train', density=True)
                    axes[i].hist(test_vals, bins=50, alpha=0.6, label='Test', density=True)
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

analyze_base_stability(train_base, test_base)

# 6. Population Stability Index (PSI) 계산
def calculate_psi(expected, actual, bins=10):
    def psi_score(expected_array, actual_array, bins):
        # 유효한 값만 사용
        expected_clean = expected_array[~np.isnan(expected_array)]
        actual_clean = actual_array[~np.isnan(actual_array)]
        
        if len(expected_clean) == 0 or len(actual_clean) == 0:
            return np.nan, None, None
        
        # 구간 설정
        breakpoints = np.arange(0, bins + 1) / bins * 100
        breakpoints = np.percentile(expected_clean, breakpoints)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # 각 구간별 비율 계산
        expected_percents = pd.cut(expected_clean, breakpoints).value_counts().sort_index() / len(expected_clean)
        actual_percents = pd.cut(actual_clean, breakpoints).value_counts().sort_index() / len(actual_clean)
        
        # 0인 값 처리 (작은 값으로 대체)
        expected_percents = expected_percents.replace(0, 1e-6)
        actual_percents = actual_percents.replace(0, 1e-6)
        
        # PSI 계산
        psi = sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi, expected_percents, actual_percents
    
    return psi_score(expected, actual, bins)

# 7. 다중 테이블 안정성 분석
def analyze_multi_table_stability(loaded_tables):
    print("\n=== 다중 테이블 안정성 분석 ===")
    
    stability_summary = []
    
    for group_name, tables in loaded_tables.items():
        print(f"\n--- {group_name} 그룹 ---")
        
        train_dfs = tables['train']
        test_dfs = tables['test']
        
        if len(train_dfs) > 0 and len(test_dfs) > 0:
            # 첫 번째 테이블로 분석
            train_df = train_dfs[0]
            test_df = test_dfs[0]
            
            print(f"분석 테이블: Train {train_df.shape}, Test {test_df.shape}")
            
            # 공통 수치형 컬럼 찾기
            common_cols = set(train_df.columns) & set(test_df.columns)
            numeric_cols = [col for col in common_cols 
                          if train_df[col].dtype in ['int64', 'float64'] 
                          and col not in ['case_id', 'WEEK_NUM', 'num_group1', 'num_group2']]
            
            if len(numeric_cols) > 0:
                print(f"분석할 수치형 변수: {len(numeric_cols)}개")
                
                # PSI 계산
                for col in numeric_cols[:5]:  # 처음 5개만
                    try:
                        train_vals = train_df[col].values
                        test_vals = test_df[col].values
                        
                        psi, _, _ = calculate_psi(train_vals, test_vals)
                        
                        if not np.isnan(psi):
                            stability = 'Stable' if psi < 0.1 else 'Moderately Unstable' if psi < 0.25 else 'Highly Unstable'
                            stability_summary.append({
                                'table_group': group_name,
                                'feature': col,
                                'psi': psi,
                                'stability': stability
                            })
                            print(f"  {col}: PSI = {psi:.4f} ({stability})")
                    except Exception as e:
                        print(f"  {col}: 계산 오류 - {e}")
    
    # 안정성 요약
    if len(stability_summary) > 0:
        stability_df = pd.DataFrame(stability_summary)
        
        print(f"\n=== 전체 안정성 요약 ===")
        stability_counts = stability_df['stability'].value_counts()
        print(stability_counts)
        
        # 상위 불안정 피처들
        top_unstable = stability_df.nlargest(10, 'psi')
        print(f"\n상위 10개 불안정 피처:")
        print(top_unstable[['table_group', 'feature', 'psi', 'stability']])
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PSI 분포
        stability_df['psi'].hist(bins=20, ax=axes[0])
        axes[0].set_title('PSI Distribution')
        axes[0].set_xlabel('PSI Value')
        axes[0].set_ylabel('Count')
        axes[0].axvline(x=0.1, color='orange', linestyle='--', label='Moderate Threshold')
        axes[0].axvline(x=0.25, color='red', linestyle='--', label='High Threshold')
        axes[0].legend()
        
        # 안정성 분포
        stability_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
        axes[1].set_title('Feature Stability Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return stability_df
    
    return pd.DataFrame()

stability_results = analyze_multi_table_stability(loaded_tables)

# 8. 시간별 데이터 볼륨 안정성 분석
def analyze_temporal_volume_stability():
    if train_base is None:
        return
    
    print("\n=== 시간별 데이터 볼륨 안정성 ===")
    
    if 'WEEK_NUM' in train_base.columns:
        # 주별 데이터 볼륨
        weekly_volume = train_base.groupby('WEEK_NUM').size()
        
        print(f"전체 주 수: {len(weekly_volume)}")
        print(f"평균 주별 케이스: {weekly_volume.mean():.1f}")
        print(f"주별 케이스 표준편차: {weekly_volume.std():.1f}")
        print(f"변동계수 (CV): {weekly_volume.std()/weekly_volume.mean():.3f}")
        
        # 시각화
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 주별 볼륨 추이
        weekly_volume.plot(ax=axes[0], marker='o')
        axes[0].set_title('Weekly Case Volume Trend')
        axes[0].set_xlabel('Week Number')
        axes[0].set_ylabel('Number of Cases')
        axes[0].grid(True, alpha=0.3)
        
        # 볼륨 분포
        weekly_volume.hist(bins=20, ax=axes[1])
        axes[1].set_title('Weekly Volume Distribution')
        axes[1].set_xlabel('Cases per Week')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(weekly_volume.mean(), color='red', linestyle='--', label='Mean')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # 안정성 경고
        cv = weekly_volume.std() / weekly_volume.mean()
        if cv > 0.3:
            print(f"⚠️ 경고: 주별 데이터 볼륨 변동이 큽니다 (CV: {cv:.3f})")
        else:
            print(f"✅ 주별 데이터 볼륨이 안정적입니다 (CV: {cv:.3f})")

analyze_temporal_volume_stability()

# 9. 종합 안정성 리포트
def generate_stability_report(train_base, test_base, stability_results):
    print("\n" + "="*60)
    print("HOME CREDIT 모델 안정성 종합 리포트")
    print("="*60)
    
    # 1. 데이터 개요
    print(f"\n1. 데이터 개요")
    if train_base is not None and test_base is not None:
        print(f"   • Train 케이스: {len(train_base):,}건")
        print(f"   • Test 케이스: {len(test_base):,}건")
        
        if 'target' in train_base.columns:
            print(f"   • 전체 부실률: {train_base['target'].mean():.4f} ({train_base['target'].mean()*100:.2f}%)")
        
        if 'WEEK_NUM' in train_base.columns:
            train_weeks = train_base['WEEK_NUM'].nunique()
            test_weeks = test_base['WEEK_NUM'].nunique()
            print(f"   • Train 기간: {train_weeks}주")
            print(f"   • Test 기간: {test_weeks}주")
    
    # 2. 안정성 평가
    print(f"\n⚖️ 2. 안정성 평가")
    if len(stability_results) > 0:
        stable_count = len(stability_results[stability_results['stability'] == 'Stable'])
        moderate_count = len(stability_results[stability_results['stability'] == 'Moderately Unstable'])
        unstable_count = len(stability_results[stability_results['stability'] == 'Highly Unstable'])
        total_features = len(stability_results)
        
        print(f"   • 안정적 피처: {stable_count}/{total_features} ({stable_count/total_features*100:.1f}%)")
        print(f"   • 약간 불안정: {moderate_count}/{total_features} ({moderate_count/total_features*100:.1f}%)")
        print(f"   • 매우 불안정: {unstable_count}/{total_features} ({unstable_count/total_features*100:.1f}%)")
        
        # 가장 불안정한 피처들
        if unstable_count > 0:
            worst_features = stability_results.nlargest(3, 'psi')
            print(f"\n   ⚠️ 가장 불안정한 피처들:")
            for _, row in worst_features.iterrows():
                print(f"      - {row['feature']} (PSI: {row['psi']:.3f}, Table: {row['table_group']})")
    else:
        print(f"   • 안정성 분석 결과 없음 (데이터 로드 문제)")
    
    # 3. 위험 요소
    print(f"\n🚨 3. 주요 위험 요소")
    risks = []
    
    if len(stability_results) > 0:
        high_risk_features = len(stability_results[stability_results['psi'] > 0.25])
        if high_risk_features > 0:
            risks.append(f"높은 PSI 값을 가진 {high_risk_features}개 피처")
    
    if train_base is not None and 'WEEK_NUM' in train_base.columns:
        weekly_volume = train_base.groupby('WEEK_NUM').size()
        cv = weekly_volume.std() / weekly_volume.mean()
        if cv > 0.3:
            risks.append(f"높은 데이터 볼륨 변동성 (CV: {cv:.3f})")
    
    if len(risks) > 0:
        for risk in risks:
            print(f"   • {risk}")
    else:
        print(f"   • 주요 위험 요소 없음")
    
    # 4. 권장사항
    print(f"\n 4. 권장사항")
    print(f"   • 정기적인 모델 성능 모니터링 (최소 월 1회)")
    print(f"   • PSI > 0.25인 피처들에 대한 특별 관리")
    print(f"   • 새로운 데이터 소스 추가 시 안정성 사전 검증")
    print(f"   • A/B 테스트를 통한 모델 업데이트 검증")
    print(f"   • 백테스팅을 통한 시간별 성능 안정성 확인")
    
    # 5. 다음 단계
    print(f"\n 5. 다음 단계")
    print(f"   • 피처 엔지니어링 최적화")
    print(f"   • 앙상블 모델을 통한 안정성 향상")
    print(f"   • 도메인 전문가와의 피처 검토")
    print(f"   • 프로덕션 모니터링 시스템 구축")

generate_stability_report(train_base, test_base, stability_results)

print(f"\n" + "="*60)
print(" EDA 완료! 모델 안정성 분석이 완료되었습니다.")
print("="*60)