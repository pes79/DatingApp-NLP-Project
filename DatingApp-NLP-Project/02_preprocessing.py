"""
02_preprocessing.py
수집 원본(dating_app_50k_eunsun.csv)을 정제하고, 라벨링용 2,000건을 무작위 추출한다.

정제 단계
  0) 데이터 범위 검증 : 별점 1~5, 작성일 파싱 가능, 앱이름 3종
  1) 결측치 제거      : 리뷰내용이 비어 있는(NaN) 행
  2) 중복 제거        : 리뷰내용이 완전히 같은 행은 첫 번째만 유지 (데이터 누수 방지)
  3) 텍스트 정제      : 한글(완성형·자모)·영문·숫자·공백 외 문자 제거
  4) 빈 문자열 제거   : 정제 후 아무 글자도 남지 않은 행

※ 길이 기준 필터(10자 이하 제거 등)는 이 단계에서 적용하지 않는다.
   - 감성 분류 모델은 실제로 들어오는 짧은 리뷰("굿", "ㅎ")도 처리해야 하며,
     '관계없음' 클래스가 바로 이런 짧은 리뷰로 구성되기 때문이다.
   - 대신 내용 분석이 목적인 텍스트 마이닝(07_text_mining.py)에서는 10자 이하 리뷰를 제외한다.

출력
  dating_app_cleaned_all.csv     : 정제 데이터 전체
  dating_app_labeling_2000.csv   : 라벨링용 무작위 샘플 2,000건
"""
import re

import pandas as pd

print('데이터 정제 및 라벨링 파일 생성을 시작합니다...')

df = pd.read_csv('dating_app_50k_eunsun.csv')
n0 = len(df)
print(f'[0] 원본 데이터 수: {n0:,}건')

# ---------- 0) 데이터 범위 검증 ----------
df['작성일_dt'] = pd.to_datetime(df['작성일'], errors='coerce')
bad_score = ~df['별점'].between(1, 5)
bad_date = df['작성일_dt'].isna()
print(f'    별점 범위 밖(1~5 아님): {bad_score.sum()}건 / 작성일 파싱 실패: {bad_date.sum()}건')
print(f'    작성일 범위: {df["작성일_dt"].min()} ~ {df["작성일_dt"].max()}')
print(f'    앱 종류: {sorted(df["앱이름"].unique())}')
df = df[~bad_score & ~bad_date].drop(columns='작성일_dt')

# ---------- 1) 결측치 제거 ----------
before = len(df)
df = df.dropna(subset=['리뷰내용'])
print(f'[1] 결측치 제거: -{before - len(df):,}건 → {len(df):,}건')

# ---------- 2) 중복 제거 ----------
before = len(df)
dup_mask = df.duplicated(subset=['리뷰내용'], keep='first')
print('    제거되는 중복 리뷰 중 빈도 상위 5개:')
print(df.loc[dup_mask, '리뷰내용'].value_counts().head(5).to_string())
df = df[~dup_mask]
print(f'[2] 중복 제거: -{before - len(df):,}건 → {len(df):,}건')


# ---------- 3) 텍스트 정제 ----------
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', '', text)
    return text.strip()


df['리뷰내용'] = df['리뷰내용'].apply(clean_text)

# ---------- 4) 빈 문자열 제거 ----------
before = len(df)
df = df[df['리뷰내용'] != '']
print(f'[3-4] 특수문자 정제 후 빈 리뷰 제거: -{before - len(df):,}건 → {len(df):,}건')
print(f'최종 정제 데이터: {len(df):,}건 (원본 대비 {len(df) / n0:.1%} 유지)')

df.to_csv('dating_app_cleaned_all.csv', index=False, encoding='utf-8-sig')

# ---------- 라벨링용 샘플 ----------
df_sample = df.sample(n=2000, random_state=42).copy()
df_sample['감성_라벨'] = ''
df_sample.to_csv('dating_app_labeling_2000.csv', index=False, encoding='utf-8-sig')

print('\n작업 완료!')
print("- 전체 정제 파일: 'dating_app_cleaned_all.csv'")
print("- 라벨링 작업용 파일: 'dating_app_labeling_2000.csv'")
