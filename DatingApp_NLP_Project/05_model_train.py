import pandas as pd
from sklearn.model_selection import train_test_split

print("KoELECTRA 모델 학습 파이프라인을 준비합니다...")

df = pd.read_csv('dating_app_labeling_2000.csv')

df = df.dropna(subset=['리뷰내용', '감성_라벨'])

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train_data.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('test_data.csv', index=False, encoding='utf-8-sig')

print(f"학습용 데이터: {len(train_df)}건")
print(f"테스트용 데이터: {len(test_df)}건")
print("데이터 분할 완료! 모델 학습을 위한 준비가 끝났습니다.")