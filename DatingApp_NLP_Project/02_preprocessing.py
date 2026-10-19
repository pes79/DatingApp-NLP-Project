import pandas as pd
import re

print("데이터 정제 및 라벨링 파일 생성을 시작합니다...")

file_path = 'dating_app_50k_eunsun.csv'
df = pd.DataFrame(pd.read_csv(file_path))

print(f"원본 데이터 수: {len(df)}건")

df = df.dropna(subset=['리뷰내용'])
df = df.drop_duplicates(subset=['리뷰내용'], keep='first')

def clean_text(text):
    text = str(text)
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s]', '', text)
    return text.strip()

df['리뷰내용'] = df['리뷰내용'].apply(clean_text)

df = df[df['리뷰내용'] != '']
print(f"정제 완료 후 남은 데이터 수: {len(df)}건")

df.to_csv('dating_app_cleaned_all.csv', index=False, encoding='utf-8-sig')

df_sample = df.sample(n=2000, random_state=42)

df_sample['감성_라벨'] = ''

df_sample.to_csv('dating_app_labeling_2000.csv', index=False, encoding='utf-8-sig')

print("\n작업 완료!")
print("- 전체 정제 파일: 'dating_app_cleaned_all.csv'")
print("- 라벨링 작업용 파일: 'dating_app_labeling_2000.csv'")