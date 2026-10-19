import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 한글 폰트 설정 (깨짐 방지)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

print("5만 건 최종 데이터에 대한 EDA 시각화를 시작합니다...")

# 52,500건 데이터 로드
df = pd.read_csv('dating_app_50k_final.csv')

# 결측치(빈칸)를 빈 문자열로 처리하여 에러 방지
df['리뷰내용'] = df['리뷰내용'].fillna("")

# 1. 앱별 리뷰 수 (eda_1_app_count_final.png)
plt.figure(figsize=(10, 6))
# 최신 seaborn 문법에 맞춰 hue와 legend=False 추가
sns.countplot(data=df, x='앱이름', hue='앱이름', order=df['앱이름'].value_counts().index, palette='viridis', legend=False)
plt.title('앱별 리뷰 데이터 분포 (총 52,500건)')
plt.xlabel('앱 이름')
plt.ylabel('리뷰 수')
plt.savefig('eda_1_app_count_final.png', bbox_inches='tight')
plt.close()

# 2. 별점 분포 (eda_2_score_dist_final.png)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='별점', hue='별점', palette='magma', legend=False)
plt.title('전체 리뷰 별점 분포')
plt.xlabel('별점')
plt.ylabel('빈도수')
plt.savefig('eda_2_score_dist_final.png', bbox_inches='tight')
plt.close()

# 3. 리뷰 길이 분포 (eda_3_length_dist_final.png)
df['리뷰길이'] = df['리뷰내용'].astype(str).apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='리뷰길이', bins=50, kde=True, color='teal')
plt.title('리뷰 길이 분포')
plt.xlabel('리뷰 길이 (글자 수)')
plt.ylabel('빈도수')
plt.xlim(0, 300)
plt.savefig('eda_3_length_dist_final.png', bbox_inches='tight')
plt.close()

print("시각화 완료! eda_1, eda_2, eda_3_final.png 파일이 정상적으로 생성되었습니다.")