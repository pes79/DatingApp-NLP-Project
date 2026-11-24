"""
06_topic_analysis.py
불만 요인(키워드 사전) 분석 + 연도별 평점 추이
(부가정보 활용: 앱이름 = 수집 원천, 작성일 = 일시)

입력 : dating_app_cleaned_all.csv  (실제 수집 리뷰 정제본, 28,475건)
출력 :
  - results/topic_neg_vs_pos.csv        요인별 부정/긍정 리뷰 언급률과 배율
  - results/topic_by_app.csv            앱별 부정 리뷰 내 요인 언급률
  - results/category_by_app.csv         앱별 4대 범주(돈/신뢰/매칭/기술) 언급률
  - results/yearly_score.csv            앱별·연도별 리뷰 수와 평균 별점
  - figures/mining_1_topic_neg_vs_pos.png
  - figures/mining_2_topic_by_app.png
  - figures/mining_3_category_by_app.png
  - figures/mining_4_yearly_score.png
"""
import os
import platform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

# ------------------------------------------------------------------
# 0. 한글 폰트 설정 (Windows: 맑은 고딕 / Mac: AppleGothic / 그 외: Noto Sans CJK)
# ------------------------------------------------------------------
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans CJK JP']:
        if name in installed:
            plt.rc('font', family=name)
            break
plt.rcParams['axes.unicode_minus'] = False

INPUT = 'dating_app_cleaned_all.csv'
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
APPS = ['Glam', 'Tinder', 'Tantan']
APP_COLORS = {'Glam': '#E8956F', 'Tinder': '#5FA88E', 'Tantan': '#8497B8'}
MIN_REVIEWS_PER_YEAR = 50  # 연도별 추이에서 표본이 너무 적은 연도는 제외

# ------------------------------------------------------------------
# 1. 불만 요인 키워드 사전
#    리뷰에 키워드가 하나라도 포함되면 해당 요인을 '언급'한 것으로 본다.
#    (한 리뷰가 여러 요인을 동시에 언급할 수 있음)
# ------------------------------------------------------------------
TOPICS = {
    '과금·결제': ['결제', '과금', '유료', '젬', '하트', '구독', '비싸', '가격', '충전', '요금', '돈'],
    '환불': ['환불'],
    '알바·가짜계정': ['알바', '가짜', '허위', '매크로', '도용', '봇계정', '유령'],
    '스캠·사기': ['사기', '스캠', '피싱', '몸캠', '코인', '주식', '투자'],
    '조건만남·성매매': ['조건', '성매매', '업소', '유흥', '스폰'],
    '오류·버그': ['오류', '에러', '버그', '튕', '먹통', '렉', '로딩', '접속', '로그인', '인증'],
    '계정정지': ['정지', '영구', '밴 '],
    '매칭': ['매칭', '매치', '연결', '좋아요', '라이크'],
    '외모·심사': ['외모', '심사', '등급', '티어', '평가'],
    '성비·이성': ['성비', '남자', '여자'],
}

# 요인을 4개 큰 범주로 묶음
CATEGORIES = {
    '돈': ['과금·결제', '환불'],
    '신뢰': ['알바·가짜계정', '스캠·사기', '조건만남·성매매'],
    '매칭': ['매칭'],
    '기술': ['오류·버그'],
}

# ------------------------------------------------------------------
# 2. 데이터 로드 및 요인 태깅
# ------------------------------------------------------------------
print('불만 요인 분석을 시작합니다...')
df = pd.read_csv(INPUT)
df['리뷰내용'] = df['리뷰내용'].fillna('').astype(str)
print(f'분석 대상: {len(df):,}건')

for topic, words in TOPICS.items():
    df[topic] = df['리뷰내용'].apply(lambda text, ws=words: any(w in text for w in ws))

for cat, topics in CATEGORIES.items():
    df['[범주]' + cat] = df[topics].any(axis=1)

# 별점으로 감성 그룹 구분 (1~2: 부정, 3: 중립, 4~5: 긍정)
df['그룹'] = pd.cut(df['별점'], bins=[0, 2, 3, 5], labels=['부정', '중립', '긍정'])
neg = df[df['그룹'] == '부정']
pos = df[df['그룹'] == '긍정']
print(f'부정 리뷰(★1~2): {len(neg):,}건 / 긍정 리뷰(★4~5): {len(pos):,}건')

topic_cols = list(TOPICS)
cat_cols = ['[범주]' + c for c in CATEGORIES]

# ------------------------------------------------------------------
# 3. 요인별 부정 vs 긍정 언급률
# ------------------------------------------------------------------
neg_vs_pos = pd.DataFrame({
    '부정_언급률(%)': neg[topic_cols].mean() * 100,
    '긍정_언급률(%)': pos[topic_cols].mean() * 100,
})
neg_vs_pos['배율(부정/긍정)'] = neg_vs_pos['부정_언급률(%)'] / neg_vs_pos['긍정_언급률(%)']
neg_vs_pos = neg_vs_pos.sort_values('부정_언급률(%)', ascending=False).round(2)
neg_vs_pos.to_csv('results/topic_neg_vs_pos.csv', encoding='utf-8-sig')
print('\n[요인별 부정 vs 긍정 언급률]')
print(neg_vs_pos.round(1).to_string())

cat_total = (neg[cat_cols].mean() * 100).round(1)
cat_total.index = list(CATEGORIES)
print('\n[부정 리뷰 내 4대 범주 언급률(%)]')
print(cat_total.to_string())

# ------------------------------------------------------------------
# 4. 앱별 부정 리뷰 내 요인 / 범주 언급률
# ------------------------------------------------------------------
topic_by_app = (neg.groupby('앱이름')[topic_cols].mean() * 100).T[APPS].round(2)
topic_by_app.loc['부정 리뷰 수'] = neg['앱이름'].value_counts()[APPS]
topic_by_app.to_csv('results/topic_by_app.csv', encoding='utf-8-sig')
print('\n[앱별 부정 리뷰 내 요인 언급률(%)]')
print(topic_by_app.round(1).to_string())

category_by_app = (neg.groupby('앱이름')[cat_cols].mean() * 100).T[APPS].round(2)
category_by_app.index = list(CATEGORIES)
category_by_app.to_csv('results/category_by_app.csv', encoding='utf-8-sig')
print('\n[앱별 4대 범주 언급률(%)]')
print(category_by_app.round(1).to_string())

# ------------------------------------------------------------------
# 5. 연도별 평균 별점
# ------------------------------------------------------------------
df['연도'] = pd.to_datetime(df['작성일'], errors='coerce').dt.year
yearly = (df.dropna(subset=['연도'])
            .groupby(['앱이름', '연도'])['별점']
            .agg(리뷰수='size', 평균별점='mean')
            .reset_index())
yearly['연도'] = yearly['연도'].astype(int)
yearly['평균별점'] = yearly['평균별점'].round(2)
yearly.to_csv('results/yearly_score.csv', index=False, encoding='utf-8-sig')

# ------------------------------------------------------------------
# 6. 시각화
# ------------------------------------------------------------------
# [EDA 4] 요인별 부정 vs 긍정 언급률
plot_df = (neg_vs_pos[['부정_언급률(%)', '긍정_언급률(%)']]
           .rename(columns={'부정_언급률(%)': '부정 리뷰 (★1~2)', '긍정_언급률(%)': '긍정 리뷰 (★4~5)'})
           .reset_index(names='요인')
           .melt(id_vars='요인', var_name='그룹', value_name='언급률'))
plt.figure(figsize=(10, 6.5))
ax = sns.barplot(data=plot_df, y='요인', x='언급률', hue='그룹',
                 palette={'부정 리뷰 (★1~2)': '#D9603B', '긍정 리뷰 (★4~5)': '#7FA7C9'})
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)
plt.title('불만 요인별 언급률: 부정 리뷰 vs 긍정 리뷰')
plt.xlabel('리뷰 중 해당 요인을 언급한 비율 (%)')
plt.ylabel('')
plt.xlim(0, neg_vs_pos['부정_언급률(%)'].max() * 1.15)
plt.legend(title='', loc='lower right')
plt.savefig('figures/mining_1_topic_neg_vs_pos.png', dpi=150, bbox_inches='tight')
plt.close()

# [EDA 5] 앱별 요인 히트맵
heat = topic_by_app.drop(index='부정 리뷰 수').astype(float)
n_neg = neg['앱이름'].value_counts()[APPS]
heat.columns = [f'{a}\n(n={n_neg[a]:,})' for a in APPS]
plt.figure(figsize=(7.5, 7))
sns.heatmap(heat, annot=True, fmt='.1f', cmap='OrRd', cbar_kws={'label': '언급률 (%)'},
            linewidths=0.5, linecolor='white')
plt.title('앱별 부정 리뷰(★1~2)의 불만 요인 언급률 (%)')
plt.xlabel('')
plt.ylabel('')
plt.yticks(rotation=0)
plt.savefig('figures/mining_2_topic_by_app.png', dpi=150, bbox_inches='tight')
plt.close()

# [EDA 6] 앱별 4대 범주
cat_plot = category_by_app.reset_index(names='범주').melt(id_vars='범주', var_name='앱', value_name='언급률')
plt.figure(figsize=(9, 5.5))
ax = sns.barplot(data=cat_plot, x='범주', y='언급률', hue='앱', palette=APP_COLORS)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=2, fontsize=9)
plt.title('앱별 부정 리뷰의 4대 불만 범주')
plt.xlabel('')
plt.ylabel('부정 리뷰 중 언급 비율 (%)')
plt.ylim(0, category_by_app.values.max() * 1.15)
plt.legend(title='')
plt.savefig('figures/mining_3_category_by_app.png', dpi=150, bbox_inches='tight')
plt.close()

# [EDA 7] 연도별 평균 별점 추이
yr = yearly[yearly['리뷰수'] >= MIN_REVIEWS_PER_YEAR]
plt.figure(figsize=(10, 5.5))
for app in APPS:
    d = yr[yr['앱이름'] == app]
    plt.plot(d['연도'], d['평균별점'], marker='o', linewidth=2, label=app, color=APP_COLORS[app])
plt.axhline(3, color='gray', linestyle='--', linewidth=1)
plt.title(f'앱별 연도별 평균 별점 추이 (연간 리뷰 {MIN_REVIEWS_PER_YEAR}건 이상)')
plt.xlabel('연도')
plt.ylabel('평균 별점')
plt.ylim(1, 5)
plt.xticks(sorted(yr['연도'].unique()))
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('figures/mining_4_yearly_score.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n완료! results/ 에 CSV 4개, figures/ 에 그래프 4개(mining_1 ~ mining_4)가 저장되었습니다.')
