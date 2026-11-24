"""
03_eda.py
(1) 수집 원본 데이터 EDA  : dating_app_50k_eunsun.csv (실제 수집 35,939건)
(2) 정제 데이터 기초 통계  : dating_app_cleaned_all.csv (28,475건)

출력 그래프 (figures/ 폴더)
  raw_1_app_count.png        앱별 리뷰 수
  raw_2_score_dist.png       별점 분포
  raw_3_app_score.png        앱별 별점 구성 비율
  raw_4_yearly_volume.png    연도별 리뷰 수 (앱별)
  raw_5_length_dist.png      리뷰 길이 분포
  raw_6_length_by_score.png  별점별 리뷰 길이 (박스플롯)
"""
import os
import platform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

# ---------------- 한글 폰트 ----------------
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

os.makedirs('figures', exist_ok=True)
APPS = ['Glam', 'Tinder', 'Tantan']
APP_COLORS = {'Glam': '#E8956F', 'Tinder': '#5FA88E', 'Tantan': '#8497B8'}

# =========================================================
# 1. 수집 원본 데이터 EDA
# =========================================================
raw = pd.read_csv('dating_app_50k_eunsun.csv')
raw['작성일'] = pd.to_datetime(raw['작성일'], errors='coerce')
raw['연도'] = raw['작성일'].dt.year
raw['리뷰길이'] = raw['리뷰내용'].fillna('').astype(str).str.len()

print('=' * 50)
print('[1] 수집 원본 데이터')
print('=' * 50)
print(f'전체 건수      : {len(raw):,}')
print(f'컬럼           : {list(raw.columns)}')
print(f'결측치         :\n{raw.isna().sum()}')
print(f'리뷰내용 중복  : {raw["리뷰내용"].duplicated().sum():,}건')
print(f'작성일 범위    : {raw["작성일"].min()} ~ {raw["작성일"].max()}')
print(f'별점 범위      : {raw["별점"].min()} ~ {raw["별점"].max()}')
print('\n앱별 건수 / 평균 별점')
print(raw.groupby('앱이름')['별점'].agg(['size', 'mean']).round(2))
print('\n별점 분포')
print(raw['별점'].value_counts().sort_index())
print('\n리뷰 길이 기술통계')
print(raw['리뷰길이'].describe().round(1))

# [raw_1] 앱별 리뷰 수
plt.figure(figsize=(7, 4.5))
order = raw['앱이름'].value_counts().index
ax = sns.countplot(data=raw, x='앱이름', order=order, hue='앱이름', palette=APP_COLORS, legend=False)
for c in ax.containers:
    ax.bar_label(c, fmt='{:,.0f}', padding=2)
plt.title(f'앱별 수집 리뷰 수 (원본 {len(raw):,}건)')
plt.xlabel('')
plt.ylabel('리뷰 수')
plt.savefig('figures/raw_1_app_count.png', dpi=150, bbox_inches='tight')
plt.close()

# [raw_2] 별점 분포
plt.figure(figsize=(7, 4.5))
ax = sns.countplot(data=raw, x='별점', color='#8497B8')
total = len(raw)
for c in ax.containers:
    ax.bar_label(c, labels=[f'{v.get_height():,.0f}\n({v.get_height() / total:.1%})' for v in c], padding=2, fontsize=9)
plt.title('별점 분포 (원본)')
plt.ylabel('리뷰 수')
plt.ylim(0, raw['별점'].value_counts().max() * 1.2)
plt.savefig('figures/raw_2_score_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# [raw_3] 앱별 별점 구성 비율 (100% 누적 막대)
ratio = pd.crosstab(raw['앱이름'], raw['별점'], normalize='index').loc[APPS] * 100
ax = ratio.plot(kind='barh', stacked=True, figsize=(9, 3.8),
                color=['#C0392B', '#E59866', '#D5D8DC', '#85C1E9', '#2471A3'])
for i, app in enumerate(APPS):
    left = 0
    for s in ratio.columns:
        v = ratio.loc[app, s]
        if v > 6:
            ax.text(left + v / 2, i, f'{v:.0f}%', ha='center', va='center', fontsize=9)
        left += v
plt.title('앱별 별점 구성 비율 (원본)')
plt.xlabel('비율 (%)')
plt.ylabel('')
plt.legend(title='별점', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.gca().invert_yaxis()
plt.savefig('figures/raw_3_app_score.png', dpi=150, bbox_inches='tight')
plt.close()

# [raw_4] 연도별 리뷰 수
yearly = raw.groupby(['연도', '앱이름']).size().unstack(fill_value=0)[APPS]
ax = yearly.plot(kind='bar', stacked=True, figsize=(10, 4.5), color=[APP_COLORS[a] for a in APPS])
plt.title('연도별 수집 리뷰 수 (원본)')
plt.xlabel('작성 연도')
plt.ylabel('리뷰 수')
plt.xticks(rotation=0)
plt.legend(title='')
plt.savefig('figures/raw_4_yearly_volume.png', dpi=150, bbox_inches='tight')
plt.close()

# [raw_5] 리뷰 길이 분포
plt.figure(figsize=(9, 4.5))
sns.histplot(raw['리뷰길이'], bins=60, color='#E8956F', binrange=(0, 300))
med = raw['리뷰길이'].median()
plt.axvline(med, color='black', linestyle='--', linewidth=1)
plt.text(med + 3, plt.ylim()[1] * 0.9, f'중앙값 {med:.0f}자', fontsize=10)
plt.title('리뷰 길이 분포 (원본, 300자 이하 구간 표시)')
plt.xlabel('리뷰 길이 (글자 수)')
plt.ylabel('리뷰 수')
plt.savefig('figures/raw_5_length_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# [raw_6] 별점별 리뷰 길이
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=raw, x='별점', y='리뷰길이', color='#85C1E9', showfliers=False)
meds = raw.groupby('별점')['리뷰길이'].median()
for i, m in enumerate(meds):
    plt.text(i, m + 2, f'{m:.0f}', ha='center', fontsize=9, fontweight='bold')
plt.title('별점별 리뷰 길이 (이상치 제외 박스플롯, 숫자=중앙값)')
plt.xlabel('별점')
plt.ylabel('리뷰 길이 (글자 수)')
plt.savefig('figures/raw_6_length_by_score.png', dpi=150, bbox_inches='tight')
plt.close()

# =========================================================
# 2. 정제 데이터 기초 통계 (비교용 출력)
# =========================================================
clean = pd.read_csv('dating_app_cleaned_all.csv')
clean['리뷰길이'] = clean['리뷰내용'].astype(str).str.len()
print('\n' + '=' * 50)
print('[2] 정제 데이터')
print('=' * 50)
print(f'전체 건수 : {len(clean):,}  (원본 대비 {len(clean) / len(raw):.1%})')
print(clean.groupby('앱이름')['별점'].agg(['size', 'mean']).round(2))
print(clean.groupby('별점')['리뷰길이'].agg(['mean', 'median']).round(1))

print('\nEDA 완료! figures/ 폴더에 raw_1 ~ raw_6 그래프가 저장되었습니다.')
