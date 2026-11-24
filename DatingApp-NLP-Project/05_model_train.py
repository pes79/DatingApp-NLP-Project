"""
05_model_train.py
라벨링 완료 데이터(dating_app_labeling_2000.csv)를 수업 KoELECTRA 코드(KOELECTRA-Finetune.py)의
입력 형식으로 변환하고, 학습 데이터 EDA를 수행한다.

  1) 라벨 표기 통일 : '관련없음' → '관계없음' (오타 통일), 허용 라벨 외 값 확인
  2) 형식 변환      : 수업 코드가 읽는 ratings_train2.txt (탭 구분, 열: id / document / label) 생성
                      라벨 번호: 0=부정, 1=긍정, 2=중립, 3=관계없음
                      (수업 예제의 0=부정, 1=긍정 의미를 그대로 유지하고 2, 3을 추가)
  3) 분할 미리보기  : 수업 코드는 내부에서 train_test_split(test_size=0.2, random_state=2026)으로
                      학습/검증을 나눈다. 같은 방식으로 나눠 라벨 분포를 미리 확인하고,
                      참고용으로 train_data.csv / test_data.csv 도 저장한다.
  4) 학습 데이터 EDA : figures/train_1 ~ train_5

출력: ratings_train2.txt, train_data.csv, test_data.csv, figures/train_*.png
"""
import numpy as np
import os
import platform
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from sklearn.model_selection import train_test_split

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

LABELS = ['긍정', '부정', '중립', '관계없음']
LABEL_COLORS = {'긍정': '#2471A3', '부정': '#C0392B', '중립': '#95A5A6', '관계없음': '#F4D03F'}
APPS = ['Glam', 'Tinder', 'Tantan']

print('KoELECTRA 학습 데이터 준비를 시작합니다...')

# =========================================================
# 1. 라벨 점검
# =========================================================
df = pd.read_csv('dating_app_labeling_2000.csv')
df = df.dropna(subset=['리뷰내용', '감성_라벨'])
df['감성_라벨'] = df['감성_라벨'].str.strip().replace({'관련없음': '관계없음'})
invalid = df[~df['감성_라벨'].isin(LABELS)]
if len(invalid):
    print(f'※ 허용되지 않은 라벨 {len(invalid)}건:', invalid['감성_라벨'].unique())
df = df[df['감성_라벨'].isin(LABELS)].copy()
df['리뷰길이'] = df['리뷰내용'].astype(str).str.len()
print(f'라벨링 데이터: {len(df):,}건')
print(df['감성_라벨'].value_counts().reindex(LABELS).to_string())

# =========================================================
# 2. 수업 코드 입력 형식으로 변환 (ratings_train2.txt)
# =========================================================
LABEL2ID = {'부정': 0, '긍정': 1, '중립': 2, '관계없음': 3}
out = pd.DataFrame({
    'id': range(1, len(df) + 1),
    # 탭·줄바꿈이 있으면 탭 구분 파일이 깨지므로 공백 하나로 바꾼다
    'document': df['리뷰내용'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip(),
    'label': df['감성_라벨'].map(LABEL2ID),
})
out.to_csv('ratings_train2.txt', sep='\t', index=False)
print('\nratings_train2.txt 저장 완료 (수업 코드 입력 파일)')
print('라벨 번호:', LABEL2ID)

# =========================================================
# 2-1. 수업 코드와 같은 방식으로 분할 미리보기
# =========================================================
check = pd.read_csv('ratings_train2.txt', sep='\t').dropna(axis=0)   # 수업 코드와 같은 읽기 방식
idx = np.arange(len(check))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=2026)
df = df.reset_index(drop=True)
train_df, test_df = df.iloc[tr_idx], df.iloc[va_idx]
cols = ['앱이름', '리뷰내용', '별점', '작성일', '감성_라벨']
train_df[cols].to_csv('train_data.csv', index=False, encoding='utf-8-sig')
test_df[cols].to_csv('test_data.csv', index=False, encoding='utf-8-sig')
print(f'\n수업 코드 분할(random_state=2026) → 학습: {len(train_df):,}건 / 검증: {len(test_df):,}건')
split_tbl = pd.DataFrame({'Train': train_df['감성_라벨'].value_counts(),
                          'Test': test_df['감성_라벨'].value_counts()}).reindex(LABELS).fillna(0).astype(int)
print(split_tbl.to_string())

# =========================================================
# 3. 학습 데이터 EDA
# =========================================================
# [train_1] 라벨 분포 + Train/Test 분할 비교
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
vc = df['감성_라벨'].value_counts().reindex(LABELS)
bars = axes[0].bar(LABELS, vc.values, color=[LABEL_COLORS[l] for l in LABELS])
for b, v in zip(bars, vc.values):
    axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:,}\n({v / len(df):.1%})', ha='center', va='bottom', fontsize=9)
axes[0].set_title(f'감성 라벨 분포 (n={len(df):,})')
axes[0].set_ylim(0, vc.max() * 1.25)
axes[0].set_ylabel('리뷰 수')
(split_tbl / split_tbl.sum() * 100).plot(kind='bar', ax=axes[1], color=['#5D6D7E', '#AEB6BF'], rot=0)
axes[1].set_title('Train / Valid 라벨 비율 (수업 코드 분할, seed=2026)')
axes[1].set_ylabel('비율 (%)')
axes[1].set_xlabel('')
plt.tight_layout()
plt.savefig('figures/train_1_label_dist.png', dpi=150, bbox_inches='tight')
plt.close()

# [train_2] 앱별 라벨 구성
ct = pd.crosstab(df['앱이름'], df['감성_라벨'], normalize='index').reindex(index=APPS, columns=LABELS).fillna(0) * 100
ax = ct.plot(kind='barh', stacked=True, figsize=(10, 3.5), color=[LABEL_COLORS[l] for l in LABELS])
for i, app in enumerate(APPS):
    left = 0
    for l in LABELS:
        v = ct.loc[app, l]
        if v > 5:
            ax.text(left + v / 2, i, f'{v:.0f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        left += v
n_app = df['앱이름'].value_counts()
ax.set_yticklabels([f'{a} (n={n_app[a]})' for a in APPS])
plt.title('앱별 감성 라벨 구성')
plt.xlabel('비율 (%)')
plt.ylabel('')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.gca().invert_yaxis()
plt.savefig('figures/train_2_label_by_app.png', dpi=150, bbox_inches='tight')
plt.close()

# [train_3] 별점 × 라벨 (별점과 텍스트 감성이 어긋나는 정도)
st = pd.crosstab(df['별점'], df['감성_라벨']).reindex(columns=LABELS, fill_value=0)
plt.figure(figsize=(7, 4.5))
sns.heatmap(st, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
plt.title('별점 × 감성 라벨 (건수)')
plt.xlabel('감성 라벨')
plt.ylabel('별점')
plt.yticks(rotation=0)
plt.savefig('figures/train_3_score_vs_label.png', dpi=150, bbox_inches='tight')
plt.close()
mismatch = ((df['별점'] >= 4) & (df['감성_라벨'] == '부정')) | ((df['별점'] <= 2) & (df['감성_라벨'] == '긍정'))
print(f'\n별점과 라벨 방향이 반대인 리뷰: {mismatch.sum()}건 ({mismatch.mean():.1%})')

# [train_4] 라벨별 리뷰 길이
plt.figure(figsize=(8, 4.5))
sns.boxplot(data=df, x='감성_라벨', y='리뷰길이', order=LABELS, hue='감성_라벨',
            palette=LABEL_COLORS, showfliers=False, legend=False)
meds = df.groupby('감성_라벨')['리뷰길이'].median().reindex(LABELS)
for i, m in enumerate(meds):
    plt.text(i, m + 2, f'{m:.0f}', ha='center', fontsize=10, fontweight='bold')
plt.title('감성 라벨별 리뷰 길이 (숫자=중앙값)')
plt.xlabel('')
plt.ylabel('리뷰 길이 (글자 수)')
plt.savefig('figures/train_4_length_by_label.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n라벨별 리뷰 길이')
print(df.groupby('감성_라벨')['리뷰길이'].agg(['mean', 'median']).reindex(LABELS).round(1).to_string())

# [train_5] 라벨별 상위 명사 (긍정 vs 부정)
try:
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    stop = set('어플 사용 이용 생각 정도 리뷰 사람 진짜 정말 그냥 하나 이거 때문 지금 처음 틴더 글램 탄탄'.split())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lab in zip(axes, ['긍정', '부정']):
        texts = df.loc[df['감성_라벨'] == lab, '리뷰내용'].astype(str).tolist()
        cnt = Counter()
        for toks in kiwi.tokenize(texts):
            cnt.update({t.form for t in toks if t.tag in ('NNG', 'NNP') and len(t.form) > 1 and t.form not in stop})
        top = cnt.most_common(15)[::-1]
        n = len(texts)
        ax.barh([w for w, _ in top], [c / n * 100 for _, c in top], color=LABEL_COLORS[lab])
        ax.set_title(f'{lab} 리뷰 상위 명사 (n={n})')
        ax.set_xlabel('해당 단어가 등장한 리뷰 비율 (%)')
        print(f'\n[{lab}] 상위 명사:', ', '.join(f'{w}({c})' for w, c in top[::-1]))
    plt.tight_layout()
    plt.savefig('figures/train_5_top_nouns.png', dpi=150, bbox_inches='tight')
    plt.close()
except ImportError:
    print('kiwipiepy가 없어 train_5(상위 명사) 그래프는 건너뜁니다. (pip install kiwipiepy)')

print('\n완료! 같은 폴더에서 KOELECTRA-Finetune.py 를 실행하면 우리 데이터로 학습됩니다.')
