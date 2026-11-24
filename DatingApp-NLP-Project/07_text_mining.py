"""
07_text_mining.py
부정 리뷰(★1~2)의 불만 주제를 사전 없이 찾아내는 텍스트 마이닝
  (1) LDA 토픽 모델링   : 부정 리뷰를 몇 개의 '주제 묶음'으로 자동 군집
  (2) 키워드 동시출현 네트워크 : 같은 리뷰에 함께 등장하는 명사들의 연결 구조

방법 요약 (보고서 §결과 분석에 설명)
  - 형태소 분석 : Kiwi(kiwipiepy)로 일반명사(NNG)·고유명사(NNP) 중 2글자 이상만 추출
  - 분석 대상   : 정제 데이터 중 10자 초과 & 별점 1~2점 리뷰 (내용이 있는 부정 리뷰)
  - LDA         : 문서-단어 행렬(CountVectorizer) → sklearn LatentDirichletAllocation
                  토픽 수 k는 3~10 범위에서 perplexity를 비교하고, 해석 가능성을 함께 고려해 결정
  - 네트워크    : 빈도 상위 40개 명사, 한 리뷰 안에 함께 나오면 연결(가중치=동시출현 횟수)
                  Greedy modularity로 커뮤니티(키워드 군집)를 찾아 색으로 구분

설치: pip install kiwipiepy scikit-learn networkx
출력: figures/mining_5 ~ mining_9, results/lda_*.csv, results/network_*.csv
"""
import itertools
import os
import platform
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from matplotlib import font_manager
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- 한글 폰트 ----------------
if platform.system() == 'Windows':
    FONT = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    FONT = 'AppleGothic'
else:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    FONT = next((n for n in ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans CJK JP'] if n in installed), None)
if FONT:
    plt.rc('font', family=FONT)
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

SEED = 42
N_TOPICS = 6           # perplexity 비교 + 해석 가능성으로 결정 (아래 1-3 참고)
MIN_LEN = 10           # 10자 이하 리뷰는 내용 분석에서 제외
TOP_N_NODES = 40       # 네트워크에 표시할 키워드 수
APPS = ['Glam', 'Tinder', 'Tantan']

# 분석에 의미가 없는 명사(불용어)와 표기 통일
STOPWORDS = set('''
어플 사용 이용 생각 정도 리뷰 경우 처음 가능 하루 마디 아무것 이상 틴더 글램 탄탄 사람 진짜 정말 그냥
하나 이거 때문 자체 다음 지금 계속 동안 이번 요즘 우리 저희 본인 기분 부분 상태 소리 얘기 이후 이전
다른 혹시 최근 와중
'''.split())
NORMALIZE = {'결재': '결제', '메세지': '메시지'}

# 토픽 이름 (LDA 결과의 상위 단어를 읽고 연구자가 붙인 이름)
TOPIC_NAMES = {
    0: 'T1 대화 단절·알바 의심',
    1: 'T2 외국인·사기 계정',
    2: 'T3 앱 오류·광고',
    3: 'T4 계정 정지·고객센터',
    4: 'T5 과금 구조 불만',
    5: 'T6 자동결제·환불',
}

# =========================================================
# 0. 데이터 준비 및 형태소 분석
# =========================================================
df = pd.read_csv('dating_app_cleaned_all.csv')
df['리뷰내용'] = df['리뷰내용'].astype(str)
df = df[df['리뷰내용'].str.len() > MIN_LEN].copy()
df['연도'] = pd.to_datetime(df['작성일'], errors='coerce').dt.year
print(f'10자 초과 리뷰: {len(df):,}건')

print('형태소 분석 중... (1~2분 소요)')
kiwi = Kiwi()
nouns = []
for tokens in kiwi.tokenize(df['리뷰내용'].tolist()):
    words = []
    for t in tokens:
        if t.tag in ('NNG', 'NNP') and len(t.form) > 1:
            w = NORMALIZE.get(t.form, t.form)
            if w not in STOPWORDS:
                words.append(w)
    nouns.append(words)
df['명사'] = nouns
df['doc'] = df['명사'].apply(' '.join)

neg = df[(df['별점'] <= 2) & (df['doc'].str.len() > 0)].copy()
print(f'분석 대상 부정 리뷰(★1~2, 명사 1개 이상): {len(neg):,}건')

# =========================================================
# 1. LDA 토픽 모델링
# =========================================================
vec = CountVectorizer(min_df=10, max_df=0.5, token_pattern=r'\S+')
X = vec.fit_transform(neg['doc'])
vocab = np.array(vec.get_feature_names_out())
print(f'문서-단어 행렬: {X.shape[0]:,}개 문서 × {X.shape[1]:,}개 단어')

# 1-1. 토픽 수 비교 (perplexity: 낮을수록 모델이 데이터를 잘 설명)
ks = list(range(3, 11))
perps = []
for k in ks:
    m = LatentDirichletAllocation(n_components=k, random_state=SEED, learning_method='batch', max_iter=30)
    m.fit(X)
    perps.append(m.perplexity(X))
    print(f'  k={k:2d}  perplexity={perps[-1]:.1f}')
pd.DataFrame({'k': ks, 'perplexity': perps}).to_csv('results/lda_perplexity.csv', index=False, encoding='utf-8-sig')

plt.figure(figsize=(7, 4))
plt.plot(ks, perps, marker='o', color='#8497B8')
plt.scatter([N_TOPICS], [perps[ks.index(N_TOPICS)]], s=150, facecolors='none', edgecolors='#D9603B', linewidths=2,
            label=f'선택 k={N_TOPICS}')
plt.title('토픽 수(k)별 Perplexity')
plt.xlabel('토픽 수 k')
plt.ylabel('Perplexity (낮을수록 좋음)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures/mining_5_lda_perplexity.png', dpi=150, bbox_inches='tight')
plt.close()

# 1-2. 최종 모델
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=SEED, learning_method='batch', max_iter=30)
doc_topic = lda.fit_transform(X)
neg['토픽'] = doc_topic.argmax(axis=1)
neg['토픽명'] = neg['토픽'].map(TOPIC_NAMES)

rows = []
for i, comp in enumerate(lda.components_):
    top = comp.argsort()[::-1][:15]
    weights = comp[top] / comp.sum()
    rows.append({'토픽': TOPIC_NAMES[i],
                 '문서 비율(%)': round((neg['토픽'] == i).mean() * 100, 1),
                 '상위 단어': ', '.join(vocab[top])})
    print(f'  {TOPIC_NAMES[i]:<16} {rows[-1]["문서 비율(%)"]:5.1f}%  {", ".join(vocab[top[:10]])}')
pd.DataFrame(rows).to_csv('results/lda_topics.csv', index=False, encoding='utf-8-sig')

# 1-3. 토픽별 상위 단어 그래프
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
colors = ['#5FA88E', '#C0392B', '#8497B8', '#7D6608', '#E8956F', '#8E44AD']
for i, ax in enumerate(axes.flat):
    comp = lda.components_[i]
    top = comp.argsort()[::-1][:10]
    ax.barh(vocab[top][::-1], (comp[top] / comp.sum())[::-1] * 100, color=colors[i])
    share = (neg['토픽'] == i).mean() * 100
    ax.set_title(f'{TOPIC_NAMES[i]}  ({share:.1f}%)', fontsize=12)
    ax.set_xlabel('토픽 내 단어 비중 (%)')
plt.suptitle(f'부정 리뷰 LDA 토픽 (k={N_TOPICS}, n={len(neg):,})', fontsize=14)
plt.tight_layout()
plt.savefig('figures/mining_6_lda_topics.png', dpi=150, bbox_inches='tight')
plt.close()

# 1-4. 앱별 토픽 구성 (부가정보: 수집 원천)
app_topic = pd.crosstab(neg['앱이름'], neg['토픽명'], normalize='index').loc[APPS] * 100
app_topic.round(1).to_csv('results/lda_topic_by_app.csv', encoding='utf-8-sig')
print('\n[앱별 토픽 비율(%)]')
print(app_topic.round(1).to_string())

ax = app_topic.plot(kind='barh', stacked=True, figsize=(11, 3.8), color=colors)
for i, app in enumerate(APPS):
    left = 0
    for t in app_topic.columns:
        v = app_topic.loc[app, t]
        if v > 5:
            ax.text(left + v / 2, i, f'{v:.0f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        left += v
plt.title('앱별 부정 리뷰의 토픽 구성 (%)')
plt.xlabel('비율 (%)')
plt.ylabel('')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.gca().invert_yaxis()
plt.savefig('figures/mining_7_topic_by_app.png', dpi=150, bbox_inches='tight')
plt.close()

# 1-5. 연도별 토픽 추이 (부가정보: 일시)
#  전체 데이터로 연도별 추이를 보면 '연도마다 앱 구성이 달라지는 효과'가 섞인다
#  (예: Glam은 2019년부터, Tantan은 2023년 이후 리뷰가 거의 없음).
#  따라서 앱별로 나누어 연간 부정 리뷰 100건 이상인 연도만 비교한다.
neg[['앱이름', '연도', '별점', '토픽명']].to_csv('results/lda_doc_topics.csv', index=False, encoding='utf-8-sig')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, app in zip(axes, APPS):
    sub = neg[neg['앱이름'] == app]
    vc = sub['연도'].value_counts()
    years = sorted(vc[vc >= 100].index)
    yt = pd.crosstab(sub['연도'], sub['토픽명'], normalize='index').reindex(columns=list(TOPIC_NAMES.values()),
                                                                            fill_value=0).loc[years] * 100
    yt.round(1).to_csv(f'results/lda_topic_by_year_{app}.csv', encoding='utf-8-sig')
    print(f'\n[{app} 연도별 토픽 비율(%)]')
    print(yt.round(1).to_string())
    for i, t in enumerate(yt.columns):
        ax.plot(yt.index.astype(int), yt[t], marker='o', label=t, color=colors[i], linewidth=2)
    ax.set_title(f'{app}', fontsize=13)
    ax.set_xlabel('연도')
    ax.set_xticks(yt.index.astype(int))
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('부정 리뷰 중 비율 (%)')
axes[-1].legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.suptitle('앱별·연도별 부정 리뷰 토픽 비율 (연간 부정 리뷰 100건 이상인 연도)', fontsize=14)
plt.tight_layout()
plt.savefig('figures/mining_8_topic_by_year.png', dpi=150, bbox_inches='tight')
plt.close()

# =========================================================
# 2. 키워드 동시출현 네트워크
# =========================================================
docs = [list(set(ws)) for ws in neg['명사']]           # 한 리뷰 안에서 중복 단어는 1번만
freq = Counter(w for ws in docs for w in ws)
top_words = [w for w, _ in freq.most_common(TOP_N_NODES)]
top_set = set(top_words)

pair_count = Counter()
for ws in docs:
    ws = sorted(w for w in ws if w in top_set)
    for a, b in itertools.combinations(ws, 2):
        pair_count[(a, b)] += 1

# 너무 촘촘하지 않도록 상위 동시출현 쌍만 사용
edges = pair_count.most_common(80)
G = nx.Graph()
for w in top_words:
    G.add_node(w, freq=freq[w])
for (a, b), c in edges:
    G.add_edge(a, b, weight=c)
G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])

communities = list(greedy_modularity_communities(G, weight='weight'))
comm_of = {n: i for i, c in enumerate(communities) for n in c}
deg_cent = nx.degree_centrality(G)

pd.DataFrame([{'단어1': a, '단어2': b, '동시출현': c} for (a, b), c in edges]) \
    .to_csv('results/network_edges.csv', index=False, encoding='utf-8-sig')
node_df = pd.DataFrame([{'단어': n, '빈도': freq[n], '연결중심성': round(deg_cent[n], 3), '군집': comm_of[n] + 1}
                        for n in G.nodes]).sort_values('연결중심성', ascending=False)
node_df.to_csv('results/network_nodes.csv', index=False, encoding='utf-8-sig')
print('\n[키워드 군집]')
for i, c in enumerate(communities):
    print(f'  군집 {i + 1}: {", ".join(sorted(c, key=lambda w: -freq[w]))}')
print('\n[연결 중심성 상위 10]')
print(node_df.head(10).to_string(index=False))

plt.figure(figsize=(13, 11))
pos = nx.kamada_kawai_layout(G)  # 노드 간 거리를 고르게 배치
palette = ['#E8956F', '#5FA88E', '#8497B8', '#C0392B', '#8E44AD', '#7D6608']
max_w = max(d['weight'] for _, _, d in G.edges(data=True))
nx.draw_networkx_edges(G, pos, width=[0.5 + 6 * d['weight'] / max_w for _, _, d in G.edges(data=True)],
                       alpha=0.35, edge_color='gray')
nx.draw_networkx_nodes(G, pos, node_size=[200 + 60 * np.sqrt(freq[n]) for n in G.nodes],
                       node_color=[palette[comm_of[n] % len(palette)] for n in G.nodes], alpha=0.85)
nx.draw_networkx_labels(G, pos, font_size=12, font_family=FONT or 'sans-serif', font_weight='bold')
plt.title(f'부정 리뷰 키워드 동시출현 네트워크 (상위 {TOP_N_NODES}개 명사, 상위 80개 연결)\n'
          '원 크기 = 등장 빈도, 선 굵기 = 동시출현 횟수, 색 = 키워드 군집', fontsize=13)
plt.axis('off')
plt.savefig('figures/mining_9_keyword_network.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n완료! figures/mining_5 ~ mining_9 그래프와 results/ 의 CSV가 저장되었습니다.')
