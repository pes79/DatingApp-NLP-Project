"""
09_evaluate.py
수업 코드(KOELECTRA-Finetune.py)로 학습·저장된 모델을 불러와 검증 데이터 400건을 예측한다.
(수업 코드는 정확도만 출력하므로, 혼동 행렬·클래스별 F1을 보기 위해 별도 스크립트로 평가한다.
 수업 코드는 전혀 수정하지 않는다.)

  - 데이터 분할은 수업 코드와 똑같이 재현한다:
      ratings_train2.txt → dropna → train_test_split(test_size=0.2, random_state=2026)
  - 모델: 수업 코드가 저장한 'koelectra-base-discriminator' 폴더
  - 토크나이저: monologg/koelectra-base-v3-discriminator (수업 코드와 동일, max_length=256)

실행 순서: python 05_model_train.py → python KOELECTRA-Finetune.py → python 09_evaluate.py → python 08_plot_training.py
출력:
  results/test_predictions.csv     label, pred (08_plot_training.py가 혼동 행렬을 그릴 때 사용)
  results/valid_predictions_detail.csv  리뷰 원문 + 정답 + 예측 + 맞음 여부 (오류 분석용)
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import ElectraForSequenceClassification, ElectraTokenizer

os.makedirs('results', exist_ok=True)
ID2LABEL = {0: '부정', 1: '긍정', 2: '중립', 3: '관계없음'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('사용하는 장치:', device)

# 1. 수업 코드와 같은 방식으로 데이터 읽기 및 분할 재현
dataset = pd.read_csv('ratings_train2.txt', sep='\t').dropna(axis=0)
text = list(dataset['document'].values)
label = dataset['label'].values
idx = np.arange(len(text))
_, valid_idx = train_test_split(idx, test_size=0.2, random_state=2026)
valid_text = [text[i] for i in valid_idx]
valid_label = label[valid_idx]
print(f'검증 데이터: {len(valid_text)}건')

# 2. 저장된 모델 불러오기
tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
model = ElectraForSequenceClassification.from_pretrained('koelectra-base-discriminator')
model.to(device)
model.eval()

# 3. 예측
preds = []
batch_size = 32
for s in range(0, len(valid_text), batch_size):
    batch = valid_text[s:s + batch_size]
    enc = tokenizer(batch, truncation=True, padding='max_length', max_length=256,
                    add_special_tokens=True, return_tensors='pt').to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
preds = np.array(preds)
print(f'검증 정확도: {(preds == valid_label).mean():.4f}  (수업 코드 마지막 epoch의 valid ACC와 같아야 정상)')

# 4. 저장
pd.DataFrame({'label': [ID2LABEL[int(v)] for v in valid_label],
              'pred': [ID2LABEL[int(v)] for v in preds]}).to_csv(
    'results/test_predictions.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'리뷰내용': valid_text,
              '정답': [ID2LABEL[int(v)] for v in valid_label],
              '예측': [ID2LABEL[int(v)] for v in preds],
              '맞음': preds == valid_label}).to_csv(
    'results/valid_predictions_detail.csv', index=False, encoding='utf-8-sig')
print('results/test_predictions.csv, results/valid_predictions_detail.csv 저장 완료')
print('이제 python 08_plot_training.py 를 실행하세요.')
