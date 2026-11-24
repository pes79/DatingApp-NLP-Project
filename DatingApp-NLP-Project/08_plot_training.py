"""
08_plot_training.py
수업 KoELECTRA 코드로 학습한 뒤, epoch별 loss·정확도 그래프와 평가 결과를 그린다.
(학습 코드 자체는 수정하지 않는다. 학습 중 출력된 값을 아래 CSV로 옮겨 적으면 된다.)

입력 1) results/training_log.txt   ← 수업 코드의 "학습 결과 요약" 출력 줄을 그대로 붙여 넣은 파일 (가장 쉬움)
   또는 results/training_log.csv   ← 학습 로그를 직접 옮겨 적은 파일
        epoch,train_loss,train_acc,val_loss,val_acc
        1,1.0213,0.5612,0.8834,0.6650
        2,...
        * 수업 코드가 val 값을 출력하지 않으면 해당 열은 비워 둬도 된다.
        * 정확도는 0~1 또는 0~100 어느 쪽이든 자동 인식한다.

입력 2) results/test_predictions.csv ← 검증 400건의 정답/예측 (09_evaluate.py가 자동 생성)
        label,pred
        부정,부정
        긍정,중립 ...

출력
  figures/model_1_loss_acc.png          epoch별 loss / accuracy
  figures/model_2_confusion_matrix.png  혼동 행렬 (입력 2가 있을 때)
  results/classification_report.csv     클래스별 precision / recall / F1
"""
import os
import platform

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

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
os.makedirs('results', exist_ok=True)

LABELS = ['긍정', '부정', '중립', '관계없음']

# ---------------- 1. epoch별 loss / accuracy ----------------
# (가장 쉬운 방법) 수업 코드가 마지막에 출력하는 "===== 학습 결과 요약 =====" 아래 줄들을
# 그대로 복사해 results/training_log.txt 에 붙여 넣으면 자동으로 읽는다.
#   Epoch: 1, Train loss: 0.8123, Train ACC: 0.7012, valid ACC: 0.6950
if os.path.exists('results/training_log.txt'):
    import re

    rows = []
    for line in open('results/training_log.txt', encoding='utf-8'):
        m = re.search(r'Epoch:\s*(\d+),\s*Train loss:\s*([\d.]+),\s*Train ACC:\s*([\d.]+),\s*valid ACC:\s*([\d.]+)', line)
        if m:
            rows.append({'epoch': int(m.group(1)), 'train_loss': float(m.group(2)),
                         'train_acc': float(m.group(3)), 'val_loss': None, 'val_acc': float(m.group(4))})
    log = pd.DataFrame(rows)
    log.to_csv('results/training_log.csv', index=False, encoding='utf-8-sig')
    print(f'results/training_log.txt 에서 {len(log)}개 epoch를 읽었습니다.')
else:
    log = pd.read_csv('results/training_log.csv')
log['val_loss'] = pd.to_numeric(log.get('val_loss'), errors='coerce')
for col in ['train_acc', 'val_acc']:
    if col in log and log[col].notna().any() and log[col].max() > 1:
        log[col] = log[col] / 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(log['epoch'], log['train_loss'], marker='o', label='Train loss', color='#2471A3')
if 'val_loss' in log and log['val_loss'].notna().any():
    axes[0].plot(log['epoch'], log['val_loss'], marker='s', label='Validation/Test loss', color='#C0392B')
axes[0].set_title('Epoch별 Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(log['epoch'], log['train_acc'] * 100, marker='o', label='Train accuracy', color='#2471A3')
if 'val_acc' in log and log['val_acc'].notna().any():
    axes[1].plot(log['epoch'], log['val_acc'] * 100, marker='s', label='Validation/Test accuracy', color='#C0392B')
    best = log.loc[log['val_acc'].idxmax()]
    axes[1].annotate(f"최고 {best['val_acc'] * 100:.1f}% (epoch {int(best['epoch'])})",
                     (best['epoch'], best['val_acc'] * 100), textcoords='offset points', xytext=(0, -20),
                     ha='center', fontsize=9)
axes[1].set_title('Epoch별 Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)
for ax in axes:
    ax.set_xticks(log['epoch'])
plt.suptitle('KoELECTRA 학습 곡선 (수업 배포 코드)', fontsize=13)
plt.tight_layout()
plt.savefig('figures/model_1_loss_acc.png', dpi=150, bbox_inches='tight')
plt.close()
print('figures/model_1_loss_acc.png 저장 완료')
print(log.to_string(index=False))

# ---------------- 2. 혼동 행렬 / 분류 리포트 ----------------
pred_path = 'results/test_predictions.csv'
if os.path.exists(pred_path):
    from sklearn.metrics import classification_report, confusion_matrix

    p = pd.read_csv(pred_path)
    labels = [l for l in LABELS if l in set(p['label']) | set(p['pred'])]
    cm = confusion_matrix(p['label'], p['pred'], labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title('혼동 행렬 (Test)')
    plt.xlabel('예측')
    plt.ylabel('정답')
    plt.savefig('figures/model_2_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    rep = classification_report(p['label'], p['pred'], labels=labels, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.round(3)
    rep_df.to_csv('results/classification_report.csv', encoding='utf-8-sig')
    print('\n[분류 리포트]')
    print(rep_df.to_string())
else:
    print(f'{pred_path} 가 없어 혼동 행렬은 건너뜁니다.')
