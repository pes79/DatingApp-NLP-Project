import pandas as pd

print("1차 자동 라벨링을 시작합니다. 잠시만 기다려주세요...")

df = pd.read_csv('dating_app_labeling_2000.csv')


def auto_label(row):
    text = str(row['리뷰내용'])
    score = row['별점']

    if len(text) < 2 or text in ['ㅇㅇ', 'ㅋㅋ', 'ㅎㅎ', '굿']:
        return '관계없음'

    negative_words = ['알바', '사기', '환불', '결제', '쓰레기', '최악', '돈', '과금', '오류', '정지', '차단', '광고']
    if any(word in text for word in negative_words):
        return '부정'

    if score >= 4:
        return '긍정'
    elif score <= 2:
        return '부정'
    else:
        return '중립'


df['감성_라벨'] = df.apply(auto_label, axis=1)

df.to_csv('dating_app_labeling_2000.csv', index=False, encoding='utf-8-sig')
print("자동 라벨링 완료! 파일을 다시 열어서 결과를 확인해 보세요.")