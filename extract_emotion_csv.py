import json
import pandas as pd
from glob import glob
import os

def extract_speaker_emotions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    emotion = data.get('info', {}).get('speaker_emotion', '중립')
    utterances = data.get('utterances', [])

    records = [
        (utt['text'], emotion)
        for utt in utterances
        if utt.get('role') == 'speaker' and 'text' in utt
    ]

    print(f"✅ {os.path.basename(json_path)} → {len(records)}개 발화 추출됨 (감정: {emotion})")
    return records

# 경로 수정: 현재 디렉토리 기준 또는 상대경로로
file_paths = glob('data/Empathy_*.json')  # 또는 './data/*.json'

# 디버깅: 파일 목록 출력
print(f"📁 총 파일 수: {len(file_paths)}개")
for path in file_paths[:3]:  # 처음 3개만 미리 출력
    print(f" - {path}")

# 전체 데이터 수집
dataset = []
for path in file_paths:
    dataset.extend(extract_speaker_emotions(path))

# 결과 저장
df = pd.DataFrame(dataset, columns=['sentence', 'emotion'])

if df.empty:
    print("⚠️ 데이터셋이 비어 있습니다. 경로 또는 JSON 구조를 확인하세요.")
else:
    df.to_csv('emotion_dataset.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {len(df)}개 문장 → emotion_dataset.csv")