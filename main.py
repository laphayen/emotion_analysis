
# main.py
# 감정 분석을 위한 모델과 토크나이저를 불러오는 코드입니다.

import kss
import json
from emotion_classifier import EmotionClassifier

def analyze_text(text: str):
    classifier = EmotionClassifier()
    results = []

    print("[문장 분리 중...]")
    sentences = kss.split_sentences(text)

    print("[감정 분석 시작]")
    for sentence in sentences:
        emotion = classifier.predict(sentence)
        print(f"{sentence} → 감정: {emotion}")
        results.append({
            "sentence": sentence,
            "emotion": emotion
        })

    return results

if __name__ == "__main__":
    sample_text = """
    깊은 숲 속에 귀여운 토끼가 살고 있었어요.
    하지만 어느 날 무서운 늑대가 나타났어요.
    토끼는 너무 무서워서 울음을 터뜨렸어요.
    다행히 곰 아저씨가 나타나 늑대를 쫓아냈어요.
    """

    result = analyze_text(sample_text)

    with open("emotion_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[감정 분석 결과 저장 완료 → emotion_result.json]")