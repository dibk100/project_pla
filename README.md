# project_PLA
phoneme-level-asr-model
- Subject :
- Purpose :
- Goal :
- Role :

## 📋 To-Do
- [x] ✅ Data : 데이터 확인 및 분석  
- [ ] 🔄 Task03 : 음소 인식 모델 테스트   

## ⚙️ 파이프라인
```
Audio (.wav)
   ↓
log-Mel Spectrogram
   ↓
[ Whisper Encoder ]
   ↓
Latent vector
   ↓
[ Custom Decoder ]
   ↓
발음 그대로의 텍스트 ("배고푼거 갓타요")
```

## 📁 Folder Structure
```
project/
│
├── data/
│   ├──raw_data
│   │   ├── 1. 한국일반
│   │   │   ├── VN10QC226_VN0005_20210801.wav
│   │   │   └── ...
│   │   ├── 2. 한국생활I
│   │   └── ...
│   └── ...
├── sketch/
│   ├── whisper_encoder.py    ← Whisper encoder만 로드
│   ├── decoder.py            ← Transformer decoder 구현
│   └── sketch.py             ← 학습/실행 파이프라인 (통합 스크립트)

```

<!--
### To-Do

- [ ] 🔄 작업 중 : 품질 예측 성능 평가 코드 개선 중
- [ ] ✅ 완료됨 : 데이터셋 병합 및 전처리 (2025-05-23)
- [ ] 📌🕒 다음 할 일 : inference 모듈 디버깅
-->
