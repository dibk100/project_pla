# sketck
테스트 공간


[whisper_encoder.py] - 음향 특성 추출   
- Whisper encoder만 로드해서 audio -> embedding 반환
    - 오디오 신호 → log-Mel Spectrogram → Encoder 인코딩
- output: [Batch, Time_steps, Hidden_dim] 형태의 음향 임베딩

[decoder.py] - 발음 기반 텍스트 생성   
- Transformer 기반 character-level decoder   
- 인코더 출력(음향 임베딩)과 이전에 생성된 토큰을 입력으로 받아 다음 토큰 예측하기
- goal: 발음 그대로의 텍스트 시퀀스 생성 (정규화 없이)

[sketch.py]   
음성 데이터 → log-Mel → Encoder → Decoder → 발음 기반 문자 예측   
   

> Issue.   
> 표준 단어 단위가 아닌, 발음 표현을 최대한 살릴 수 있는 문자 단위 토크나이저..