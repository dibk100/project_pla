import torch
from whisper_encoder import WhisperEncoder
from decoder import CharTransformerDecoder

# --- 간단 vocab (음절 수준, 실제에 맞게 수정 필요) ---
vocab = ['<pad>', '<s>', '</s>', '<unk>'] + list("가나다라마바사아자차카타파하거갓타요...")
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화
encoder = WhisperEncoder().to(device)
decoder = CharTransformerDecoder(vocab_size=len(vocab), d_model=512).to(device)

encoder.eval()
decoder.eval()

# 오디오 불러와서 전처리
audio_path = "../data/audio1.wav"
mel = encoder.preprocess_audio(audio_path).to(device)

with torch.no_grad():
    memory = encoder(mel)  # Whisper encoder 출력 (B, T, H)

# 디코더 greedy 디코딩 (최대 30자)
max_len = 30
start_token = char2idx['<s>']
end_token = char2idx['</s>']

tgt_tokens = torch.full((1, 1), start_token, dtype=torch.long, device=device)
output_tokens = []

for _ in range(max_len):
    logits = decoder(tgt_tokens, memory)  # (1, seq_len, vocab_size)
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).item()

    if next_token == end_token:
        break

    output_tokens.append(next_token)
    tgt_tokens = torch.cat([tgt_tokens, torch.tensor([[next_token]], device=device)], dim=1)

# 결과 출력
predicted_text = "".join(idx2char[idx] for idx in output_tokens)
print("=== 음성 인식 (발음 텍스트) 결과 ===")
print(predicted_text)
