from transformers import WhisperModel, WhisperProcessor
import torch

class WhisperEncoderWrapper(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-small"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.encoder.gradient_checkpointing = False  # 필요시 off

    def forward(self, input_waveform, sampling_rate=16000):
        # Whisper 전처리: waveform → log-Mel
        inputs = self.processor(
            input_waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        input_features = inputs["input_features"]  # [1, 80, 3000]

        with torch.no_grad():  # 학습 가능하게 하려면 제거
            encoder_outputs = self.model.encoder(input_features)
        return encoder_outputs.last_hidden_state  # [B, T, H]


import torch
import torchaudio
import whisper

class WhisperEncoder(torch.nn.Module):
    def __init__(self, model_size="base"):
        super().__init__()
        self.model = whisper.load_model(model_size)
        self.encoder = self.model.encoder
        self.tokenizer = self.model.tokenizer

    def forward(self, mel):
        # mel: (batch_size, 80, time)
        return self.encoder(mel)

    def preprocess_audio(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)  # (80, time)
        mel = mel.unsqueeze(0)  # (1, 80, time)
        return mel


if __name__ == "__main__":
    encoder = WhisperEncoder()
    mel = encoder.preprocess_audio("../data/audio1.wav")
    with torch.no_grad():
        encoded = encoder(mel)
    print("Encoded shape:", encoded.shape)  # (1, time, 768) for base model