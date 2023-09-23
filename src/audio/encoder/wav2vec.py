import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


class SpeechEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.input_values = None
        self.tokenizer = None
        self.model = None
        self.source_waveform = None
        self.embedding = None

    def load_model(self) -> None:
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name)

    def load_source_audio(self, source_audio_path: str) -> None:
        self.source_waveform, _ = torchaudio.load(source_audio_path)

    def tokenize_source_audio(self) -> None:
        self.input_values = self.tokenizer(self.source_waveform.squeeze().numpy(), return_tensors="pt").input_values

    def generate_embedding(self) -> None:
        with torch.no_grad():
            self.embedding = self.model(self.input_values)

    def get_embedding(self) -> torch.Tensor:
        return self.embedding

    def encode(self, source_audio_path: str) -> torch.Tensor:
        self.load_model()
        self.load_source_audio(source_audio_path)
        self.tokenize_source_audio()
        self.generate_embedding()
        return self.get_embedding()
