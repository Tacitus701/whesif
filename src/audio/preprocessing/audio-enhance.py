import librosa
import noisereduce as nr
import soundfile as sf
import torch


class ReduceNoise:

    ALLOWED_EXTENSIONS = ".wav"

    def __init__(self, input_audio_path: str, output_audio_path: str) -> None:
        assert input_audio_path.lower().endswith(self.ALLOWED_EXTENSIONS)
        if output_audio_path:
            assert output_audio_path.lower().endswith(self.ALLOWED_EXTENSIONS)
        self.input_audio_path = input_audio_path
        self.output_audio_path = output_audio_path

        self.load()

    def load(self) -> None:
        self.audio_data, self.sample_rate = librosa.load(self.input_audio_path, sr=None)

    def reduce_noise(self) -> None:
        self.reduced_audio_data = nr.reduce_noise(y=self.audio_data, sr=self.sample_rate)

    def save_as_wav(self) -> None:
        sf.write(self.output_audio_path, self.reduced_audio_data, self.sample_rate)

    def save_as_torch_tensor(self) -> torch.Tensor:
        return torch.tensor(self.reduced_audio_data, dtype=torch.float32)
