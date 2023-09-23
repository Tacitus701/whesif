import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "facebook/wav2vec2-base-960h"
test_audio_file = "data/audio/Projet_fini.wav"
train_audio_file = "data/audio/macron.wav"
source_audio_file = "data/audio/Projet_fini.wav"
target_audio_file = "data/audio/split/macron_split_54.wav"


def main():
    """ "
    speech_encoder = SpeechEncoder(model_name)

    test_embedding = speech_encoder.encode(test_audio_file)
    train_embedding = speech_encoder.encode(train_audio_file)

    # TEST
    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    inputs = processor(
        text=[
            "Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other
            interests such as playing tic tac toe."],
        return_tensors="pt",
    )

    # Regarder la signature de la fonction model.forward() -> indice sur le dataset que la classe
    # Trainer de transformers peut prendre

    speech_values = model.generate(**inputs, do_sample=True)

    Audio(speech_values.cpu().numpy().squeeze(), rate=22050)

    scipy.io.wavfile.write("bark_out.wav", rate=22050, data=speech_values.cpu().numpy().squeeze())
    """

    tts = TTS("tts_models/fr/css10/vits")
    tts.tts_with_vc_to_file(
        "Bonjour, le projet est fini !", speaker_wav=target_audio_file, file_path="tts_tts_output.wav"
    )


if __name__ == "__main__":
    main()
