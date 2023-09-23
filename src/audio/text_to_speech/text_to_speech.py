class TextToSpeech:
    def __init__(self):
        pass


"""
# Chargement du fichier audio du locuteur cible
target_audio_file = "target_audio.wav"
target_waveform, _ = torchaudio.load(target_audio_file)

# Conversion de l'embedding du locuteur source en audio du locuteur cible
target_length = len(target_waveform.squeeze())
# là c'est les embeddings calculés avant
source_embedding = source_embedding[:, :target_length, :]

# Génération de l'audio du locuteur cible à partir de l'embedding
with torch.no_grad():
    generated_waveform = model.generate(input_values=source_embedding)

# Sauvegarde de l'audio généré
output_audio_file = "output_audio.wav"
torchaudio.save(output_audio_file, generated_waveform.squeeze().numpy(), sample_rate=16000)
"""
