from .effects import default_noisy_pipeline
import torchaudio
import random

def create_distorter(rirs, bg):

    # Parameters
    codec_probability = 0.3
    codecs = [
        {'format': "wav", 'encoder': "pcm_mulaw"},
        {'format': "g722"},
        {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=8_000)},
        {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=64_000)}
    ]

    # Pipeline
    pipeline = default_noisy_pipeline(rirs = rirs, bg = bg)

    # Implementation
    def effector(audio):

        # Apply pipeline
        audio = pipeline.apply(audio, 16000)

        # Apply codec effect
        codec = None
        if random.random() < codec_probability:
            codec = random.choice(codecs)
            args = {}
            args.update(codec)
            effector = torchaudio.io.AudioEffector(**args)
            audio = effector.apply(audio.unsqueeze(0).T, 16000).T[0]

        return audio

    # Return 
    return effector