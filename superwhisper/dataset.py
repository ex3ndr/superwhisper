import gzip
import json
import math
import random
import torch
import torchaudio
from .audio import load_mono_audio

def load_libriheavy_sampler(index):

    # Load ids
    rows = []
    with gzip.open(index, "r") as f:
        for line in f:
            cut = json.loads(line)
            rows.append(cut)

    def sample():
        while True:

            # Pick ID
            record = random.choice(rows)
            recording_id = record["supervisions"][0]["recording_id"]
            text = record["supervisions"][0]["custom"]["texts"][0]
            start = record["start"]
            duration = record["duration"]

            # Try load
            try:

                # Resolve folder
                if recording_id.startswith("small/"):
                    audio_file = "./external_datasets/librilight/" + recording_id[len("small/"):] + ".flac"
                elif recording_id.startswith("medium/"):
                    audio_file = "./external_datasets/librilight-medium/" + recording_id[len("medium/"):] + ".flac"
                elif recording_id.startswith("large/"):
                    audio_file = "./external_datasets/librilight-large/" + recording_id[len("large/"):] + ".flac"
                else:
                    raise Exception("Invalid id")

                batch = {}

                # Load audio
                audio =  load_mono_audio(audio_file, 16000)
                
                # Trim
                start_frame = math.floor(start * 16000)
                end_frame = math.floor((start + duration) * 16000)
                audio = audio[start_frame:end_frame]
                
                # Features
                batch["audio"] = audio

                # Load text
                batch["text"] = text

                # Return
                return batch
            except:
                print("Invalid file: " + recording_id)
                raise
    
    return sample

def load_hifitts_sampler(index):

    # Load ids
    rows = []
    with open(index, "r") as f:
        for line in f:
            cut = json.loads(line)
            rows.append(cut)

    def sample():
        while True:

            # Pick ID
            record = random.choice(rows)
            audio_filepath = record["audio_filepath"]
            text = record["text_normalized"]

            # Try load
            try:
                batch = {}

                # Load audio
                audio =  load_mono_audio("./external_datasets/hifi-tts/" + audio_filepath, 16000)
                
                # Features
                batch["audio"] = audio

                # Load text
                batch["text"] = text

                # Return
                return batch
            except:
                print("Invalid file: " + audio_filepath)
                raise
    
    return sample

def create_whisper_sampler(sampler, processor):
    def sample():
        batch = {}

        # Sample
        source = sampler()
    
        # Audio
        batch["input_features"] = processor.feature_extractor(source["audio"], sampling_rate=16000).input_features[0]
    
        # Text
        batch["labels"] = processor.tokenizer(source["text"]).input_ids
    
        # Return
        return batch
    
    return sample

def create_async_dataset(sampler):
    class AsyncDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = AsyncDataset(sampler)
    return dataset
