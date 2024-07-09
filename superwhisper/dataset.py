import gzip
import json
import math
import random
import torch
import torchaudio
from .audio import load_mono_audio
from .distorter import create_distorter
from pathlib import Path

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
            speaker = record["supervisions"][0]["speaker"]

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

                # Speaker
                batch["speaker"] = speaker

                # Return
                return batch
            except:
                print("Invalid file: " + recording_id)
                raise
    
    return sample

def load_hifitts_sampler(indexes):

    # Load ids
    rows = []
    for index in indexes:
        with open(index, "r") as f:
            for line in f:
                cut = json.loads(line)
                rows.append((cut, index))

    def sample():
        while True:

            # Pick ID
            (record, speaker) = random.choice(rows)
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

                # Speaker
                batch["speaker"] = speaker

                # Return
                return batch
            except:
                print("Invalid file: " + audio_filepath)
                raise
    
    return sample

def load_distorted_sampler(sampler):

    # Load RIR files
    rir_files = []
    with open('./external_datasets/rir-1/files.txt', 'r') as file:
        for line in file:
            rir_files.append("./external_datasets/rir-1/" + line.strip())

    # Load BG files
    bg_files = []
    for p in Path("./external_datasets/dns-noise").rglob("*.wav"):
        bg_files.append(str(p))

    # Create distorter
    distorter = create_distorter(rir_files, bg_files)

    def sample():
        batch = sampler()
        batch["audio"] = distorter(batch["audio"])
        return batch

    return sample

def create_mixing_sampler(sampler):
    def sample():

        # Load base samples
        sample_0 = sampler()
        sample_1 = sampler()

        # Combine
        batch = {}

        # Audio
        audio_0 = sample_0["audio"]
        audio_1 = sample_1["audio"]
        batch["audio"] = torch.cat([audio_0, torch.zeros(random.randint(100, 300)), audio_1], dim=0) # Add pause?

        # Text
        text_0 = sample_0["text"]
        text_1 = sample_1["text"]

        # Combine
        if sample_0["speaker"] == sample_1["speaker"]:
            batch["text"] = text_0 + " " + text_1
        else:
            batch["text"] = text_0 + " <|startoflm|> " + text_1

        return batch
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

def create_static_dataset(sampler, count):

    # Sample count times
    samples = []
    random.seed(42)
    for _ in range(count):
        samples.append(sampler())
    random.seed(None)

    # Create dataset
    class StaticDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = StaticDataset(samples)
    return dataset