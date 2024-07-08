import gzip
import json
import math
import random
import torch
import torchaudio

def load_clean_sampler(index, processor):

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
                audio, sr =  torchaudio.load(audio_file)
                if audio.shape[0] == 2:
                    audio = audio.mean(dim=0, keepdim=True)
                audio.squeeze_(0)
                print(audio_file, audio.shape)
                
                # Trim
                start_frame = math.floor(start * sr)
                end_frame = math.floor((start + duration) * sr)
                audio = audio[start_frame:end_frame]
                
                # Features
                batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
                batch["input_length"] = len(audio) / sr

                # Load text
                batch["labels"] = processor.tokenizer(text).input_ids

                # Return
                return batch
            except Error:
                print("Invalid file: " + recording_id)
                raise
    
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

def create_async_loader(sampler, num_workers = 1):

    # Dataset
    class AsyncDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = AsyncDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader