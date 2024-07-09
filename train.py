#
# Ignore warnings
#

import warnings
warnings.filterwarnings("ignore")

#
# Imports
#

import evaluate
from transformers import WhisperFeatureExtractor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from superwhisper.dataset import create_whisper_sampler, load_libriheavy_sampler, create_async_dataset, load_hifitts_sampler, create_mixing_sampler, create_static_dataset

#
# Parameters
#

train_model_name = "openai/whisper-small"
train_model_language = "en"
train_run_name = "train-small"

#
# Load base model
#

print("Loading base model...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(train_model_name)
tokenizer = WhisperTokenizer.from_pretrained(train_model_name, language=train_model_language, task="transcribe")
processor = WhisperProcessor.from_pretrained(train_model_name, language=train_model_language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(train_model_name)

#
# Load metrics
#

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#
# Load dataset
#

print("Loading dataset...")
# clean_sampler = load_libriheavy_sampler("./external_datasets/libriheavy/libriheavy_cuts_small.jsonl.gz")
clean_sampler = load_hifitts_sampler("./external_datasets/hifi-tts/9017_manifest_clean_train.json")
mixing_sampler = create_mixing_sampler(clean_sampler)
dataset = create_async_dataset(create_whisper_sampler(mixing_sampler, processor))
eval_dataset = create_static_dataset(create_whisper_sampler(clean_sampler, processor), 32)

#
# Load data collator
#

class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, *, processor):
        self.processor = processor

    def __call__(self, features):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


#
# Load training arguments
#

training_args = Seq2SeqTrainingArguments(
    output_dir="./output/",
    run_name=train_run_name,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=4
)

#
# Load trainer
#

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

#
# Start trainer
#

trainer.train()