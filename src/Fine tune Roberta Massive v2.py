from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
from torch.utils.data import DataLoader
import math

def encode_label_function(examples) :
    return {'labels':[category_to_label[c] for c in examples["category"]]}

def tokenize_function(examples):
    return tokenizer(examples["utt"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'f1_macro': f1_score(labels, predictions, average='macro'),
            'f1_micro': f1_score(labels, predictions, average='micro')}

DATASET_NAME = "massive"
DATASET = f"benayas/{DATASET_NAME}"
TYPE = "baseline" # "artificial" or "baseline"
DATASET_PREFIX = "baseline" # not used in baseline


# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

for TYPE in ['baseline']:
  DATASET_PREFIX = TYPE

  for VERSION in [3, 4, 5]:
    for PCT in [5, 10, 20]:
      DATASET_ARTIFICIAL = f"benayas/{DATASET_NAME}_{DATASET_PREFIX}_{PCT}pct_v{VERSION}"

      # download dataset
      dataset = load_dataset(DATASET)

      if TYPE == "baseline":
        df = pd.DataFrame(dataset["train"])

        # Extract a sample per class
        df_sample = []

        for name, g in df.groupby('category'):
          n = math.ceil(len(g)*PCT/100.0)
          sample = g.sample(n, random_state=VERSION)
          df_sample.append(sample)

        df_sample = pd.concat(df_sample)
        dataset_artificial = Dataset.from_pandas(df_sample)
        print(f"Dataset size at {PCT} pct {len(df_sample)}")

      else:
        dataset_artificial = load_dataset(DATASET_ARTIFICIAL)['train']

      labelencoder = LabelEncoder().fit(dataset['train']['category'])
      category_to_label = {c: i for i, c in enumerate(labelencoder.classes_)}
      mapping = {v:k for k,v in category_to_label.items()}

      tokenized_datasets = dataset.map(tokenize_function2, batched=True)
      tokenized_datasets = tokenized_datasets.map(encode_label_function, batched=True)

      tokenized_datasets_artificial = dataset_artificial.map(tokenize_function, batched=True)
      tokenized_datasets_artificial = tokenized_datasets_artificial.map(encode_label_function, batched=True)

      # Remove columns
      tokenized_datasets = tokenized_datasets.remove_columns(["utt","category"])
      tokenized_datasets.set_format("torch")

      tokenized_datasets_artificial = tokenized_datasets_artificial.remove_columns(["utt","category"])
      tokenized_datasets_artificial.set_format("torch")

      # split datasets
      train_dataset = tokenized_datasets_artificial.shuffle(seed=42)
      eval_dataset = tokenized_datasets["test"]

      # Model
      model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=len(set(mapping)))

      # Training Args
      training_args = TrainingArguments(
                                    #run_name=f"roberta-{DATASET_NAME}-{PCT}pct-v{VERSION}",
                                    output_dir="test_trainer",
                                    learning_rate=0.00005,
                                    num_train_epochs=15,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    load_best_model_at_end=True,
                                    save_strategy="epoch",
                                    evaluation_strategy="epoch",
                                    #metric_for_best_model="eval_loss",
                                    metric_for_best_model="f1_macro",
                                    greater_is_better=True,
                                    logging_strategy="epoch",
                                    )

      # Trainer
      trainer = Trainer(
          model=model,
          args=training_args,
          train_dataset=train_dataset,
          eval_dataset=eval_dataset,
          compute_metrics=compute_metrics,
      )
      trainer.train()

      # Upload model to HuggingFace
      new_model = f"roberta-{TYPE}-finetuned-atis_{str(PCT)}pct_v{VERSION}"
      model.push_to_hub(new_model, use_temp_dir=True)
      tokenizer.push_to_hub(new_model, use_temp_dir=True)

      # Evaluate
      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
      eval_dataloader = DataLoader(eval_dataset, batch_size=16)

      all_outputs = []
      all_labels = []
      all_logits = []

      model.eval()
      model.to(device)
      for batch in tqdm(eval_dataloader):
          batch = {'input_ids': batch['input_ids'].to(device),
                  'attention_mask': batch['attention_mask'].to(device),
                  'labels': batch['labels'].to(device)}
          with torch.no_grad():
              outputs = model(**batch)

          logits = outputs.logits
          all_logits.append(logits.detach().cpu().numpy())
          predictions = torch.argmax(logits, dim=-1)
          all_outputs += predictions.detach().cpu().numpy().tolist()
          all_labels += batch["labels"].detach().cpu().numpy().tolist()

      all_labels = np.array(all_labels)
      all_outputs = np.array(all_outputs)
      all_logits = np.concatenate(all_logits, axis=0)

      f1_score_macro = f1_score(all_labels, all_outputs, average='macro')
      f1_score_micro = f1_score(all_labels, all_outputs, average='micro')
      print(f'f1_score_macro: {f1_score_macro}, f1_score_micro: {f1_score_micro}')

      # Generate CSV with full results
      df_outputs = pd.DataFrame({'text': np.array(dataset['test']['utt']),
                                'label': np.array([category_to_label[value] for value in np.array(dataset['test']['category'])]),
                                'label2': all_labels,
                                'roberta_prediction': all_outputs,
                                'intent':  np.array(dataset['test']['category'])
      })
      assert df_outputs['label'].equals(df_outputs['label2'])
      df_outputs = df_outputs[['text','intent','label', 'roberta_prediction']]
      df_outputs.to_csv(f'/content/drive/MyDrive/Estudios/PHD/LLM augmentation/results/roberta_{TYPE}_outputs_{DATASET_NAME}_{PCT}pct_v{VERSION}.csv', index=False)

