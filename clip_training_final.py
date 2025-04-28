## Imports ------------

##General
import os
import pandas as pd
import numpy as np
import logging
import time
import random
import sys
##Torch
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import(CLIPProcessor, CLIPModel, TrainingArguments, Trainer)

#Image Dataset Handling
from PIL import Image

#Model
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import accelerate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import UnidentifiedImageError
#Weights and Bias
import wandb


# --- Configuration ---
# Using Local paths as per last working version before simplification
#DATASET_ROOT_PATH = "/home/omalley_sherlock_uri_edu/ondemand/datasets/artifact"
DATASET_ROOT_PATH = "/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/Datasets/artifact" # PATH FOR LOCAL
MODEL_ID = "openai/clip-vit-base-patch32"
#OUTPUT_DIR = '/home/omalley_sherlock_uri_edu/ondemand/AI_Detection_Project/results/train_10_percent_simplified_v2'
OUTPUT_DIR = "/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/clip_model/results/small_train_run_wandb" # Local Output Dir
WANDB_PROJECT_NAME = "clip_artifact_detection_train"

# --- Parameters ---
NUM_SAMPLES_TOTAL = 100
VALIDATION_SPLIT_SIZE = 0.1
NUM_TRAIN_EPOCHS = 1.0 
EVAL_STEPS = 500
LOGGING_STEPS = 50
NUM_DATALOADER_WORKERS = 1 # Number of requested CPUs
RANDOM_SEED = 42


# Hyperparameters tuned by optuna
PER_DEVICE_BATCH_SIZE = 16
LEARNING_RATE = 1.0e-4
lora_r = 4
lora_alpha = 16

# Basic path check
if not os.path.isdir(DATASET_ROOT_PATH):
    print("Path error")
    exit()


# --- Data Loading Function ( ---
def load_all_metadata_binary(data_dir, progress_interval=100000):
    
    all_metadata_records = []
    processed_record_count = 0
    last_log_count = 0
    start_time = time.time()
    print(f"Scanning image files and assigning labels (0=Real, 1=Fake) in: {data_dir}") 

    for subdir, dirs, files in os.walk(data_dir):
        if 'metadata.csv' in files:
            metadata_path = os.path.join(subdir, 'metadata.csv')
            print(f"Processing metadata: {os.path.relpath(metadata_path, data_dir)}") 
            try:
                df = pd.read_csv(metadata_path)
                # column check
                if 'image_path' not in df.columns or 'target' not in df.columns:
                    print(f"WARNING: Skipping {metadata_path}: missing 'image_path' or 'target' column.") 
                    continue

                relative_subdir = os.path.relpath(subdir, data_dir)

                for index, row in df.iterrows():
                    csv_image_path = row['image_path']
                    if relative_subdir == '.': full_relative_path = csv_image_path
                    else: full_relative_path = os.path.join(relative_subdir, csv_image_path)
                    full_relative_path = os.path.normpath(full_relative_path)

                    try:
                        original_target = int(row['target'])
                        binary_label = 0 if original_target == 0 else 1
                        all_metadata_records.append((full_relative_path, binary_label))
                        processed_record_count += 1
                        if processed_record_count >= last_log_count + progress_interval:
                            elapsed_time = time.time() - start_time
                            print(f"  ... processed {processed_record_count // 1000}k records ({elapsed_time:.1f} seconds elapsed)") 
                            last_log_count = processed_record_count
                    except (ValueError, TypeError):
                        print(f"WARNING: Skipping row with non-integer label '{row['target']}' in {metadata_path}") 
                        continue 

            except Exception as e:
                print(f"ERROR") 

    end_time = time.time()
    print(f"Finished scanning. Found {len(all_metadata_records)} images") 
    print(f"Total time: {end_time - start_time:.2f} seconds") 
    if not all_metadata_records:
         raise ValueError("No valid image found")
    return all_metadata_records

# --- Custom Dataset  ---
class ImageDetectionDataset(Dataset):
    def __init__(self, data_dir, image_files_list, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.image_files = image_files_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        full_relative_path, label = self.image_files[idx]
        img_path = os.path.join(self.data_dir, full_relative_path)

        
        if not os.path.exists(img_path):
            #Prevent crashing during dataloading
            return None 

        image = Image.open(img_path).convert("RGB")
        processed = self.processor(images=image, return_tensors="pt", padding=False, truncation=True)
        pixel_values = processed['pixel_values'][0]
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.float)}


    @staticmethod
    def collate_fn(batch):

        batch_filtered = list(filter(lambda x: x is not None, batch))

        # If the entire batch failed return an empty structure
        if not batch_filtered:
             print("WARNING: All items in the batch were None or failed before collation.") 
             return {'pixel_values': torch.tensor([]), 'labels': torch.tensor([])}

        # Collate the filtered batch
        try:
            return torch.utils.data.dataloader.default_collate(batch_filtered)
        except Exception as e:
             print(f"ERROR during collation: {e}", file=sys.stderr) 
             return {'pixel_values': torch.tensor([]), 'labels': torch.tensor([])}



# --- Model Definition ---
class CLIPBinaryClassifier(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_id)
        self.classifier_head = torch.nn.Linear(self.clip.config.vision_config.hidden_size, 1)

    def forward(self, pixel_values, labels=None, **kwargs):
        if pixel_values.dim() == 3: pixel_values = pixel_values.unsqueeze(0)

        if pixel_values.nelement() == 0:
             if labels is not None and labels.nelement() == 0: return {"loss": torch.tensor(0.0, device=self.classifier_head.weight.device, requires_grad=True), "logits": torch.tensor([])}
             else: return {"logits": torch.tensor([])}

        image_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output
        logits = self.classifier_head(image_features)
        loss = None

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(-1), labels.float())

        if loss is not None: return {"loss": loss, "logits": logits}
        else: return {"logits": logits}

# --- Metrics Calculation ---
def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy()
    binary_predictions = (predictions >= 0.5).astype(np.int32).flatten()
    labels = labels.astype(np.int32).flatten()

    if len(np.unique(labels)) < 2 or len(np.unique(binary_predictions)) < 2:
         acc = accuracy_score(labels, binary_predictions)
         return {'accuracy': acc, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    else:
         precision, recall, f1, _ = precision_recall_fscore_support(labels, binary_predictions, average='binary', zero_division=0)
         acc = accuracy_score(labels, binary_predictions)
         return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}



# --- Main Execution Block ---
if __name__ == '__main__':





    #Load the Data
    print("--- Starting Data Preparation ---") 
    all_files = load_all_metadata_binary(DATASET_ROOT_PATH)
    if not all_files: raise ValueError("Failed to load")

    if NUM_SAMPLES_TOTAL and NUM_SAMPLES_TOTAL < len(all_files):
         import random
         random.seed(RANDOM_SEED)
         random.shuffle(all_files)
         all_files = all_files[:NUM_SAMPLES_TOTAL]
         print(f"Using subset of {len(all_files)} images") 
    else:
         print(f"Using all {len(all_files)} samples.") 

    


    #--------- split the data ----------------


    print("Splitting data into Train and Validation sets...")
    train_files, eval_files = train_test_split(all_files, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED)
    print(f"Split data: {len(train_files)} train samples, {len(eval_files)} eval samples.") 




    # ----------Get number of steps based on train size ---------------

    STEPS_PER_EPOCH= max(1, len(train_files) // PER_DEVICE_BATCH_SIZE)
    
    
    EVAL_STEPS = max(500, STEPS_PER_EPOCH // 4) 
    LOGGING_STEPS = max(50, STEPS_PER_EPOCH // 20) 
    print(f"Steps per Epoch: {STEPS_PER_EPOCH}") 
    print(f"Eval/Save Steps: {EVAL_STEPS}") 
    print(f"Logging Steps: {LOGGING_STEPS}") 


    # -------- Create Processor, Model, PEFT ------------

    print(f"processor for model: {MODEL_ID}") 
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    print("Initializing Model...")
    model = CLIPBinaryClassifier(MODEL_ID)

    print("Applying PEFT/LoRA...")
    lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", modules_to_save=["classifier_head"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 


    #------- instantiate datasets ----------------
    print("Creating Train and Validation Dataset objects...") 
    train_dataset = ImageDetectionDataset(DATASET_ROOT_PATH, train_files, processor)
    eval_dataset = ImageDetectionDataset(DATASET_ROOT_PATH, eval_files, processor)
    print("Datasets created") 



    # -------- set up the trainer for training ---------
    print("Setting up Trainer args") 
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS, 
        eval_strategy="steps",
        eval_steps=EVAL_STEPS, 
        save_strategy="steps",
        save_steps=EVAL_STEPS, 
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"clip_train_simple_lr{LEARNING_RATE:.1e}_bs{PER_DEVICE_BATCH_SIZE}_r{lora_r}_a{lora_alpha}_ep{NUM_TRAIN_EPOCHS}",
        label_names=["labels"],
        fp16=False,
        dataloader_num_workers=NUM_DATALOADER_WORKERS,
    )


    # Set project name for wandb
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME


    # Start trainer
    print("Starting Trainer...") 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=ImageDetectionDataset.collate_fn, 
    )

    #------- Train -------
    print(f"Starting Training for {NUM_TRAIN_EPOCHS} epochs...") 
    try:
        trainer.train()
        print("--- Training Completed ---")

        final_adapter_path = os.path.join(OUTPUT_DIR, "final_best_adapter")
        model.save_pretrained(final_adapter_path)
        processor.save_pretrained(final_adapter_path)
        print(f"Best model saved to {final_adapter_path}") 

        print("Final metrics from best checkpoint:") 
        if trainer.state.best_metric is not None:
             print(f"  Best {training_args.metric_for_best_model}: {trainer.state.best_metric}")
             print(f"  Best Model Checkpoint: {trainer.state.best_model_checkpoint}")
        else:
             print("Could not determine best metric/checkpoint from trainer state.")

    except Exception as e:
        print(f"--- Error during training: {e} ---", file=sys.stderr)

    # Finish wandb run
    if wandb.run is not None:
        print("Finishing wandb run...") 
        wandb.finish()

    print("--- Script End ---") 
