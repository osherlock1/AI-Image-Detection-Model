# pip install torch torchvision torchaudio transformers datasets peft accelerate pillow pandas scikit-learn wandb optuna
# --- Imports ---
import torch
import os
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPProcessor,
    CLIPModel,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import accelerate
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import UnidentifiedImageError
import time
import optuna 
import wandb 
from multiprocessing import freeze_support


# --- Configuration ---
DATASET_ROOT_PATH = "/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/Datasets/artifact" # Path on Local Machine
MODEL_ID = "openai/clip-vit-base-patch32"
BASE_OUTPUT_DIR = '/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/clip_model/results/optuna_hpo_local' # Base dir for HPO results on Local Machine
WANDB_PROJECT_NAME = "optuna_test" 

# --- Parameters for the HPO run ---
VALIDATION_SPLIT_SIZE = 0.1 
NUM_SAMPLES_TOTAL_FOR_SPLIT = 500 

#Fixed training parameters for optuna study
NUM_TRAIN_EPOCHS = 1.0
PER_DEVICE_EVAL_BATCH_SIZE_FACTOR = 2
NUM_DATALOADER_WORKERS = 1 # number of cpus requeseted from unity
RANDOM_SEED = 42

# Optuna Study Configuration
N_TRIALS = 1000 
METRIC_TO_OPTIMIZE = "eval_accuracy"

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


# --- Optuna Integration ---

#Define model class functino for variable model structure
def model_init(trial=None):
    lora_r = trial.params.get("lora_r", 8) if trial else 8
    lora_alpha = trial.params.get("lora_alpha", 16) if trial else 16
    print(f"Initializing model for trial {trial.number if trial else 'N/A'} with r={lora_r}, alpha={lora_alpha}")
    model = CLIPBinaryClassifier(MODEL_ID)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", modules_to_save=["classifier_head"],
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model



#Define the object fuction for the trials 
def objective(trial: optuna.trial.Trial, train_dataset, eval_dataset, processor) -> float:
    print(f"\n--- Objective Function for Trial {trial.number} ---")
    trial_num = trial.number

    #Suggest parameters for tuning
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    lora_r = trial.suggest_categorical("lora_r", [4, 8, 16])
    lora_alpha = trial.suggest_categorical("lora_alpha", [4, 8, 16, 32])


    run_output_dir = os.path.join(BASE_OUTPUT_DIR, f"trial_{trial_num}")

    run_name = f"hpo_local_trial_{trial_num}_lr{learning_rate:.1e}" \
               f"_bs{per_device_train_batch_size}" \
               f"_r{trial.params['lora_r']}_a{trial.params['lora_alpha']}"

    # Calculate steps based on batch size
    num_processes = accelerate.Accelerator().num_processes
    total_train_batch_size = per_device_train_batch_size * num_processes
    if len(train_dataset) == 0 or total_train_batch_size == 0: steps_per_epoch = 1
    else: steps_per_epoch = len(train_dataset) // total_train_batch_size
    eval_steps = max(10, steps_per_epoch // 2)
    logging_steps = max(5, steps_per_epoch // 10) 


    print(f"Trial {trial_num}: lr={learning_rate:.2e}, batch_size={per_device_train_batch_size}, r={lora_r}, alpha={lora_alpha}, eval_steps={eval_steps}")

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        run_name=run_name,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * PER_DEVICE_EVAL_BATCH_SIZE_FACTOR,
        learning_rate=learning_rate,
        logging_dir=f'{run_output_dir}/logs',
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_TO_OPTIMIZE,
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="wandb",
        label_names=["labels"],
        fp16=False,
        dataloader_num_workers=NUM_DATALOADER_WORKERS,
        disable_tqdm=False,
        use_mps_device=torch.backends.mps.is_available()
    )

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME

    trainer = Trainer(
        model_init=model_init, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=ImageDetectionDataset.collate_fn, 
    )

    metric_value = -1.0

    print(f"--- Starting Training for Trial {trial_num} ---") 
    trainer.train()
    metrics = trainer.evaluate() 
    metric_value = metrics.get(METRIC_TO_OPTIMIZE)
    print(f"--- Trial {trial_num} Completed --- Metric ({METRIC_TO_OPTIMIZE}): {metric_value}") 


    # finish the wandb run for this trial
    if wandb.run is not None:
        wandb.finish()
    print(f"--- Exiting Objective Function for Trial {trial.number} ---") # Use print
    return float(metric_value) if metric_value is not None else -1.0


# --- MAIN ---
if __name__ == '__main__':


    print("--- Starting Data Preparation ---") 
    all_files = load_all_metadata_binary(DATASET_ROOT_PATH)
    if not all_files: raise ValueError("Failed to load metadata.")
    print("\n" + "="*10 + " FINISHED METADATA SCAN " + "="*10 + "\n")

    if NUM_SAMPLES_TOTAL_FOR_SPLIT and NUM_SAMPLES_TOTAL_FOR_SPLIT < len(all_files):
         import random
         random.seed(RANDOM_SEED)
         random.shuffle(all_files)
         all_files = all_files[:NUM_SAMPLES_TOTAL_FOR_SPLIT]
         print(f"Using subset of {len(all_files)} samples for HPO study.")
    else:
         print(f"Using all {len(all_files)} samples for HPO study.") 

    train_files, eval_files = train_test_split(all_files, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED)
    print(f"Split data for HPO: {len(train_files)} train samples, {len(eval_files)} eval samples.") 

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    train_dataset = ImageDetectionDataset(DATASET_ROOT_PATH, train_files, processor)
    eval_dataset = ImageDetectionDataset(DATASET_ROOT_PATH, eval_files, processor)
    print("--- Data Preparation Complete ---") 
    print("\n" + "="*10 + " DATASETS CREATED - ENTERING OPTUNA PHASE " + "="*10 + "\n") 


    # Create and run the Optuna study
    print(f"--- Starting Optuna Hyperparameter Optimization ({N_TRIALS} trials) ---") 
    study = optuna.create_study(direction="maximize", study_name=f"clip_artifact_hpo_local_{int(time.time())}")
    
    objective_func = lambda trial: objective(trial, train_dataset=train_dataset, eval_dataset=eval_dataset, processor=processor)
    study.optimize(objective_func, n_trials=N_TRIALS)

    # --- Print Results ---
    print("--- Hyperparameter Optimization Finished ---") 
    print(f"Number of finished trials: {len(study.trials)}") 
   
    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (Max {METRIC_TO_OPTIMIZE}): {best_trial.value}") 
    print("  Params: ") 
    for key, value in best_trial.params.items(): print(f"    {key}: {value}") 



    print("--- Script Complete ---") 

