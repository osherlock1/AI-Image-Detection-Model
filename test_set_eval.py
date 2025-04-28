# pip install torch torchvision torchaudio transformers datasets peft accelerate pillow pandas scikit-learn matplotlib seaborn
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
from peft import PeftModel 
import accelerate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import UnidentifiedImageError
import sys 


# --- Configuration ---
#DATASET_ROOT_PATH = "/home/omalley_sherlock_uri_edu/ondemand/datasets/artifact" # Path to the main dataset directory on Unity
DATASET_ROOT_PATH = "/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/Datasets/artifact" # PATH FOR LOCAL
TRAINING_OUTPUT_DIR = "/Users/omalleysherlock/Documents/URI_Spring_2025/ELE392_AI_ML/Assignemnets/Final_Project/clip_model/results_unity/final_training_run" # Local Output Dir
CHECKPOINT_NAME = "checkpoint-156045" 
ADAPTER_PATH = os.path.join(TRAINING_OUTPUT_DIR, CHECKPOINT_NAME)
TEST_SET_FILELIST_PATH = os.path.join(TRAINING_OUTPUT_DIR, "test_set_files.txt") # txt file with the test files not used in training

# deinfe modee
MODEL_ID = "openai/clip-vit-base-patch32"
#Output dir
EVAL_OUTPUT_DIR = os.path.join(TRAINING_OUTPUT_DIR, f"test_set_eval_{CHECKPOINT_NAME}")


EVAL_BATCH_SIZE = 32 



# Basic path checks
if not os.path.isdir(DATASET_ROOT_PATH):
    raise FileNotFoundError(f"ERROR: Dataset root path '{DATASET_ROOT_PATH}' not found.")
if not os.path.isdir(ADAPTER_PATH):
     raise FileNotFoundError(f"ERROR: Checkpoint adapter path '{ADAPTER_PATH}' not found. Check the checkpoint name and training output directory.")
if not os.path.exists(TEST_SET_FILELIST_PATH):

     raise FileNotFoundError(f"ERROR: Test set file list '{TEST_SET_FILELIST_PATH}' not found. Make sure the training script saved it.")
else:
     print(f"Paths verified.")

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

#
all_test_labels = []
all_test_predictions = []

def compute_metrics_for_test_set(eval_pred):
    logits, labels = eval_pred

    #Convert to np array
    if isinstance(logits, torch.Tensor): logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

    #Get predictions
    predictions = 1 / (1 + np.exp(-logits)) 
    binary_predictions = (predictions >= 0.5).astype(np.int32).flatten()
    labels = labels.astype(np.int32).flatten()


    all_test_labels.extend(labels.tolist())
    all_test_predictions.extend(binary_predictions.tolist())

    # Calculate metrics 
    acc = accuracy_score(labels, binary_predictions)
    return {'accuracy': acc}

# --- Load test set from txt file ---
def load_test_file_list(filepath):
    test_files = []
    print(f"Loading test set file list from: {filepath}") 
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    path, label_str = parts
                    try:
                        label = int(label_str)
                        test_files.append((path, label))
                    except ValueError:
                        print(f"WARNING: Skipping line with invalid label '{label_str}' in {filepath}") 
                else:
                    print(f"WARNING: Skipping malformed line in {filepath}: {line.strip()}") 
    except Exception as e:
        print(f"ERROR reading test set file list {filepath}: {e}", file=sys.stderr) 
        raise
    print(f"Loaded {len(test_files)} file paths from test set list.") 
    if not test_files:
        raise ValueError("Test set file list is empty or could not be parsed.")
    return test_files



# --- Main ---
if __name__ == '__main__':



    #---------- LOAD IN THE MODEL -----------
    print(f"loading processor path: {ADAPTER_PATH}") 
    processor = CLIPProcessor.from_pretrained(ADAPTER_PATH)
    print(f"loaded processor") 

    print(f"Loading model: {MODEL_ID}") 
    base_model = CLIPBinaryClassifier(MODEL_ID)

    #Load in the weights
    print(f"loading weights {ADAPTER_PATH}") 
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("loaded weights") 

    # ----- test --------
    model.eval()
    test_files_list = load_test_file_list(TEST_SET_FILELIST_PATH)
    test_dataset = ImageDetectionDataset(DATASET_ROOT_PATH, test_files_list, processor)

    #-----set trainer to eval -------
    print("Setting trainer")
    eval_args = TrainingArguments(
        output_dir=EVAL_OUTPUT_DIR, 
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        do_train=False,
        do_eval=True, 
        report_to="none", 
        remove_unused_columns=False,
        label_names=["labels"],
        fp16=False, 
    )

    trainer = Trainer(
        model=model, 
        args=eval_args,
        eval_dataset=test_dataset, 
        compute_metrics=compute_metrics_for_test_set, 
        data_collator=ImageDetectionDataset.collate_fn, 
    )

    #----- run evaluation -------
    print(f"Running test set eval on the test set ({len(test_dataset)} samples") 
    eval_output = trainer.evaluate()
    print(f"output: {eval_output}") 


   #------- display results -----------
    print(f"\n--- Final Results from test set ---")
    print(f"Calculating final metrics based on {len(all_test_labels)} test samples.") 
    final_accuracy = accuracy_score(all_test_labels, all_test_predictions)
    final_precision, final_recall, final_f1, _ = precision_recall_fscore_support(all_test_labels, all_test_predictions, average='binary', zero_division=0)
    print(f"Test Accuracy:  {final_accuracy:.4f}") 
    print(f"Test F1-Score:  {final_f1:.4f}") 
    print(f"Test Precision: {final_precision:.4f}") 
    print(f"Test Recall:    {final_recall:.4f}") 
    print("\nClassification Report (Test Set):")
    target_names = ['Real (0)', 'Fake (1)']
    print(classification_report(all_test_labels, all_test_predictions, target_names=target_names, zero_division=0))



    # -------- confusion matrix ----------------

    print("Generating matrix") 
    cm = confusion_matrix(all_test_labels, all_test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Real', 'Pred Fake'], yticklabels=['True Real', 'True Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Test Set)')
    # Save 
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    cm_path = os.path.join(EVAL_OUTPUT_DIR, f"test_set_confusion_matrix_{CHECKPOINT_NAME}.png")
    plt.savefig(cm_path)
    print(f"Test set confusion matrix saved to: {cm_path}") 

    print("--- Test Set eval done ---") 
