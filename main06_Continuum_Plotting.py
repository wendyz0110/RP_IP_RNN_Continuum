import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load results from practice schedule continuum experiment
#
# This folder contains:
#   (1) pre-training generalization metrics
#   (2) training/retention/generalization metrics by block size
#   (3) sequence-level retention and retention imbalance
#   (4) schedule indices for debugging
# ============================================================

lr = 0.02
results_folder = f"results_block_continuum_lr_{str(lr).replace('.', '_')}"

# -------------------------
# Pre-training results
# -------------------------
loss_test_pre_array = np.load(os.path.join(results_folder, "loss_test_pre_array.npy"))
loss_test_array_pre_array = np.load(os.path.join(results_folder, "loss_test_array_pre_array.npy"))

# -------------------------
# Block-size dictionaries
# need allow_pickle=True + .item() because these were saved as dictionaries
# -------------------------
loss_array_by_block = np.load(os.path.join(results_folder, "loss_array_by_block.npy"), allow_pickle=True).item()
loss_retention_by_block = np.load(os.path.join(results_folder, "loss_retention_by_block.npy"), allow_pickle=True).item()
loss_test_by_block = np.load(os.path.join(results_folder, "loss_test_by_block.npy"), allow_pickle=True).item()
loss_retention_array_by_block = np.load(os.path.join(results_folder, "loss_retention_array_by_block.npy"), allow_pickle=True).item()
loss_test_array_by_block = np.load(os.path.join(results_folder, "loss_test_array_by_block.npy"), allow_pickle=True).item()
seq_retention_means_by_block = np.load(os.path.join(results_folder, "seq_retention_means_by_block.npy"), allow_pickle=True).item()
retention_imbalance_by_block = np.load(os.path.join(results_folder, "retention_imbalance_by_block.npy"), allow_pickle=True).item()
schedule_idx_by_block = np.load(os.path.join(results_folder, "schedule_idx_by_block.npy"), allow_pickle=True).item()

print("Loaded continuum results from:", results_folder)

# basic settings

block_sizes = np.array([1000, 500, 200, 100, 50, 20, 10, 5, 1])
selected_blocks = [1000, 100, 10, 1] # for clear visualizations

num_runs = len(loss_retention_by_block[block_sizes[0]])
num_blocks = len(block_sizes)

# ============================================================
# Summary dataframe: one row per run per block size
# ============================================================

summary_rows = [] # store one dictionary per row

for block_size in block_sizes:
    for run_idx in range(num_runs):
        summary_rows.append({ # one row added, fpr a specific run, and specific block size
            "Run": run_idx,
            "BlockSize": block_size,
            "ScheduleLabel": f"Block-{block_size}",
            "RetentionLoss": loss_retention_by_block[block_size][run_idx],
            "GeneralizationLoss": loss_test_by_block[block_size][run_idx],
            "RetentionImbalance": retention_imbalance_by_block[block_size][run_idx]
        })

df_summary = pd.DataFrame(summary_rows) # convert list to df

# make sure block sizes are in descending order
df_summary["ScheduleLabel"] = pd.Categorical(
    df_summary["ScheduleLabel"],
    categories=[f"Block-{b}" for b in block_sizes],
    ordered=True
)

print(df_summary.head())

# ============================================================
# Sequence-level retention dataframe
# One row per run × block size × trained sequence
# ============================================================

seq_rows = [] 

for block_size in block_sizes:
    seq_means = seq_retention_means_by_block[block_size]  # shape: (num_runs, 3)

    for run_idx in range(num_runs):
        for seq_idx in range(seq_means.shape[1]):
            seq_rows.append({
                "Run": run_idx,
                "BlockSize": block_size,
                "ScheduleLabel": f"Block-{block_size}",
                "Sequence": f"Seq-{seq_idx + 1}",
                "SeqRetentionLoss": seq_means[run_idx, seq_idx]
            })

df_seq_retention = pd.DataFrame(seq_rows)

df_seq_retention["ScheduleLabel"] = pd.Categorical(
    df_seq_retention["ScheduleLabel"],
    categories=[f"Block-{b}" for b in block_sizes],
    ordered=True
)

df_seq_retention["Sequence"] = pd.Categorical(
    df_seq_retention["Sequence"],
    categories=["Seq-1", "Seq-2", "Seq-3"],
    ordered=True
)

print(df_seq_retention.head())

# ============================================================
# Training loss curve dataframe
# One row per run × block size × epoch
# Use selected blocks only to avoid visual clutter
# ============================================================

training_rows = []

for block_size in selected_blocks:
    loss_array = loss_array_by_block[block_size]  # shape: (num_runs, num_epochs)

    for run_idx in range(loss_array.shape[0]):
        for epoch_idx in range(loss_array.shape[1]):
            training_rows.append({
                "Run": run_idx,
                "BlockSize": block_size,
                "ScheduleLabel": f"Block-{block_size}",
                "Epoch": epoch_idx + 1,
                "TrainingLoss": loss_array[run_idx, epoch_idx]
            })

df_training = pd.DataFrame(training_rows)

df_training["ScheduleLabel"] = pd.Categorical(
    df_training["ScheduleLabel"],
    categories=[f"Block-{b}" for b in selected_blocks],
    ordered=True
)

print(df_training.head())

# image folder
image_folder = "images_continuum"

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# ============================================================
# Plot 1: Training acquisition curves by selected block sizes
# ============================================================

plt.figure(figsize=(9, 5))

sns.lineplot(data=df_training,x="Epoch",y="TrainingLoss",hue="ScheduleLabel",errorbar=None)

plt.xlabel("Training Epoch")
plt.ylabel("Training Loss")
plt.title("Training Acquisition Curves Across Selected Block Sizes")
plt.legend(title="Practice Schedule")
plt.tight_layout()

plt.savefig(os.path.join(image_folder, "plot1_training_acquisition_curves.jpg"), dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# Plot 2: Average retention loss vs block size
# ============================================================

plt.figure(figsize=(8, 5))

sns.lineplot(data=df_summary,x="BlockSize",y="RetentionLoss",marker="o",errorbar="sd")

plt.xscale("log") # to make the x-axis log scaled, for equal spacing between block sizes
plt.gca().invert_xaxis()

plt.xlabel("Block Size")
plt.ylabel("Average Retention Loss")
plt.title("Average Retention Loss Across Practice Schedule Continuum")
plt.tight_layout()

plt.savefig(os.path.join(image_folder, "plot2_avg_retention_loss_by_block_size.jpg"), dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# Plot 3: Sequence-level retention loss vs block size
# ============================================================

plt.figure(figsize=(9, 5))

sns.lineplot(data=df_seq_retention,x="BlockSize",y="SeqRetentionLoss",hue="Sequence",marker="o",errorbar="sd")

plt.xscale("log")
plt.gca().invert_xaxis()

plt.xlabel("Block Size")
plt.ylabel("Sequence-Level Retention Loss")
plt.title("Retention Loss for Each Trained Sequence Across Block Sizes")
plt.legend(title="Trained Sequence")
plt.tight_layout()

plt.savefig(os.path.join(image_folder, "plot3_sequence_level_retention_by_block_size.jpg"), dpi=600, bbox_inches="tight")
plt.show()
plt.close()
    
# ============================================================
# Plot 4: Retention imbalance vs block size
# ============================================================

plt.figure(figsize=(8, 5))

sns.lineplot(data=df_summary,x="BlockSize",y="RetentionImbalance",marker="o",errorbar="sd")

plt.xscale("log")
plt.gca().invert_xaxis()

plt.xlabel("Block Size")
plt.ylabel("Retention Imbalance")
plt.title("Retention Imbalance Across Practice Schedule Continuum")
plt.tight_layout()

plt.savefig(os.path.join(image_folder, "plot4_retention_imbalance_by_block_size.jpg"), dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# ============================================================
# Plot 5: Generalization loss vs block size
# ============================================================

plt.figure(figsize=(8, 5))

sns.lineplot(data=df_summary,x="BlockSize",y="GeneralizationLoss",marker="o",errorbar="sd")

plt.xscale("log")
plt.gca().invert_xaxis()

plt.xlabel("Block Size")
plt.ylabel("Generalization Loss")
plt.title("Generalization Loss Across Practice Schedule Continuum")
plt.tight_layout()

plt.savefig(os.path.join(image_folder, "plot5_generalization_loss_by_block_size.jpg"), dpi=600, bbox_inches="tight")
plt.show()
plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    