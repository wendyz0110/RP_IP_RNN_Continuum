import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load results from serial + random-block continuum experiment
#
# This folder contains:
#   (1) pre-training generalization metrics
#   (2) training/retention/generalization metrics by schedule type and block size
#   (3) sequence-level retention and retention imbalance
#   (4) schedule indices for debugging
# ============================================================

lr = 0.02
results_folder = f"results_block_continuum_serial_and_random_lr_{str(lr).replace('.', '_')}"

# -------------------------
# Pre-training results
# -------------------------
loss_test_pre_array = np.load(os.path.join(results_folder, "loss_test_pre_array.npy"))
loss_test_array_pre_array = np.load(os.path.join(results_folder, "loss_test_array_pre_array.npy"))

# -------------------------
# Schedule-type dictionaries
# Need allow_pickle=True + .item() because these were saved as nested dictionaries
# Structure:
#   metric_by_schedule[schedule_type][block_size]
# -------------------------
loss_array_by_schedule = np.load(os.path.join(results_folder, "loss_array_by_schedule.npy"), allow_pickle=True).item()
loss_retention_by_schedule = np.load(os.path.join(results_folder, "loss_retention_by_schedule.npy"), allow_pickle=True).item()
loss_test_by_schedule = np.load(os.path.join(results_folder, "loss_test_by_schedule.npy"), allow_pickle=True).item()
loss_retention_array_by_schedule = np.load(os.path.join(results_folder, "loss_retention_array_by_schedule.npy"), allow_pickle=True).item()
loss_test_array_by_schedule = np.load(os.path.join(results_folder, "loss_test_array_by_schedule.npy"), allow_pickle=True).item()
seq_retention_means_by_schedule = np.load(os.path.join(results_folder, "seq_retention_means_by_schedule.npy"), allow_pickle=True).item()
retention_imbalance_by_schedule = np.load(os.path.join(results_folder, "retention_imbalance_by_schedule.npy"), allow_pickle=True).item()
schedule_idx_by_schedule = np.load(os.path.join(results_folder, "schedule_idx_by_schedule.npy"), allow_pickle=True).item()

print("Loaded continuum results from:", results_folder)

# ============================================================
# Basic settings
# ============================================================

block_sizes = np.array([1000, 500, 200, 100, 50, 20, 10, 5, 1])
selected_blocks = [1000, 100, 10, 1]
schedule_types = ["serial", "random_block"] 
num_runs = len(loss_retention_by_schedule["serial"][block_sizes[0]])
num_blocks = len(block_sizes)
schedule_order = [f"Block-{b}" for b in block_sizes]
x_positions = np.arange(len(schedule_order))

# image folder
image_folder = "images_continuum_serial_and_random"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# ============================================================
# Helper function: mean ± SEM by schedule label
# ============================================================

def mean_sem_by_schedule(df, value_col, schedule_order):
    grouped = df.groupby("ScheduleLabel", observed=True)[value_col].agg(["mean", "sem"])
    grouped.index = grouped.index.astype(str)
    grouped = grouped.reindex(schedule_order)
    return grouped["mean"], grouped["sem"]

# ============================================================
# Helper function: create DataFrames and plots for one schedule type
# ============================================================

def plot_one_schedule_type(schedule_type):
    print(f"\n================ Plotting schedule type: {schedule_type} ================")

    # readable label for titles / filenames
    if schedule_type == "serial":
        schedule_title = "Serial Block Order"
        file_prefix = "serial"
    elif schedule_type == "random_block":
        schedule_title = "Random Block Order"
        file_prefix = "random_block"
    else:
        schedule_title = schedule_type
        file_prefix = schedule_type

    # ------------------------------------------------------------
    # Summary dataframe: one row per run per block size
    # ------------------------------------------------------------

    summary_rows = []

    for block_size in block_sizes:
        for run_idx in range(num_runs):
            summary_rows.append({
                "Run": run_idx,
                "BlockSize": block_size,
                "ScheduleLabel": f"Block-{block_size}",
                "RetentionLoss": loss_retention_by_schedule[schedule_type][block_size][run_idx],
                "GeneralizationLoss": loss_test_by_schedule[schedule_type][block_size][run_idx],
                "RetentionImbalance": retention_imbalance_by_schedule[schedule_type][block_size][run_idx]
            })

    df_summary = pd.DataFrame(summary_rows)

    df_summary["ScheduleLabel"] = pd.Categorical(
        df_summary["ScheduleLabel"],
        categories=schedule_order,
        ordered=True
    )

    # ------------------------------------------------------------
    # Sequence-level retention dataframe
    # One row per run × block size × trained sequence
    # ------------------------------------------------------------

    seq_rows = []

    for block_size in block_sizes:
        seq_means = seq_retention_means_by_schedule[schedule_type][block_size]  # shape: (num_runs, 3)

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
        categories=schedule_order,
        ordered=True
    )

    df_seq_retention["Sequence"] = pd.Categorical(
        df_seq_retention["Sequence"],
        categories=["Seq-1", "Seq-2", "Seq-3"],
        ordered=True
    )

    # ------------------------------------------------------------
    # Training loss curve dataframe
    # One row per run × selected block size × epoch
    # ------------------------------------------------------------

    training_rows = []

    for block_size in selected_blocks:
        loss_array = loss_array_by_schedule[schedule_type][block_size]  # shape: (num_runs, num_epochs)

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
    
    # check shapes
    print("df_summary:", df_summary.shape)
    print("df_seq_retention:", df_seq_retention.shape)
    print("df_training:", df_training.shape)

    # ============================================================
    # Plot 1: Training acquisition curves by selected block sizes
    # ============================================================

    plt.figure(figsize=(9, 5))

    sns.lineplot(
        data=df_training,
        x="Epoch",
        y="TrainingLoss",
        hue="ScheduleLabel",
        errorbar=None
    )

    plt.xlabel("Training Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Curves: {schedule_title}")
    plt.legend(title="Practice Schedule")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"{file_prefix}_plot1_training_acquisition_curves.jpg"), dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    # ============================================================
    # Plot 2: Average retention loss vs block size
    # ============================================================

    retention_means, retention_sems = mean_sem_by_schedule(df_summary, "RetentionLoss", schedule_order)

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        x_positions,
        retention_means.values,
        yerr=retention_sems.values,
        marker="o",
        linestyle="-",
        capsize=4
    )

    plt.xticks(x_positions, schedule_order, rotation=45)
    plt.xlabel("Practice Schedule (larger block = more repetitive)")
    plt.ylabel("Average Retention Loss")
    plt.title(f"Average Retention Loss: {schedule_title}")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"{file_prefix}_plot2_avg_retention_loss_by_block_size.jpg"), dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    # ============================================================
    # Plot 3: Sequence-level retention loss vs block size
    # ============================================================

    seq_summary = (
        df_seq_retention
        .groupby(["ScheduleLabel", "Sequence"], observed=True)["SeqRetentionLoss"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    plt.figure(figsize=(9, 5))

    for seq in ["Seq-1", "Seq-2", "Seq-3"]:
        seq_data = seq_summary[seq_summary["Sequence"] == seq].copy()
        seq_data["ScheduleLabel"] = pd.Categorical(
            seq_data["ScheduleLabel"],
            categories=schedule_order,
            ordered=True
        )
        seq_data = seq_data.sort_values("ScheduleLabel")

        plt.errorbar(
            x_positions,
            seq_data["mean"].values,
            yerr=seq_data["sem"].values,
            marker="o",
            linestyle="-",
            capsize=4,
            label=seq
        )

    plt.xticks(x_positions, schedule_order, rotation=45)
    plt.xlabel("Practice Schedule")
    plt.ylabel("Sequence-Level Retention Loss")
    plt.title(f"Retention Loss for Each Trained Sequence: {schedule_title}")
    plt.legend(title="Trained Sequence")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"{file_prefix}_plot3_sequence_level_retention_by_block_size.jpg"), dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    # ============================================================
    # Plot 4: Retention imbalance vs block size
    # ============================================================

    imbalance_means, imbalance_sems = mean_sem_by_schedule(df_summary, "RetentionImbalance", schedule_order)

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        x_positions,
        imbalance_means.values,
        yerr=imbalance_sems.values,
        marker="o",
        linestyle="-",
        capsize=4
    )

    plt.xticks(x_positions, schedule_order, rotation=45)
    plt.xlabel("Practice Schedule")
    plt.ylabel("Retention Imbalance")
    plt.title(f"Retention Imbalance: {schedule_title}")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"{file_prefix}_plot4_retention_imbalance_by_block_size.jpg"), dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

    # ============================================================
    # Plot 5: Generalization loss vs block size
    # ============================================================

    generalization_means, generalization_sems = mean_sem_by_schedule(df_summary, "GeneralizationLoss", schedule_order)

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        x_positions,
        generalization_means.values,
        yerr=generalization_sems.values,
        marker="o",
        linestyle="-",
        capsize=4
    )

    plt.xticks(x_positions, schedule_order, rotation=45)
    plt.xlabel("Practice Schedule")
    plt.ylabel("Generalization Loss")
    plt.title(f"Generalization Loss: {schedule_title}")
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"{file_prefix}_plot5_generalization_loss_by_block_size.jpg"), dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()


# ============================================================
# Execution: Plot serial first, then random-block
# ============================================================

for schedule_type in schedule_types:
    plot_one_schedule_type(schedule_type)

print(f"\nAll plots saved to: {image_folder}")


# ============================================================
# Re-draw Plot 3 for random_block with fixed y-range
# ============================================================

schedule_type = "random_block"
schedule_title = "Random Block Order"
file_prefix = "random_block"

seq_rows = []

for block_size in block_sizes:
    seq_means = seq_retention_means_by_schedule[schedule_type][block_size]

    for run_idx in range(num_runs):
        for seq_idx in range(seq_means.shape[1]):
            seq_rows.append({
                "Run": run_idx,
                "BlockSize": block_size,
                "ScheduleLabel": f"Block-{block_size}",
                "Sequence": f"Seq-{seq_idx + 1}",
                "SeqRetentionLoss": seq_means[run_idx, seq_idx]
            })

df_seq_retention_random = pd.DataFrame(seq_rows)

df_seq_retention_random["ScheduleLabel"] = pd.Categorical(
    df_seq_retention_random["ScheduleLabel"],
    categories=schedule_order,
    ordered=True
)

df_seq_retention_random["Sequence"] = pd.Categorical(
    df_seq_retention_random["Sequence"],
    categories=["Seq-1", "Seq-2", "Seq-3"],
    ordered=True
)

seq_summary = (
    df_seq_retention_random
    .groupby(["ScheduleLabel", "Sequence"], observed=True)["SeqRetentionLoss"]
    .agg(["mean", "sem"])
    .reset_index()
)

plt.figure(figsize=(9, 5))

for seq in ["Seq-1", "Seq-2", "Seq-3"]:
    seq_data = seq_summary[seq_summary["Sequence"] == seq].copy()
    seq_data["ScheduleLabel"] = pd.Categorical(
        seq_data["ScheduleLabel"],
        categories=schedule_order,
        ordered=True
    )
    seq_data = seq_data.sort_values("ScheduleLabel")

    plt.errorbar(
        x_positions,
        seq_data["mean"].values,
        yerr=seq_data["sem"].values,
        marker="o",
        linestyle="-",
        capsize=4,
        label=seq
    )

plt.xticks(x_positions, schedule_order, rotation=45)
plt.ylim(1.05, 1.40)
plt.xlabel("Practice Schedule")
plt.ylabel("Sequence-Level Retention Loss")
plt.title(f"Retention Loss for Each Trained Sequence: {schedule_title}")
plt.legend(title="Trained Sequence")
plt.tight_layout()

plt.savefig(
    os.path.join(image_folder, f"{file_prefix}_plot3_sequence_level_retention_fixed_ylim.jpg"),
    dpi=600,
    bbox_inches="tight"
)

plt.show()
plt.close()









