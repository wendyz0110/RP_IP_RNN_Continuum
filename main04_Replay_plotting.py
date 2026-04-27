import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Load results from replay experiment folder
# This folder contains BOTH:
#   (1) before-replay metrics
#   (2) after-replay / balanced-replay metrics
# ============================================================

lr = 0.02
results_folder = f"results_replay_lr_{str(lr).replace('.', '_')}"

# -------------------------
# Before replay (same as original)
# -------------------------
loss_test_pre_array = np.load(os.path.join(results_folder, 'loss_test_pre_array.npy'))
loss_array_blocked_array = np.load(os.path.join(results_folder, 'loss_array_blocked_array.npy'))
loss_retention_blocked_array = np.load(os.path.join(results_folder, 'loss_retention_blocked_array.npy'))
loss_test_blocked_array = np.load(os.path.join(results_folder, 'loss_test_blocked_array.npy'))
loss_array_random_array = np.load(os.path.join(results_folder, 'loss_array_random_array.npy'))
loss_retention_random_array = np.load(os.path.join(results_folder, 'loss_retention_random_array.npy'))
loss_test_random_array = np.load(os.path.join(results_folder, 'loss_test_random_array.npy'))
loss_retention_array_blocked_array = np.load(os.path.join(results_folder, 'loss_retention_array_blocked_array.npy'))
loss_test_array_blocked_array = np.load(os.path.join(results_folder, 'loss_test_array_blocked_array.npy'))
loss_retention_array_random_array = np.load(os.path.join(results_folder, 'loss_retention_array_random_array.npy'))
loss_test_array_random_array = np.load(os.path.join(results_folder, 'loss_test_array_random_array.npy'))
loss_test_array_pre_array = np.load(os.path.join(results_folder, 'loss_test_array_pre_array.npy'))
loss_retention_noisy_array_blocked = np.load(os.path.join(results_folder, 'loss_retention_noisy_array_blocked.npy'))
loss_test_noisy_array_blocked = np.load(os.path.join(results_folder, 'loss_test_noisy_array_blocked.npy'))
loss_retention_noisy_array_random = np.load(os.path.join(results_folder, 'loss_retention_noisy_array_random.npy'))
loss_test_noisy_array_random = np.load(os.path.join(results_folder, 'loss_test_noisy_array_random.npy'))
loss_retention_pruned_array_blocked = np.load(os.path.join(results_folder, 'loss_retention_pruned_array_blocked.npy'))
loss_test_pruned_array_blocked = np.load(os.path.join(results_folder, 'loss_test_pruned_array_blocked.npy'))
loss_retention_pruned_array_random = np.load(os.path.join(results_folder, 'loss_retention_pruned_array_random.npy'))
loss_test_pruned_array_random = np.load(os.path.join(results_folder, 'loss_test_pruned_array_random.npy'))
loss_retention_interf_array_blocked = np.load(os.path.join(results_folder, 'loss_retention_interf_array_blocked.npy'))
loss_test_interf_array_blocked = np.load(os.path.join(results_folder, 'loss_test_interf_array_blocked.npy'))
loss_retention_interf_array_random = np.load(os.path.join(results_folder, 'loss_retention_interf_array_random.npy'))
loss_test_interf_array_random = np.load(os.path.join(results_folder, 'loss_test_interf_array_random.npy'))

# -------------------------
# After replay (Balanced Replay)
# -------------------------
loss_array_rp_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_array_rp_replay_balanced.npy'))
loss_retention_rp_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_retention_rp_replay_balanced.npy'))
loss_test_rp_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_test_rp_replay_balanced.npy'))
loss_retention_array_rp_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_retention_array_rp_replay_balanced.npy'))
loss_test_array_rp_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_test_array_rp_replay_balanced.npy'))

loss_array_ip_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_array_ip_replay_balanced.npy'))
loss_retention_ip_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_retention_ip_replay_balanced.npy'))
loss_test_ip_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_test_ip_replay_balanced.npy'))
loss_retention_array_ip_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_retention_array_ip_replay_balanced.npy'))
loss_test_array_ip_replay_balanced_array = np.load(os.path.join(results_folder, 'loss_test_array_ip_replay_balanced.npy'))

loss_retention_noisy_array_rp_replay = np.load(os.path.join(results_folder, 'loss_retention_noisy_array_rp_replay.npy'))
loss_test_noisy_array_rp_replay = np.load(os.path.join(results_folder, 'loss_test_noisy_array_rp_replay.npy'))
loss_retention_noisy_array_ip_replay = np.load(os.path.join(results_folder, 'loss_retention_noisy_array_ip_replay.npy'))
loss_test_noisy_array_ip_replay = np.load(os.path.join(results_folder, 'loss_test_noisy_array_ip_replay.npy'))

loss_retention_pruned_array_rp_replay = np.load(os.path.join(results_folder, 'loss_retention_pruned_array_rp_replay.npy'))
loss_test_pruned_array_rp_replay = np.load(os.path.join(results_folder, 'loss_test_pruned_array_rp_replay.npy'))
loss_retention_pruned_array_ip_replay = np.load(os.path.join(results_folder, 'loss_retention_pruned_array_ip_replay.npy'))
loss_test_pruned_array_ip_replay = np.load(os.path.join(results_folder, 'loss_test_pruned_array_ip_replay.npy'))

loss_retention_interf_array_rp_replay = np.load(os.path.join(results_folder, 'loss_retention_interf_array_rp_replay.npy'))
loss_test_interf_array_rp_replay = np.load(os.path.join(results_folder, 'loss_test_interf_array_rp_replay.npy'))
loss_retention_interf_array_ip_replay = np.load(os.path.join(results_folder, 'loss_retention_interf_array_ip_replay.npy'))
loss_test_interf_array_ip_replay = np.load(os.path.join(results_folder, 'loss_test_interf_array_ip_replay.npy'))

# -------------------------
# Selected Replay (RP only)
# -------------------------
loss_array_rp_replay_selected_array = np.load(os.path.join(results_folder, 'loss_array_rp_replay_selected.npy'))
loss_retention_rp_replay_selected_array = np.load(os.path.join(results_folder, 'loss_retention_rp_replay_selected.npy'))
loss_test_rp_replay_selected_array = np.load(os.path.join(results_folder, 'loss_test_rp_replay_selected.npy'))
loss_retention_array_rp_replay_selected_array = np.load(os.path.join(results_folder, 'loss_retention_array_rp_replay_selected.npy'))
loss_test_array_rp_replay_selected_array = np.load(os.path.join(results_folder, 'loss_test_array_rp_replay_selected.npy'))

loss_retention_noisy_array_rp_replay_selected = np.load(os.path.join(results_folder, 'loss_retention_noisy_array_rp_replay_selected.npy'))
loss_test_noisy_array_rp_replay_selected = np.load(os.path.join(results_folder, 'loss_test_noisy_array_rp_replay_selected.npy'))
loss_retention_pruned_array_rp_replay_selected = np.load(os.path.join(results_folder, 'loss_retention_pruned_array_rp_replay_selected.npy'))
loss_test_pruned_array_rp_replay_selected = np.load(os.path.join(results_folder, 'loss_test_pruned_array_rp_replay_selected.npy'))
loss_retention_interf_array_rp_replay_selected = np.load(os.path.join(results_folder, 'loss_retention_interf_array_rp_replay_selected.npy'))
loss_test_interf_array_rp_replay_selected = np.load(os.path.join(results_folder, 'loss_test_interf_array_rp_replay_selected.npy'))

# =================================================
#         Learning Curves (before replay)
# =================================================

# Plot 1: Average loss across all runs after each epoch
plt.figure()
plt.plot(range(1, loss_array_blocked_array.shape[1]+1), loss_array_blocked_array.mean(axis=0), label='RP')
plt.plot(range(1, loss_array_random_array.shape[1]+1), loss_array_random_array.mean(axis=0), label='IP')
# phase boundaries
for v in [160, 320]:
    plt.axvline(x=v, linestyle='--', color='gray', alpha=0.6)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0.4, 2.2)
plt.legend()
plt.title('Average training loss per epoch')
plt.savefig("images_replay/Average training loss per epoch.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =================================================
#    Learning Curves (During replay training)
# =================================================

plt.figure()

# RP after replay
plt.plot(range(1, loss_array_rp_replay_balanced_array.shape[1] + 1), loss_array_rp_replay_balanced_array.mean(axis=0), label='RP (Replay)')
# IP after replay
plt.plot(range(1, loss_array_ip_replay_balanced_array.shape[1] + 1), loss_array_ip_replay_balanced_array.mean(axis=0), label='IP (Replay)')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0.5, 2.0)
plt.legend()
plt.title('Average training loss per epoch (Replay Phase)')

plt.savefig("images_replay/Average training loss per epoch replay.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =================================================
#    Learning Curves (RP: Balanced vs Selected Replay)
# =================================================

plt.figure()

# RP - Balanced Replay
plt.plot(range(1, loss_array_rp_replay_balanced_array.shape[1] + 1),loss_array_rp_replay_balanced_array.mean(axis=0),label='RP (Balanced Replay)')
# RP - Selected Replay
plt.plot(range(1, loss_array_rp_replay_selected_array.shape[1] + 1),loss_array_rp_replay_selected_array.mean(axis=0),label='RP (Selected Replay)', color='#2ca02c')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0.5, 2.0)
plt.legend()
plt.title('RP Learning Curve: Balanced vs Selected Replay')

plt.savefig("images_replay/RP_learning_curve_balanced_vs_selected.jpg",dpi=600,bbox_inches="tight")
plt.show()
plt.close()

# =================================================
#       Average Retention Losses Bar Plot
# =================================================

rp_pre = loss_retention_array_blocked_array.mean()
ip_pre = loss_retention_array_random_array.mean()
rp_replay_balanced = loss_retention_array_rp_replay_balanced_array.mean()
ip_replay = loss_retention_array_ip_replay_balanced_array.mean()
rp_replay_selected = loss_retention_array_rp_replay_selected_array.mean()

plt.figure(figsize=(8,6))
x = np.arange(2) # RP, IP
width = 0.25

# ----- RP (3 bars) -----
plt.bar(x[0] - width, rp_pre, width, color='#1f77b4', alpha=0.4, label='RP Initial')
plt.bar(x[0], rp_replay_balanced, width, color='#1f77b4', alpha=1.0, label='RP Balanced Replay')
plt.bar(x[0] + width, rp_replay_selected, width, color='#2ca02c', alpha=1.0, label='RP Selected Replay')

# ----- IP (2 bars) -----
plt.bar(x[1] - width/2, ip_pre, width, color='#ff7f0e', alpha=0.4, label='IP Initial')
plt.bar(x[1] + width/2, ip_replay, width, color='#ff7f0e', alpha=1.0, label='IP Replay')

plt.xticks(x, ['RP', 'IP'])
plt.ylabel('Average Retention Loss')
plt.title('Retention Performance: Before vs After Replay')
plt.ylim(0, 2.0)

# fix legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.savefig("images_replay/bar_retention_pre_vs_replay.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# not surprising, IP has better retention from initial training, and RP improved a lot due to replay

# =================================================
#     Average Generalization Losses Bar Plot
# =================================================

rp_pre = loss_test_array_blocked_array.mean()
ip_pre = loss_test_array_random_array.mean()
rp_replay_balanced = loss_test_array_rp_replay_balanced_array.mean()
ip_replay = loss_test_array_ip_replay_balanced_array.mean()
rp_replay_selected = loss_test_array_rp_replay_selected_array.mean()

plt.figure(figsize=(8,6))
x = np.arange(2)  # RP, IP
width = 0.25

# ----- RP (3 bars) -----
plt.bar(x[0] - width, rp_pre, width, color='#1f77b4', alpha=0.4, label='RP Initial')
plt.bar(x[0], rp_replay_balanced, width, color='#1f77b4', alpha=1.0, label='RP Balanced Replay')
plt.bar(x[0] + width, rp_replay_selected, width, color='#2ca02c', alpha=1.0, label='RP Selected Replay')

# ----- IP (2 bars) -----
plt.bar(x[1] - width/2, ip_pre, width, color='#ff7f0e', alpha=0.4, label='IP Initial')
plt.bar(x[1] + width/2, ip_replay, width, color='#ff7f0e', alpha=1.0, label='IP Replay')

plt.xticks(x, ['RP', 'IP'])
plt.ylabel('Average Generalization Loss')
plt.title('Generalization Performance: RP Balanced vs Selected Replay')
plt.ylim(0, 2.4)

# fix legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.savefig("images_replay/bar_generalization_pre_vs_replay.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# small overfitting happened to IP

# =================================================
#     Violin for Per Sequence Retention Loss
# =================================================

# Violin Plot for Initial Training Results
seq_num = 3
# split the retention loss according to sequences
loss_retention_array_blocked_each = np.split(loss_retention_array_blocked_array, seq_num, axis=1)
loss_retention_array_random_each = np.split(loss_retention_array_random_array, seq_num, axis=1)

# flatten each block into a list, storing all the loss pre sequence
loss_retention_array_blocked_each_vec = [arr.flatten() for arr in loss_retention_array_blocked_each]
loss_retention_array_random_each_vec = [arr.flatten() for arr in loss_retention_array_random_each]

# put information into dataframe
df_blocked = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_blocked_each_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_blocked_each_vec[i]) for i in range(seq_num)]),
    'Condition': 'RP'
})

df_random = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_random_each_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_random_each_vec[i]) for i in range(seq_num)]),
    'Condition': 'IP'
})

# stack
df_all = pd.concat([df_blocked, df_random], ignore_index=True)

# generate the violin plot
plt.figure(figsize=(10, 5))
sns.violinplot(
    x='Phase',
    y='Loss',
    hue='Condition', # RP vs IP
    data=df_all,
    palette={'RP': '#1f77b4', 'IP': '#ff7f0e'},
)
ymin, ymax = df_all['Loss'].min(), df_all['Loss'].max()
margin = 0.05 * (ymax - ymin)
plt.ylim(ymin - margin, ymax + margin)
plt.title('Retention Distribution: RP vs IP (no replay)', fontsize=16)
plt.savefig("images_replay/violin_retention_initial_training.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =============== Retention AFTER replay ================
loss_retention_array_rp_replay_each = np.split(loss_retention_array_rp_replay_balanced_array, seq_num, axis=1)
loss_retention_array_ip_replay_each = np.split(loss_retention_array_ip_replay_balanced_array, seq_num, axis=1)

# flatten
loss_retention_array_rp_replay_vec = [arr.flatten() for arr in loss_retention_array_rp_replay_each]
loss_retention_array_ip_replay_vec = [arr.flatten() for arr in loss_retention_array_ip_replay_each]

# dataframe for RP
df_rp_replay = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_rp_replay_vec),
    'Phase': np.concatenate([
        ['Seq-' + str(i+1)] * len(loss_retention_array_rp_replay_vec[i])
        for i in range(seq_num)
    ]),
    'Condition': 'RP (Replay)'
})

# dataframe for IP
df_ip_replay = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_ip_replay_vec),
    'Phase': np.concatenate([
        ['Seq-' + str(i+1)] * len(loss_retention_array_ip_replay_vec[i])
        for i in range(seq_num)
    ]),
    'Condition': 'IP (Replay)'
})

df_all_replay = pd.concat([df_rp_replay, df_ip_replay], ignore_index=True)

plt.figure(figsize=(10, 5))
sns.violinplot(
    x='Phase',
    y='Loss',
    hue='Condition',
    data=df_all_replay,
    palette={
        'RP (Replay)': '#1f77b4',
        'IP (Replay)': '#ff7f0e'
    },
)
ymin, ymax = df_all['Loss'].min(), df_all['Loss'].max()
margin = 0.05 * (ymax - ymin)
plt.ylim(ymin - margin, ymax + margin)
plt.title('Retention Distribution: RP vs IP Balanced Replay', fontsize=16)
plt.savefig("images_replay/violin_retention_after_replay.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =================================================
#          Violin for RP Selected Replay
# =================================================

# =============== RP: Balanced vs Selected Replay ================

# split
loss_retention_array_rp_bal_each = np.split(loss_retention_array_rp_replay_balanced_array, seq_num, axis=1)
loss_retention_array_rp_sel_each = np.split(loss_retention_array_rp_replay_selected_array, seq_num, axis=1)

# flatten
loss_retention_array_rp_bal_vec = [arr.flatten() for arr in loss_retention_array_rp_bal_each]
loss_retention_array_rp_sel_vec = [arr.flatten() for arr in loss_retention_array_rp_sel_each]

# dataframe balanced
df_rp_bal = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_rp_bal_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_rp_bal_vec[i]) for i in range(seq_num)]),
    'Condition': 'RP (Balanced Replay)'
})

# dataframe selected
df_rp_sel = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_rp_sel_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_rp_sel_vec[i]) for i in range(seq_num)]),
    'Condition': 'RP (Selected Replay)'
})

df_all = pd.concat([df_rp_bal, df_rp_sel], ignore_index=True)

plt.figure(figsize=(10,5))
sns.violinplot(x='Phase', y='Loss', hue='Condition', data=df_all, palette={'RP (Balanced Replay)': '#1f77b4', 'RP (Selected Replay)': '#2ca02c'})
ymin, ymax = df_all['Loss'].min(), df_all['Loss'].max()
margin = 0.05 * (ymax - ymin)
plt.ylim(ymin - margin, ymax + margin)
plt.title('RP Retention Distribution: Balanced vs Selected Replay', fontsize=16)
plt.savefig("images_replay/violin_rp_balanced_vs_selected.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =============== RP Selected vs IP Balanced ================

# reuse RP selected
loss_retention_array_rp_sel_each = np.split(loss_retention_array_rp_replay_selected_array, seq_num, axis=1)
loss_retention_array_rp_sel_vec = [arr.flatten() for arr in loss_retention_array_rp_sel_each]

# IP balanced
loss_retention_array_ip_bal_each = np.split(loss_retention_array_ip_replay_balanced_array, seq_num, axis=1)
loss_retention_array_ip_bal_vec = [arr.flatten() for arr in loss_retention_array_ip_bal_each]

# dataframe RP selected
df_rp_sel = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_rp_sel_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_rp_sel_vec[i]) for i in range(seq_num)]),
    'Condition': 'RP (Selected Replay)'
})

# dataframe IP balanced
df_ip_bal = pd.DataFrame({
    'Loss': np.concatenate(loss_retention_array_ip_bal_vec),
    'Phase': np.concatenate([['Seq-' + str(i+1)] * len(loss_retention_array_ip_bal_vec[i]) for i in range(seq_num)]),
    'Condition': 'IP (Balanced Replay)'
})

df_all = pd.concat([df_rp_sel, df_ip_bal], ignore_index=True)

plt.figure(figsize=(10,5))
sns.violinplot(x='Phase', y='Loss', hue='Condition', data=df_all, palette={'RP (Selected Replay)': '#2ca02c', 'IP (Balanced Replay)': '#ff7f0e'})
plt.ylim(0.0,3.5)
plt.title('Retention Distribution: RP Selected vs IP Balanced', fontsize=16)
plt.savefig("images_replay/violin_rp_selected_vs_ip_balanced.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =================================================
#     Violin for Per Sequence Generalization Loss
# =================================================

# ---------- PRE ----------
pre_vec = loss_test_array_pre_array.flatten()

df_pre_rp = pd.DataFrame({
    'Loss': pre_vec,
    'Phase': 'Pre',
    'Condition': 'RP'
})

df_pre_ip = pd.DataFrame({
    'Loss': pre_vec,
    'Phase': 'Pre',
    'Condition': 'IP'
})

# ---------- AFTER TRAINING ----------
df_train_rp = pd.DataFrame({
    'Loss': loss_test_array_blocked_array.flatten(),
    'Phase': 'After Training',
    'Condition': 'RP'
})

df_train_ip = pd.DataFrame({
    'Loss': loss_test_array_random_array.flatten(),
    'Phase': 'After Training',
    'Condition': 'IP'
})

# ---------- AFTER REPLAY ----------
df_replay_rp = pd.DataFrame({
    'Loss': loss_test_array_rp_replay_balanced_array.flatten(),
    'Phase': 'After Replay',
    'Condition': 'RP'
})

df_replay_ip = pd.DataFrame({
    'Loss': loss_test_array_ip_replay_balanced_array.flatten(),
    'Phase': 'After Replay',
    'Condition': 'IP'
})

# ---------- COMBINE ----------
df_all = pd.concat([
    df_pre_rp, df_pre_ip,
    df_train_rp, df_train_ip,
    df_replay_rp, df_replay_ip
], ignore_index=True)

# ---------- PLOT ----------
plt.figure(figsize=(10, 5))
sns.violinplot(
    x='Phase',
    y='Loss',
    hue='Condition',
    data=df_all,
    palette={'RP': '#1f77b4', 'IP': '#ff7f0e'}
)

plt.title('Generalization Performance Across Training Phases', fontsize=16)
plt.ylabel('Loss')
plt.xlabel('Phase')

plt.savefig("images_replay/violin_generalization_phases.jpg", dpi=600, bbox_inches="tight")
plt.show()
plt.close()

# =================================================
#       Vulnerability Test Results (Before vs. Balanced Replay)
# =================================================

def plot_four_curves(
    data_rp_before, data_ip_before,
    data_rp_after, data_ip_after,
    xlabel, ylabel, title
):
    plt.figure()

    mean_rp_before = data_rp_before.mean(axis=0)
    mean_ip_before = data_ip_before.mean(axis=0)

    mean_rp_after = data_rp_after.mean(axis=0)
    mean_ip_after = data_ip_after.mean(axis=0)

    x = np.arange(mean_rp_before.shape[0])

    plt.plot(x, mean_rp_before, label='RP (Before)', color='#1f77b4')
    plt.plot(x, mean_ip_before, label='IP (Before)', color='#ff7f0e')

    plt.plot(x, mean_rp_after, linestyle='--', label='RP (Replay)', color='#1f77b4')
    plt.plot(x, mean_ip_after, linestyle='--', label='IP (Replay)', color='#ff7f0e')

    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend(fontsize=11)
    plt.title(title, fontsize=12)

    plt.savefig("images_replay/" + title.replace(" ", "_") + ".jpg", dpi=600)
    plt.show()
    plt.close()

# Noise tests
plot_four_curves(
    loss_retention_noisy_array_blocked,
    loss_retention_noisy_array_random,
    loss_retention_noisy_array_rp_replay,
    loss_retention_noisy_array_ip_replay,
    'Repetition of noise injection', 'Loss',
    'Noise Vulnerability - Trained Sequence (Before vs Replay)'
)

plot_four_curves(
    loss_test_noisy_array_blocked,
    loss_test_noisy_array_random,
    loss_test_noisy_array_rp_replay,
    loss_test_noisy_array_ip_replay,
    'Repetition of noise injection', 'Loss',
    'Noise Vulnerability - Generalization (Before vs Replay)'
)

# Pruning tests
plot_four_curves(
    loss_retention_pruned_array_blocked,
    loss_retention_pruned_array_random,
    loss_retention_pruned_array_rp_replay,
    loss_retention_pruned_array_ip_replay,
    'Repetition of weight pruning', 'Loss',
    'Pruning Vulnerability - Trained Sequence (Before vs Replay)'
)

plot_four_curves(
    loss_test_pruned_array_blocked,
    loss_test_pruned_array_random,
    loss_test_pruned_array_rp_replay,
    loss_test_pruned_array_ip_replay,
    'Repetition of weight pruning', 'Loss',
    'Pruning Vulnerability - Generalization (Before vs Replay)'
)

# Interference tests
plot_four_curves(
    loss_retention_interf_array_blocked,
    loss_retention_interf_array_random,
    loss_retention_interf_array_rp_replay,
    loss_retention_interf_array_ip_replay,
    'Retraining Epoch', 'Loss',
    'Interference Vulnerability - Trained Sequence (Before vs Replay)'
)

plot_four_curves(
    loss_test_interf_array_blocked,
    loss_test_interf_array_random,
    loss_test_interf_array_rp_replay,
    loss_test_interf_array_ip_replay,
    'Retraining Epoch', 'Loss',
    'Interference Vulnerability - Generalization (Before vs Replay)'
)


# =================================================
#    Vulnerability Test Results (3 RP conditions)
# =================================================

def plot_3_curves(rp_before, rp_balanced, rp_selected, title, xlabel, ylabel, save_path):
    mean_before = rp_before.mean(axis=0)
    mean_balanced = rp_balanced.mean(axis=0)
    mean_selected = rp_selected.mean(axis=0)

    plt.figure()
    plt.plot(mean_before, label='RP (Before Replay)', linestyle='--')
    plt.plot(mean_balanced, label='RP (Balanced Replay)', linewidth=2)
    plt.plot(mean_selected, label='RP (Selected Replay)', linewidth=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

# Noise Retention
plot_3_curves(loss_retention_noisy_array_blocked, loss_retention_noisy_array_rp_replay, loss_retention_noisy_array_rp_replay_selected, 'RP Noise Vulnerability (Retention)', 'Noise Steps', 'Retention Loss', 'images_replay/noise_retention_rp.jpg')
# Noise generalization
plot_3_curves(loss_test_noisy_array_blocked, loss_test_noisy_array_rp_replay, loss_test_noisy_array_rp_replay_selected, 'RP Noise Vulnerability (Generalization)', 'Noise Steps', 'Generalization Loss', 'images_replay/noise_generalization_rp.jpg')
# Pruning retention
plot_3_curves(loss_retention_pruned_array_blocked, loss_retention_pruned_array_rp_replay, loss_retention_pruned_array_rp_replay_selected, 'RP Pruning Vulnerability (Retention)', 'Pruning Steps', 'Retention Loss', 'images_replay/pruning_retention_rp.jpg')
# Pruning generalization
plot_3_curves(loss_test_pruned_array_blocked, loss_test_pruned_array_rp_replay, loss_test_pruned_array_rp_replay_selected, 'RP Pruning Vulnerability (Generalization)', 'Pruning Steps', 'Generalization Loss', 'images_replay/pruning_generalization_rp.jpg')
# Interference retention
plot_3_curves(loss_retention_interf_array_blocked, loss_retention_interf_array_rp_replay, loss_retention_interf_array_rp_replay_selected, 'RP Interference Vulnerability (Retention)', 'Interference Steps', 'Retention Loss', 'images_replay/interference_retention_rp.jpg')
# Interference generalization
plot_3_curves(loss_test_interf_array_blocked, loss_test_interf_array_rp_replay, loss_test_interf_array_rp_replay_selected, 'RP Interference Vulnerability (Generalization)', 'Interference Steps', 'Generalization Loss', 'images_replay/interference_generalization_rp.jpg')















