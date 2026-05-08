import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from Data_Preprocessing import Mydata

warnings.filterwarnings('ignore')

# Timing data (seconds) from the benchmark run
sla_times = [46.2, 46.5, 14.1, 29.4]
cmf_times = [88.1, 64.9, 13.7, 15.4]

pairs = [
    ("Office → Movies", 'ratings_Office_Products.csv', 'ratings_Movies_and_TV.csv'),
    ("Sports → CDs", 'ratings_Sports_and_Outdoors.csv', 'ratings_CDs_and_Vinyl.csv'),
    ("Apps → Games", 'ratings_Apps_for_Android.csv', 'ratings_Video_Games.csv'),
    ("Toys → Auto", 'ratings_Toys_and_Games.csv', 'ratings_Automotive.csv')
]

base_path = '/Users/daksh15/RECSYS/Spectral-Latent-Alignment/Data'

counts = []
labels = []
for name, s_file, t_file in pairs:
    s_path = os.path.join(base_path, s_file)
    t_path = os.path.join(base_path, t_file)
    dataset = Mydata(s_path, t_path, train=None, preprocessed=True)
    
    s_cnt = np.count_nonzero(dataset.S_data)
    t_cnt = np.count_nonzero(dataset.T_data)
    total_cnt = s_cnt + t_cnt
    counts.append(total_cnt)
    
    # Format label to include dataset name and rating count
    labels.append(f"{name}\n({total_cnt:,} ratings)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

x = np.arange(len(labels))
width = 0.5

# Plot SLA
rects1 = ax1.bar(x, sla_times, width, color='#1f77b4', edgecolor='black')
ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
ax1.set_title('SLA Execution Time\n(Scales with matrix dimensions/rank)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)

for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f}s',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 4),  
                 textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot CMF
rects2 = ax2.bar(x, cmf_times, width, color='#ff7f0e', edgecolor='black')
ax2.set_title('CMF Execution Time\n(Scales with number of ratings)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=11)

for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f'{height:.1f}s',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 4),  
                 textcoords="offset points",
                 ha='center', va='bottom', fontweight='bold', fontsize=11)

ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('timing_vs_density.png', dpi=300)
print("Saved timing_vs_density.png")
