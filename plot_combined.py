import matplotlib.pyplot as plt
import numpy as np
import os

fractions = np.arange(10, 110, 10)

# SLA Data
sla_office = [1.2251, 1.2064, 1.2083, 1.2061, 1.1976, 1.1889, 1.2016, 1.1947, 1.2027, 1.2002]
sla_sports = [1.0009, 0.9668, 0.9504, 0.9662, 0.9383, 1.0054, 0.9799, 0.9572, 1.0222, 0.9626]
sla_apps = [1.1902, 1.1928, 1.2061, 1.1881, 1.1894, 1.1867, 1.1855, 1.1879, 1.1980, 1.1941]
sla_toys = [1.0872, 1.0919, 1.0880, 1.0797, 1.0804, 1.0804, 1.0698, 1.0765, 1.0786, 1.0729]

# CMF Data
cmf_office = [1.2120, 1.2166, 1.2200, 1.2191, 1.2132, 1.2167, 1.2150, 1.2154, 1.2127, 1.2148]
cmf_sports = [0.8107, 0.8082, 0.8086, 0.8067, 0.8061, 0.8068, 0.8086, 0.8069, 0.8094, 0.8086]
cmf_apps = [1.1413, 1.1373, 1.1344, 1.1321, 1.1312, 1.1314, 1.1297, 1.1279, 1.1258, 1.1267]
cmf_toys = [1.0257, 1.0216, 1.0201, 1.0195, 1.0196, 1.0199, 1.0190, 1.0188, 1.0186, 1.0191]

datasets = [
    ("Office Products -> Movies", sla_office, cmf_office),
    ("Sports -> CDs & Vinyls", sla_sports, cmf_sports),
    ("Android Apps -> Video Games", sla_apps, cmf_apps),
    ("Toys -> Automotive", sla_toys, cmf_toys)
]

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axs = axs.flatten()

for i, (title, sla_data, cmf_data) in enumerate(datasets):
    ax = axs[i]
    ax.plot(fractions, sla_data, marker='o', linewidth=2, color='#1f77b4', label='SLA (LightGBM)')
    ax.plot(fractions, cmf_data, marker='s', linewidth=2, color='#ff7f0e', label='CMF')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage of Anchors (%)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_ylim(0.75, 1.30)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=11)

plt.suptitle('SLA vs CMF Performance Across Anchor Fractions', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('combined_anchors_rmse.png', bbox_inches='tight', dpi=300)
print("Saved combined_anchors_rmse.png")
