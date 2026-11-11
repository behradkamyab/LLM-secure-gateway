from matplotlib import pyplot as plt
import numpy as np
import os

# Create docs/charts directory if it doesn't exist
os.makedirs("docs/charts", exist_ok=True)

# Data from your diagnostic run
labels = ['Recall', 'Precision', 'F1-Score']
regex_baseline = [0.07, 0.02, 0.03]
deberta_zeroshot = [0.54, 0.62, 0.57]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, regex_baseline, width, label='Regex Baseline', color='#ff9999')
rects2 = ax.bar(x + width/2, deberta_zeroshot, width, label='DeBERTa Zero-Shot', color='#66b3ff')

ax.set_ylabel('Score')
ax.set_title('PII Redaction: Baseline vs Zero-Shot AI (Domain Shift Vulnerability)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('docs/charts/ner_comparison.png', dpi=300, bbox_inches='tight')
print("Chart saved to docs/charts/ner_comparison.png")