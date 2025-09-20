#!/usr/bin/env python3
"""
Create a standalone legend for the figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use the exact colors from the reference code
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
COLOR_SL, COLOR_SSL = colors[0], colors[2]

# Create standalone legend - stacked vertically (tall, not wide)
fig, ax = plt.subplots(figsize=(2.5, 2))  # Taller than wide for stacked layout
ax.axis('off')

# Create legend patches
sl_patch = mpatches.Patch(color=COLOR_SL, label='DeepECG-SL')
ssl_patch = mpatches.Patch(color=COLOR_SSL, label='DeepECG-SSL')

# Add legend - stacked vertically (ncol=1)
legend = ax.legend(handles=[sl_patch, ssl_patch], 
                   loc='center', 
                   frameon=True,
                   fontsize=14,
                   ncol=1,  # Single column for vertical stacking
                   columnspacing=1)

# Style the legend frame
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)
legend.get_frame().set_linewidth(1)

plt.tight_layout()
# Save as both PNG and SVG
plt.savefig('/volume/fairseq-signals/legend.png', dpi=300, bbox_inches='tight')
plt.savefig('/volume/fairseq-signals/legend.svg', format='svg', bbox_inches='tight')
plt.show()
print("Legend saved as 'legend.png' and 'legend.svg'")