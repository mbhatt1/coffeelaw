import numpy as np
import matplotlib.pyplot as plt

# Create Pe_ctx sweep (logarithmic spacing)
pe_ctx_values = np.logspace(-1, 5, 100)  # 0.1 to 100,000

# Coffee Law relationships
# 1. Cube-root sharpening: W/√D_eff ∝ Pe_ctx^(-1/3)
w_normalized = pe_ctx_values ** (-1/3)

# 2. Entropy scaling: H = a + (2/3)*ln(Pe_ctx)
a = 2.5  # baseline entropy
H = a + (2/3) * np.log(pe_ctx_values)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('Coffee Law Pe_ctx Sweep', fontsize=16, fontweight='bold')

# Plot 1: Cube-root Sharpening Law
ax1.loglog(pe_ctx_values, w_normalized, 'b-', linewidth=2.5, label='W/√D_eff ∝ Pe_ctx^(-1/3)')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlabel('Pe_ctx (Context Péclet Number)', fontsize=12)
ax1.set_ylabel('W/√D_eff (Normalized Width)', fontsize=12)
ax1.set_title('Cube-root Sharpening Law', fontsize=14)

# Add annotations for different Pe_ctx regions
ax1.axvspan(0.1, 1, alpha=0.2, color='red', label='High Ambiguity')
ax1.axvspan(1, 10, alpha=0.2, color='yellow', label='Production Range')
ax1.axvspan(10, 100, alpha=0.2, color='green', label='Diminishing Returns')
ax1.axvspan(100, 100000, alpha=0.2, color='blue', label='Extreme Optimization')

# Add slope indicator
x_slope = [1, 10]
y_slope = [1, 10**(-1/3)]
ax1.plot(x_slope, y_slope, 'r--', linewidth=2, alpha=0.7)
ax1.text(3, 0.3, 'Slope = -1/3', fontsize=11, color='red', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

ax1.legend(loc='upper right')

# Plot 2: Entropy Scaling Law
ax2.semilogx(pe_ctx_values, H, 'g-', linewidth=2.5, label='H = a + (2/3)ln(Pe_ctx)')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Pe_ctx (Context Péclet Number)', fontsize=12)
ax2.set_ylabel('H (Coarse Entropy)', fontsize=12)
ax2.set_title('Entropy Scaling Law', fontsize=14)

# Add regions
ax2.axvspan(0.1, 1, alpha=0.2, color='red')
ax2.axvspan(1, 10, alpha=0.2, color='yellow')
ax2.axvspan(10, 100, alpha=0.2, color='green')
ax2.axvspan(100, 100000, alpha=0.2, color='blue')

# Add slope indicator
x_log = [1, 10]
y_log = [a + (2/3)*np.log(1), a + (2/3)*np.log(10)]
ax2.plot(x_log, y_log, 'r--', linewidth=2, alpha=0.7)
ax2.text(3, 3.2, 'Slope = 2/3', fontsize=11, color='red',
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

ax2.legend(loc='lower right')

# Add context quality descriptions
fig.text(0.12, 0.02, 'Low Pe (<1):\nChaotic\n14-60% acc',
         fontsize=9, ha='left', va='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffcccc', alpha=0.8))

fig.text(0.32, 0.02, 'Good Pe (1-10):\nProduction\n60-81% acc',
         fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='#ffffcc', alpha=0.8))

fig.text(0.52, 0.02, 'High Pe (10-100):\nDiminishing\n81-91% acc',
         fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='#ccffcc', alpha=0.8))

fig.text(0.75, 0.02, 'Extreme Pe (>100):\nTheoretical\n91-98% acc',
         fontsize=9, ha='center', va='bottom',
         bbox=dict(boxstyle="round,pad=0.5", facecolor='#ccccff', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# Save the plot
plt.savefig('coffee_law_sweep.png', dpi=300, bbox_inches='tight')
plt.show()

# Print example Pe_ctx values and their implications
print("Pe_ctx Sweep Examples:")
print("="*50)
for pe in [0.1, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0, 10000.0, 100000.0]:
    w = pe ** (-1/3)
    h = a + (2/3) * np.log(pe)
    accuracy = 1 - 0.4 * w  # Approximate accuracy based on width
    print(f"Pe_ctx = {pe:6.1f} → W/√D_eff = {w:5.3f}, H = {h:5.2f}, ~{accuracy*100:3.0f}% accurate")