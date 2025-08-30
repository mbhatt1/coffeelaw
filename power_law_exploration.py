import numpy as np
import matplotlib.pyplot as plt

# Create figure with multiple power law scenarios
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Exploring Possible Power Laws for Context Chunks', fontsize=16, fontweight='bold')

# Number of chunks
N = np.logspace(0, 2, 100)  # 1 to 100 chunks

# Subplot 1: Different power law exponents for performance
ax1 = axes[0, 0]
for beta in [0.1, 0.2, 0.33, 0.5, 0.67]:
    performance = N ** beta
    ax1.loglog(N, performance, linewidth=2.5, label=f'Performance ∝ N^{beta:.2f}')

ax1.set_xlabel('Number of Chunks (N)', fontsize=12)
ax1.set_ylabel('Performance', fontsize=12)
ax1.set_title('Performance vs Chunks: Different Power Laws', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3, which='both')

# Subplot 2: Marginal benefit (diminishing returns)
ax2 = axes[0, 1]
for gamma in [0.33, 0.5, 0.67, 1.0]:
    marginal_benefit = N ** (-gamma)
    ax2.loglog(N, marginal_benefit, linewidth=2.5, label=f'Benefit ∝ N^{-gamma:.2f}')

ax2.set_xlabel('Number of Chunks (N)', fontsize=12)
ax2.set_ylabel('Marginal Benefit per Chunk', fontsize=12)
ax2.set_title('Diminishing Returns: Marginal Benefit', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

# Subplot 3: What the "failed" law was trying to measure
ax3 = axes[1, 0]
# α(N) as defined in the code: W_normalized / N_eff^(1/3)
# If W ∝ Pe^(-1/3) and we assume Pe is roughly constant per chunk
for scenario in ['constant_pe', 'decreasing_pe', 'increasing_pe']:
    if scenario == 'constant_pe':
        alpha = 1.0 / (N ** (1/3))
        label = 'α = const/N^(1/3) (code definition)'
        style = '-'
    elif scenario == 'decreasing_pe':
        # Pe decreases with more chunks (dilution)
        Pe_per_chunk = 1.0 / N ** 0.2
        W = Pe_per_chunk ** (-1/3)
        alpha = W / (N ** (1/3))
        label = 'α with decreasing Pe/chunk'
        style = '--'
    else:
        # Pe increases with more chunks (reinforcement)
        Pe_per_chunk = N ** 0.1
        W = Pe_per_chunk ** (-1/3)
        alpha = W / (N ** (1/3))
        label = 'α with increasing Pe/chunk'
        style = ':'
    
    ax3.loglog(N, alpha, linewidth=2.5, label=label, linestyle=style)

ax3.set_xlabel('Number of Chunks (N)', fontsize=12)
ax3.set_ylabel('α(N) - Coupling Parameter', fontsize=12)
ax3.set_title('What the Code Actually Measures', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# Subplot 4: Reality check - what might actually happen
ax4 = axes[1, 1]

# Scenario 1: Simple diminishing returns
simple_performance = 1 - np.exp(-N/10)  # Asymptotic to 1
ax4.semilogx(N, simple_performance, 'b-', linewidth=2.5, label='Exponential saturation')

# Scenario 2: Power law with saturation
power_saturation = N**0.3 / (1 + N**0.3/10)
ax4.semilogx(N, power_saturation, 'r--', linewidth=2.5, label='Power law + saturation')

# Scenario 3: Logarithmic growth
log_growth = np.log(1 + N) / np.log(101)
ax4.semilogx(N, log_growth, 'g:', linewidth=2.5, label='Logarithmic growth')

ax4.set_xlabel('Number of Chunks (N)', fontsize=12)
ax4.set_ylabel('Normalized Performance', fontsize=12)
ax4.set_title('Realistic Performance Models', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('power_law_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

# Print analysis
print("\nPOWER LAW ANALYSIS:")
print("="*50)
print("\n1. IF there's a power law for chunks, it could be:")
print("   - Performance ∝ N^β where 0 < β < 1")
print("   - Marginal benefit ∝ N^(-γ) where γ > 0")
print("   - But NOT the circular α(N) ∝ N^(-1/3) as currently measured")

print("\n2. More likely scenarios:")
print("   - Exponential saturation: P = 1 - e^(-N/N₀)")
print("   - Logarithmic: P ∝ log(N)")
print("   - Power law with ceiling: P = N^β / (1 + N^β/P_max)")

print("\n3. The key insight:")
print("   - Coffee Laws 1 & 2 relate Pe_ctx to performance")
print("   - But Pe_ctx itself might not scale simply with N")
print("   - Need to measure: Pe_ctx(N) first, then apply Laws 1 & 2")

print("\n4. Why the -1/3 exponent might appear:")
print("   - It's the same exponent from Law 1")
print("   - Suggests deep connection to information geometry")
print("   - But current measurement method obscures this")