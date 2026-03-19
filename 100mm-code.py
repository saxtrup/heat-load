"""
VTI Spacer Model - FIXED VERSION (100 mm bore)
- Fixed temperature axis (now 0–310 K, no negative nonsense)
- Much more stable solver for large bores
- Everything else exactly as before
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# ====================== PARAMETERS ======================
L_total = 1.10
L_vti = 0.30
T_cold = 1.8
T_wall = 300.0
OD_stick = 0.014
ID_stick = 0.012
A = np.pi * ((OD_stick/2)**2 - ((ID_stick/2)**2))
r_inner = ID_stick / 2

# ====================== USER SETTINGS ======================
distance_from_vti = 0.04
baffle_spacing = 0.140
n_list = [0, 4, 6, 8, 9, 10, 12]
cryostat_bore_mm = 100.0                  # ← change anytime (50, 70, 100...)
# =======================================================

r_cryo = cryostat_bore_mm / 2000.0
baffle_start = L_vti + distance_from_vti
eps_base = 0.04
eps_inner = 0.03
sigma = 5.67e-8

T_k_table = np.array([1.8, 4, 10, 20, 50, 100, 200, 300])
k_table = np.array([0.15, 0.22, 0.45, 1.1, 4.2, 9.0, 13.0, 15.3])
n_points = 800

def k(T):
    return np.interp(T, T_k_table, k_table)

def q_rad_outer(x, T, eps_eff):
    if x <= L_vti:
        return 0.0
    return 2 * np.pi * r_cryo * sigma * eps_eff * (T_wall**4 - T**4)

def q_rad_inner(x, T):
    return 2 * np.pi * r_inner * sigma * eps_inner * (T_wall**4 - T**4)

def ode(t, y, eps_eff):
    T = min(max(y[0], 1.0), 320.0)
    Q = y[1]
    if t <= L_vti:
        return [0.0, 0.0]
    dT_dx = Q / (k(T) * A)
    dQ_dx = -q_rad_outer(t, T, eps_eff) - q_rad_inner(t, T)
    return [dT_dx, dQ_dx]

def shoot(Q_guess, eps_eff):
    sol = solve_ivp(ode, [0, L_total], [T_cold, Q_guess], args=(eps_eff,),
                    method='LSODA', atol=1e-10, rtol=1e-10, dense_output=True)
    return sol.sol(L_total)[0] - T_wall

# ====================== COMPUTE ======================
T_profiles = []
Q_list = []
baffle_pos = np.array([baffle_start + i * baffle_spacing for i in range(max(n_list)+2)])
baffle_pos = baffle_pos[baffle_pos <= L_total]

for n in n_list:
    eps_eff = eps_base / (n + 1)
    bracket_max = 20.0 if n == 0 else 5.0
    res = root_scalar(lambda q: shoot(q, eps_eff), bracket=[0.001, bracket_max], maxiter=100)
    Q = res.root
    Q_list.append(Q)
    
    sol = solve_ivp(ode, [0, L_total], [T_cold, Q], args=(eps_eff,),
                    t_eval=np.linspace(0, L_total, n_points), method='LSODA', atol=1e-10, rtol=1e-10)
    T_profiles.append(sol.y[0])

x = sol.t
T_profiles = np.array(T_profiles)

# ====================== PLOTS ======================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2, 1]})
colors = plt.cm.plasma(np.linspace(0.95, 0.1, len(n_list)))

def spacer_label(n):
    return "No spacers" if n == 0 else f"{n} spacers"

for i, n in enumerate(n_list):
    ax1.plot(x, T_profiles[i], color=colors[i], lw=2.5, label=spacer_label(n))

# Spacer labels
for pos in baffle_pos:
    ax1.axvline(pos, color='green', ls=':', alpha=0.7)
    idx = np.argmin(np.abs(x - pos))
    T_b = T_profiles[-1][idx]
    dist_mm = (pos - L_vti) * 1000
    ax1.text(pos + 0.008, T_b + 12, 
             f'Spacer\n{T_b:.0f} K\n({dist_mm:.0f} mm)', 
             ha='left', va='bottom', fontsize=8.5, color='green')

ax1.axvline(L_vti, color='red', ls='--', lw=2, label='Top of VTI')
ax1.set_ylabel('Temperature [K]')
ax1.set_title(f'Temperature profiles — {int(cryostat_bore_mm)} mm cryostat')
ax1.set_ylim(0, 310)          # ← THIS FIXES THE NEGATIVE AXIS
ax1.grid(True, alpha=0.3)
ax1.legend()

# Bar chart
bars = ax2.bar(n_list, [q*1000 for q in Q_list], color=colors, width=0.65)
ax2.set_xlabel('Number of spacers')
ax2.set_ylabel('Heat load at 1.8 K [mW]')
ax2.set_title(f'Parasitic heat load — {int(cryostat_bore_mm)} mm cryostat (lower = better)')
ax2.set_xticks(n_list)
ax2.set_xticklabels([spacer_label(n) for n in n_list])
ax2.grid(True, alpha=0.3)

for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, 
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# ====================== SAVE ======================
pdf_filename = f"VTI_{int(cryostat_bore_mm)}mm_FIXED_{min(n_list)}-{max(n_list)}spacers.pdf"
plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
print(f"✅ Saved clean plot: {pdf_filename}")

plt.show()

# ====================== RESULTS ======================
print(f"\n=== RESULTS for {int(cryostat_bore_mm)} mm cryostat ===")
for n, q in zip(n_list, Q_list):
    label = "No spacers" if n == 0 else f"{n} spacers"
    print(f" {label:12} → {q*1000:.1f} mW")
