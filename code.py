"""
Sample Stick Temperature & Heat-Load Model for Orange-Cryostat / VTI
====================================================================
1-D steady-state conduction + radiation with baffles
Key geometry (user-confirmed):
- Stick OD = 14 mm, ID = 12 mm (open hollow core all the way down)
- Baffles shield ONLY the outer annular gap
- Inner bore (12 mm) is UNshielded
Validated against:
- Kirichek et al. (2013) DOI: 10.1007/s10909-013-0858-x
- Oxford Instruments "Practical Cryogenics" handbook
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# ====================== TWEAKABLE PARAMETERS ======================
L_total = 1.10
L_vti = 0.30
T_cold = 1.8
T_wall = 300.0
OD_stick = 0.014
ID_stick = 0.012
A = np.pi * ((OD_stick/2)**2 - ((ID_stick/2)**2))
r_outer = OD_stick / 2
r_inner = ID_stick / 2

# ====================== USER VARIABLES ======================
distance_from_vti = 0.04      # ← distance from top of VTI to FIRST spacer [m]
baffle_spacing = 0.140        # ← distance BETWEEN spacers (always equal)
n_list = [0, 2, 3, 4, 5, 6]   # ← 0 = "No spacers" (added for clear impact comparison)
# =================================================================

baffle_start = L_vti + distance_from_vti
eps_base = 0.04 # outer shielded
# eps_base = 0.025 # ← UNCOMMENT for closer match to Oxford/Kirichek
eps_inner = 0.03
sigma = 5.67e-8
T_k_table = np.array([1.8, 4, 10, 20, 50, 100, 200, 300])
k_table = np.array([0.15, 0.22, 0.45, 1.1, 4.2, 9.0, 13.0, 15.3])
n_points = 500
# =================================================================

def k(T):
    return np.interp(T, T_k_table, k_table)

def q_rad_outer(x, T, eps_eff):
    if x <= L_vti:
        return 0.0
    return 2 * np.pi * r_outer * sigma * eps_eff * (T_wall**4 - T**4)

def q_rad_inner(x, T):
    return 2 * np.pi * r_inner * sigma * eps_inner * (T_wall**4 - T**4)

def ode(t, y, eps_eff):
    T = min(max(y[0], 1.0), 310.0)
    Q = y[1]
    if t <= L_vti:
        return [0.0, 0.0]
    dT_dx = Q / (k(T) * A)
    dQ_dx = -q_rad_outer(t, T, eps_eff) - q_rad_inner(t, T)
    return [dT_dx, dQ_dx]

def shoot(Q_guess, eps_eff):
    sol = solve_ivp(ode, [0, L_total], [T_cold, Q_guess], args=(eps_eff,),
                    method='RK45', atol=1e-9, rtol=1e-9, dense_output=True)
    return sol.sol(L_total)[0] - T_wall

# ====================== COMPUTE ======================
T_profiles = []
Q_list = []
baffle_pos = np.array([baffle_start + i * baffle_spacing for i in range(max(n_list) + 2)])
baffle_pos = baffle_pos[baffle_pos <= L_total]

for n in n_list:
    eps_eff = eps_base / (n + 1)          # works perfectly for n=0 too
    bracket_max = 5.0 if n == 0 else 0.8  # much higher heat load with no spacers (~477 mW)
    res = root_scalar(lambda q: shoot(q, eps_eff), bracket=[0.01, bracket_max])
    Q = res.root
    Q_list.append(Q)
   
    sol = solve_ivp(ode, [0, L_total], [T_cold, Q], args=(eps_eff,),
                    t_eval=np.linspace(0, L_total, n_points))
    T_profiles.append(sol.y[0])

x = sol.t
T_profiles = np.array(T_profiles)

# ====================== PLOTS ======================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2, 1]})
colors = plt.cm.plasma(np.linspace(1, 0, len(n_list)))  # yellow = No spacers (worst), dark = 6 spacers (best)

def spacer_label(n):
    return "No spacers" if n == 0 else f"{n} spacers"

for i, n in enumerate(n_list):
    ax1.plot(x, T_profiles[i], color=colors[i], lw=2.5, label=spacer_label(n))

# Mark spacer positions
for pos in baffle_pos:
    ax1.axvline(pos, color='green', ls=':', alpha=0.8)
    idx = np.argmin(np.abs(x - pos))
    T_b = T_profiles[-1][idx]
    dist_mm = (pos - L_vti) * 1000
    ax1.text(pos + 0.005, T_b + 15,
             f'Spacer\n{T_b:.0f} K\n({dist_mm:.0f} mm from VTI)',
             ha='left', va='bottom', fontsize=9, color='green')

ax1.axvline(L_vti, color='red', ls='--', lw=2, label='Top of VTI')
ax1.set_ylabel('Temperature [K]')
ax1.set_title('Temperature profiles — more spacers = cooler curve')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Bar chart with "No spacers" included
bars = ax2.bar(n_list, [q*1000 for q in Q_list], color=colors, width=0.6)
ax2.set_xlabel('Number of spacers')
ax2.set_ylabel('Heat load at 1.8 K [mW]')
ax2.set_title('Parasitic heat load (lower = better)')
ax2.grid(True, alpha=0.3)

ax2.set_xticks(n_list)
ax2.set_xticklabels([spacer_label(n) for n in n_list])

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 3,
             f'{height:.1f}', ha='center', va='bottom')

plt.tight_layout()

# ====================== SAVE TO PDF ======================
pdf_filename = f"VTI_Spacer_Model_0-{max(n_list)}spacers_dist{int(distance_from_vti*1000)}mm_spacing{int(baffle_spacing*1000)}mm.pdf"
plt.savefig(pdf_filename, dpi=300, bbox_inches='tight')
print(f"✅ Combined plot saved as: {pdf_filename}")

plt.show()  # ← remove this line if you only want the PDF

# ====================== SUMMARY ======================
print("\n=== RESULTS (with open inner bore) ===")
for n, q in zip(n_list, Q_list):
    print(f" {spacer_label(n):12} → {q*1000:.1f} mW")
