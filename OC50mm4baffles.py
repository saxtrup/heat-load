import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings("ignore")

# ========================== EXACT GEOMETRY ==========================
L_total       = 1.137   # m   total stick length (flange → cold tip)
L_cold        = 0.142   # m   fully cold section (up to top of VTI)
n_baffles     = 4
T_cold        = 1.8
T_cold_cycle  = 40.0
T_wall        = 300.0
T_ref         = 293.0
cryostat_bore_mm = 50.0
eps_base      = 0.04
eps_inner     = 0.03
sigma         = 5.67e-8
E             = 193e9
F_side        = 3.0     # N side-load
baseline_idx  = 4

def k(T):
    T_tab = np.array([1.8, 4, 10, 20, 50, 100, 200, 300])
    k_tab = np.array([0.15, 0.22, 0.45, 1.1, 4.2, 9.0, 13.0, 15.3])
    return np.interp(T, T_tab, k_tab)

def get_alpha(T):
    T_tab = np.array([1.8, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                      80.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    alpha_tab = np.array([0.20, 0.30, 1.00, 3.00, 4.80, 6.30, 7.60, 8.60,
                          10.20, 11.40, 13.50, 14.90, 15.90, 16.50]) * 1e-6
    return np.interp(T, T_tab, alpha_tab, left=0.20e-6, right=16.50e-6)

T_grid = np.linspace(1.8, 300.0, 3000)
alpha_grid = get_alpha(T_grid)
delta_grid = cumulative_trapezoid(alpha_grid, T_grid, initial=0.0)
delta_grid -= np.interp(T_ref, T_grid, delta_grid)
def thermal_strain(T):
    return np.interp(T, T_grid, delta_grid)

r_cryo = cryostat_bore_mm / 2000.0
eps_eff = eps_base / (n_baffles + 1)

# ========================== ODE ==========================
def ode(t, y, eps_eff, A, r_inner, T_cold_local):
    T = np.clip(y[0], 1.0, 320.0)
    Q = y[1]
    if t <= L_cold:
        return [0.0, 0.0]
    rad_outer = 2 * np.pi * r_cryo * sigma * eps_eff * max(0.0, T_wall**4 - T**4)
    rad_inner = 2 * np.pi * r_inner * sigma * eps_inner * max(0.0, T_wall**4 - T**4)
    return [Q / (k(T) * A), -(rad_outer + rad_inner)]

def shoot(Q_guess, eps_eff, A, r_inner, T_cold_local):
    sol = solve_ivp(ode, [0, L_total], [T_cold_local, Q_guess],
                    args=(eps_eff, A, r_inner, T_cold_local),
                    method='LSODA', atol=1e-10, rtol=1e-8, dense_output=True)
    return sol.sol(L_total)[0] - T_wall

# ========================== NEUTRAL CONFIGURATIONS ==========================
stick_data = [
    {"od_mm": 12.7, "wall_mm": 0.50, "label": "12.7 mm (½\") / 11.7 mm ID × 0.50 mm", "note": "UK thin-wall"},
    {"od_mm": 12.7, "wall_mm": 0.89, "label": "12.7 mm (½\") / 10.92 mm ID × 0.89 mm", "note": "UK standard"},
    {"od_mm": 12.7, "wall_mm": 1.00, "label": "12.7 mm (½\") / 10.7 mm ID × 1.00 mm",  "note": "UK thick-wall"},
    {"od_mm": 14.0, "wall_mm": 0.50, "label": "14.0 mm / 13.0 mm ID × 0.50 mm",      "note": "14 mm thin"},
    {"od_mm": 14.0, "wall_mm": 1.00, "label": "14.0 mm / 12.0 mm ID × 1.00 mm",      "note": "14 mm standard"},
    {"od_mm": 16.0, "wall_mm": 0.50, "label": "16.0 mm / 15.0 mm ID × 0.50 mm",      "note": "16 mm thin"},
    {"od_mm": 16.0, "wall_mm": 1.00, "label": "16.0 mm / 14.0 mm ID × 1.00 mm",      "note": "16 mm thick"},
]

# ========================== CALCULATION ==========================
print("Computing with exact geometry (1137 mm stick, 142 mm cold section)...\n")

T_profiles_1p8 = []
Q_list = []
I_list = []
defl_side_list = []
shrinkage_1p8_list = []
shrinkage_span_list = []
short_labels = []

for i, cfg in enumerate(stick_data):
    print(f"[{i+1}/7] {cfg['label']}", end=" ... ")
    
    OD = cfg["od_mm"] / 1000.0
    wall = cfg["wall_mm"] / 1000.0
    ID = OD - 2 * wall
    A = np.pi * ((OD/2)**2 - (ID/2)**2)
    r_inner = ID / 2

    res = root_scalar(lambda q: shoot(q, eps_eff, A, r_inner, T_cold), bracket=[0.01, 10.0], xtol=1e-10)
    Q = res.root
    Q_list.append(Q)

    I = (np.pi / 64.0) * (OD**4 - ID**4)
    I_list.append(I)
    delta_mm = (F_side * L_total**3) / (3 * E * I) * 1000
    defl_side_list.append(delta_mm)

    x_dense = np.linspace(0, L_total, 1000)
    sol = solve_ivp(ode, [0, L_total], [T_cold, Q], args=(eps_eff, A, r_inner, T_cold),
                    method='LSODA', atol=1e-10, rtol=1e-8, dense_output=True)
    T_prof = sol.sol(x_dense)[0]
    T_profiles_1p8.append(T_prof)

    shrinkage_1p8 = np.trapz(thermal_strain(T_prof), x_dense) * 1000
    shrinkage_1p8_list.append(shrinkage_1p8)

    res40 = root_scalar(lambda q: shoot(q, eps_eff, A, r_inner, T_cold_cycle), bracket=[0.01, 10.0], xtol=1e-10)
    Q40 = res40.root
    sol40 = solve_ivp(ode, [0, L_total], [T_cold_cycle, Q40], args=(eps_eff, A, r_inner, T_cold_cycle),
                      method='LSODA', atol=1e-10, rtol=1e-8, dense_output=True)
    T_prof40 = sol40.sol(x_dense)[0]
    shrinkage_40 = np.trapz(thermal_strain(T_prof40), x_dense) * 1000

    span_mm = abs(shrinkage_1p8 - shrinkage_40)
    shrinkage_span_list.append(span_mm)

    short_labels.append(f"{cfg['od_mm']}/{ID*1000:.1f}")
    print(f"done  Q={Q*1000:5.1f} mW  span={span_mm:6.3f} mm")

print("\nAll calculations finished.\n")

# ========================== SMART RECOMMENDATION ==========================
# Only consider sticks with Good stiffness or better (defl. ≤ 14 mm)
good_candidates = [i for i in range(len(Q_list)) if defl_side_list[i] <= 14.0]
best_idx = min(good_candidates, key=lambda i: Q_list[i]) if good_candidates else np.argmin(Q_list)

# ========================== PDF OUTPUT ==========================
pdf_filename = "VTI_50mm_4baffles_1137mm_FINAL.pdf"

with PdfPages(pdf_filename) as pdf:
    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.0, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    colors = plt.cm.plasma(np.linspace(0.95, 0.05, 7))
    x = np.linspace(0, L_total, 1000)

    for i in range(7):
        ax1.plot(x, T_profiles_1p8[i], color=colors[i], lw=2.2, label=stick_data[i]["label"])

    # Baffles
    baffle_positions = [L_total - 0.400, L_total - 0.550, L_total - 0.700, L_total - 0.850]
    for i, xb in enumerate(baffle_positions):
        ax1.axvline(xb, color='darkgray', linestyle=':', lw=1.5, alpha=0.85, label='Baffle midpoints' if i == 0 else None)
    # VTI top
    ax1.axvline(L_cold, color='black', linestyle='--', lw=1.2, alpha=0.85, label='VTI top / cold section end')

    ax1.set_title('Temperature Profiles along SE Sticks\n(185 mm cold tip + 142 mm cold section + 4 baffles @ 150 mm, total 1137 mm)', fontsize=14)
    ax1.set_ylabel('Temperature (K)')
    ax1.set_ylim(0, 310)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8.5, loc='upper left')

    ax2.bar(range(7), [q*1000 for q in Q_list], color=colors, width=0.65)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(short_labels, rotation=45, ha='right')
    ax2.set_ylabel('Heat load at 1.8 K (mW)')
    ax2.set_title('Parasitic Heat Load (lower = better)')
    ax2.grid(True, axis='y', alpha=0.3)

    diff_shrink = np.array(shrinkage_1p8_list) - shrinkage_1p8_list[baseline_idx]
    ax3.bar(range(7), diff_shrink, color=colors, width=0.65)
    ax3.set_xticks(range(7))
    ax3.set_xticklabels(short_labels, rotation=45, ha='right')
    ax3.set_ylabel('Δ Shrinkage vs Baseline (mm)')
    ax3.set_title('Length Shrinkage Difference at 1.8 K')
    ax3.axhline(0, color='gray', lw=1)
    ax3.grid(True, axis='y', alpha=0.3)

    ax4.bar(range(7), shrinkage_span_list, color=colors, width=0.65)
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(short_labels, rotation=45, ha='right')
    ax4.set_ylabel('Contraction Span (mm)')
    ax4.set_title('Length Variation Span when Cold End Cycles 1.8 K ↔ 40 K')
    ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)

    # ====================== PAGE 2 ==========================
    fig2 = plt.figure(figsize=(13.5, 9))
    ax = fig2.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.axis('off')

    text = "=== 50 mm Cryostat — 4 Baffles @ 150 mm (3 N side-load) ===\n"
    text += "Legend: Deflection → Excellent (<7 mm) | Very Good (7–10) | Good (10–14) | Acceptable (14–19) | Borderline (19–26) | Risky (>26 mm)\n\n"
    text += "OD/ID     Wall   Heat Load   Defl. (3N)   Rel. Stiff.   Contraction   ΔShrink   Span (1.8↔40K)   Rating          Description\n"
    text += "-" * 140 + "\n"

    for i, cfg in enumerate(stick_data):
        OD = cfg["od_mm"]
        wall = cfg["wall_mm"]
        ID = OD - 2 * wall
        Q_mW = Q_list[i] * 1000
        delta = defl_side_list[i]
        rel_stiff = 100.0 if i == baseline_idx else (I_list[i] / I_list[baseline_idx]) * 100
        shrink = shrinkage_1p8_list[i]
        dshrink = shrink - shrinkage_1p8_list[baseline_idx]
        span = shrinkage_span_list[i]

        if delta < 7.0: rating = "Excellent"
        elif delta < 10: rating = "Very Good"
        elif delta < 14: rating = "Good"
        elif delta < 19: rating = "Acceptable"
        elif delta < 26: rating = "Borderline"
        else: rating = "Risky"

        text += f"{OD:4.1f}/{ID:4.1f}  {wall:5.2f}   {Q_mW:6.1f} mW   {delta:5.1f} mm    {rel_stiff:5.0f}%     "
        text += f"{-shrink:8.3f} mm   {dshrink:+8.3f} mm   {span:7.3f} mm      {rating:12}   {cfg['note']}\n"

    text += f"\nRECOMMENDED: {stick_data[best_idx]['label']}\n"
    text += f"   (best balance: low heat load + Good stiffness)\n"

    text += "\n==================================================================================\n"
    text += "GEOMETRY: Cryostat bottom to flange = 1322 mm. Stick total length = 1137 mm.\n"
    text += "Cold tip at 185 mm from bottom. Cold section (up to VTI top) = 142 mm.\n"
    text += "4 baffles at 150 mm spacing (midpoints: 400 / 550 / 700 / 850 mm from flange).\n"
    text += "==================================================================================\n"

    ax.text(0, 1, text, va='top', ha='left', fontsize=10.5, family='monospace')
    pdf.savefig(fig2)
    plt.close(fig2)

print(f"PDF saved: {pdf_filename}")
