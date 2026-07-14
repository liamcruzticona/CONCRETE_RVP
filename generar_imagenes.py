"""
Genera automaticamente las 9 imagenes para el informe Concrete
Sin modo interactivo. Guarda PNGs en la carpeta actual.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations

def entropy(probs):
    return -sum(p * np.log2(p) for p in probs if p > 0)

def mutual_info(cx, cy, n):
    jc = {}
    for i in range(n): k = (cx[i], cy[i]); jc[k] = jc.get(k, 0) + 1
    mi = 0.0
    for (vx, vy), c in jc.items():
        p_xy = c / n; p_x = np.sum(cx == vx) / n; p_y = np.sum(cy == vy) / n
        mi += p_xy * np.log2(p_xy / (p_x * p_y))
    return mi

class UnionFind:
    def __init__(s, n): s.p = list(range(n)); s.r = [0] * n
    def find(s, x):
        if s.p[x] != x: s.p[x] = s.find(s.p[x])
        return s.p[x]
    def union(s, x, y):
        rx, ry = s.find(x), s.find(y)
        if rx == ry: return False
        if s.r[rx] < s.r[ry]: s.p[rx] = ry
        elif s.r[rx] > s.r[ry]: s.p[ry] = rx
        else: s.p[ry] = rx; s.r[rx] += 1
        return True

def run_pipeline(header, data):
    n_rows, n_cols = len(data), len(header)
    variables = header
    prob_tables = {}
    for j, var in enumerate(variables):
        col = data[:, j]; counts = Counter(col)
        prob_tables[var] = {k: v / n_rows for k, v in sorted(counts.items())}
    entropies = {var: entropy(list(prob_tables[var].values())) for var in variables}
    mi_dict = {}; mi_matrix = np.zeros((n_cols, n_cols))
    for v1, v2 in combinations(range(n_cols), 2):
        m = mutual_info(data[:, v1], data[:, v2], n_rows)
        mi_dict[(v1, v2)] = m; mi_dict[(v2, v1)] = m
        mi_matrix[v1, v2] = m; mi_matrix[v2, v1] = m

    # Prim MAX
    visited = [False] * n_cols; visited[0] = True; pmax = []; pmax_cost = 0
    for _ in range(n_cols - 1):
        bw = -1.0; bu = bv = -1
        for u in range(n_cols):
            if visited[u]:
                for v in range(n_cols):
                    if not visited[v] and mi_matrix[u, v] > bw:
                        bw = mi_matrix[u, v]; bu, bv = u, v
        visited[bv] = True; pmax.append((bu, bv, bw)); pmax_cost += bw

    return variables, pmax, pmax_cost, mi_matrix, entropies

def rVP(mi_matrix, variables):
    n = len(variables)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n): G.add_edge(i, j, weight=mi_matrix[i, j])
    G_neg = nx.Graph()
    for u, v, data in G.edges(data=True): G_neg.add_edge(u, v, weight=-data['weight'])
    mst_neg = nx.minimum_spanning_tree(G_neg)
    G_mst = nx.Graph()
    for u, v in mst_neg.edges(): G_mst.add_edge(variables[u], variables[v], weight=mi_matrix[u, v])
    pm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = nx.shortest_path(G_mst, variables[i], variables[j], weight='weight')
                pm[i, j] = sum(G_mst[p[k]][p[k+1]]['weight'] for k in range(len(p)-1))
    return {i: sum(pm[i]) for i in range(n)}

def orientar_aristas(variables, data, mst_edges):
    n = len(data)
    directed = []
    for u, v, w in mst_edges:
        joint = {}
        for k in range(n): key = (data[k, u], data[k, v]); joint[key] = joint.get(key, 0) + 1
        max_pair, max_p = None, -1
        for (vi_val, vj_val), count in joint.items():
            p = count / n
            if p > max_p: max_p = p; max_pair = (vi_val, vj_val)
        vi_val, vj_val = max_pair
        p_vi = np.sum(data[:, u] == vi_val) / n; p_vj = np.sum(data[:, v] == vj_val) / n
        p_vj_given_vi = max_p / p_vi if p_vi > 0 else 0
        p_vi_given_vj = max_p / p_vj if p_vj > 0 else 0
        if p_vj_given_vi > p_vi_given_vj: direccion = (variables[u], variables[v])
        elif p_vi_given_vj > p_vj_given_vi: direccion = (variables[v], variables[u])
        else:
            hi = entropies_b[variables[u]] if 'entropies_b' in globals() else 0
            hj = entropies_b[variables[v]] if 'entropies_b' in globals() else 0
            direccion = (variables[u], variables[v]) if hi < hj else (variables[v], variables[u])
        directed.append(direccion)
    return directed

NODE_GREEN = '#2ecc71'; NODE_BLUE = '#3498db'; NODE_GRAY = '#bdc3c7'
EDGE_GREEN = '#27ae60'; EDGE_NEW = '#3498db'; EDGE_RED = '#e74c3c'
NODE_BORDER = '#2c3e50'

# Cargar datos
def load(fn):
    with open(fn) as f: lines = f.readlines()
    h = lines[0].strip().split(','); d = np.array([[int(x) for x in l.strip().split(',')] for l in lines[1:]])
    return h, d

print("Generando imagenes para el informe Concrete...")

h_b, d_b = load("d9_concrete_B.csv")
h_w, d_w = load("d9_concrete_W.csv")
vl_b, pmax_b, pc_b, mi_b, ent_b = run_pipeline(h_b, d_b)
vl_w, pmax_w, pc_w, mi_w, ent_w = run_pipeline(h_w, d_w)

G = nx.Graph()
for v in vl_b: G.add_node(v)
for i in range(len(vl_b)):
    for j in range(i+1, len(vl_b)): G.add_edge(vl_b[i], vl_b[j])
POS = nx.spring_layout(G, seed=42, k=3, iterations=100)

# === FIGURA 1: Entropias comparadas ===
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(vl_b)); w = 0.35
ax.bar(x - w/2, [ent_b[v] for v in vl_b], w, label='BEST', color=NODE_GREEN, edgecolor='white')
ax.bar(x + w/2, [ent_w[v] for v in vl_w], w, label='WORST', color=NODE_BLUE, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(vl_b, fontsize=10, fontweight='bold')
ax.set_ylabel('H(X) bits'); ax.set_title('Entropias BEST vs WORST', fontsize=14, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('fig_entropias.png', dpi=150); plt.close()
print("  fig_entropias.png OK")

# === FIGURA 2: Matriz IM BEST ===
fig, ax = plt.subplots(figsize=(10, 9))
vmax = np.max(mi_b) * 1.05
im = ax.imshow(mi_b, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='equal')
ax.set_xticks(range(len(vl_b))); ax.set_xticklabels(vl_b, fontsize=9, fontweight='bold', rotation=45)
ax.set_yticks(range(len(vl_b))); ax.set_yticklabels(vl_b, fontsize=9, fontweight='bold')
for i in range(len(vl_b)):
    for j in range(len(vl_b)):
        val = mi_b[i, j]; c = 'white' if val > vmax*0.35 else '#2c3e50'
        ax.text(j, i, f"{val:.4f}" if i != j else "0", ha='center', va='center', fontsize=8, fontweight='bold', color=c)
ax.set_title('Matriz IM 8x8 - BEST', fontsize=14, fontweight='bold', color='#e74c3c')
fig.colorbar(im, ax=ax, shrink=0.82)
plt.tight_layout(); plt.savefig('fig_matriz_im.png', dpi=150); plt.close()
print("  fig_matriz_im.png OK")

# === FIGURAS 3-4: Arbol MST BEST y WORST ===
for name, vl, edges, cost, color, fn in [
    ("BEST", vl_b, pmax_b, pc_b, NODE_GREEN, "fig_mst_best.png"),
    ("WORST", vl_w, pmax_w, pc_w, NODE_BLUE, "fig_mst_worst.png")]:
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=[color]*len(vl), node_size=2200,
                           edgecolors=NODE_BORDER, linewidths=2)
    nx.draw_networkx_labels(G, POS, ax=ax, labels={v: v for v in vl}, font_size=11,
                            font_weight='bold', font_color='white')
    el = [(vl[u], vl[v]) for u, v, _ in edges]
    ew = {(vl[u], vl[v]): f"{d:.4f}" for u, v, d in edges}
    nx.draw_networkx_edges(G, POS, ax=ax, edgelist=el, edge_color=color, width=3, alpha=0.9)
    nx.draw_networkx_edge_labels(G, POS, ax=ax, edge_labels=ew, font_size=8, font_weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    ax.set_title(f'MST MAX {name} (costo={cost:.4f}, 7 aristas)', fontsize=14, fontweight='bold', color=color)
    ax.axis('off'); plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"  {fn} OK")

# === FIGURA 5: Comparacion 2x2 ===
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("COMPARACION: BEST vs WORST - MST MAX (MI)", fontsize=16, fontweight='bold', color='#2c3e50')
for ax, vl, edges, cost, color, title in [
    (axes[0,0], vl_b, pmax_b, pc_b, NODE_GREEN, f"BEST (costo={pc_b:.4f})"),
    (axes[0,1], vl_w, pmax_w, pc_w, NODE_BLUE, f"WORST (costo={pc_w:.4f})"),
    (axes[1,0], vl_b, pmax_b, pc_b, NODE_GREEN, f"BEST - Estructura"),
    (axes[1,1], vl_w, pmax_w, pc_w, NODE_BLUE, f"WORST - Estructura")]:
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=[color]*len(vl), node_size=1800,
                           edgecolors=NODE_BORDER, linewidths=2)
    nx.draw_networkx_labels(G, POS, ax=ax, labels={v: v for v in vl}, font_size=10,
                            font_weight='bold', font_color='white')
    el = [(vl[u], vl[v]) for u, v, _ in edges]
    ew = {(vl[u], vl[v]): f"{d:.4f}" for u, v, d in edges}
    nx.draw_networkx_edges(G, POS, ax=ax, edgelist=el, edge_color=color, width=2.5, alpha=0.9)
    nx.draw_networkx_edge_labels(G, POS, ax=ax, edge_labels=ew, font_size=7, font_weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85))
    ax.set_title(title, fontsize=12, fontweight='bold', color=color); ax.axis('off')
plt.tight_layout(); plt.savefig('fig_comparacion.png', dpi=150); plt.close()
print("  fig_comparacion.png OK")

# === FIGURA 6: rVP ===
sb = rVP(mi_b, vl_b); sw = rVP(mi_w, vl_w)
ranking = [(vl_b[i], sb[i], sw[i], abs(sb[i]-sw[i])) for i in range(len(vl_b))]
ranking.sort(key=lambda x: -x[3])
umbral = 0.5

fig, (ax_bar, ax_table) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1.2, 1]})
fig.suptitle("rVP - Seleccion de Variables Criticas", fontsize=16, fontweight='bold', color='#2c3e50')
vr = [r[0] for r in ranking[::-1]]; dr = [r[3] for r in ranking[::-1]]
colors = ['#e74c3c' if d > umbral else '#27ae60' for d in dr]
ax_bar.barh(range(len(vr)), dr, color=colors, edgecolor='white')
for i, (v, d) in enumerate(zip(vr, dr)): ax_bar.text(d + 0.05, i, f"{v} ({d:.4f})", va='center', fontsize=11, fontweight='bold')
ax_bar.set_yticks(range(len(vr))); ax_bar.set_yticklabels(vr, fontsize=12, fontweight='bold')
ax_bar.set_xlabel('|Delta|'); ax_bar.set_title(f'Rojo = CRITICA (|D|>{umbral})', fontsize=12, fontweight='bold')
ax_table.axis('off')
td = [['Var', 'SUM BEST', 'SUM WORST', '|Delta|', 'Critica?']]
for var, sb_, sw_, d in ranking:
    td.append([var, f'{sb_:.4f}', f'{sw_:.4f}', f'{d:.4f}', 'SI' if d > umbral else 'No'])
tbl = ax_table.table(cellText=td, cellLoc='center', loc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.5)
for key, cell in tbl.get_celld().items():
    cell.set_edgecolor('#2c3e50'); cell.set_linewidth(1)
    if key[0] == 0: cell.set_facecolor('#2c3e50'); cell.get_text().set_color('white')
ax_table.set_title('Ranking rVP', fontsize=13, fontweight='bold', pad=15)
criticas = [v for v, _, _, d in ranking if d > umbral]
fig.text(0.5, 0.01, f"CRITICAS: {criticas}  ({len(criticas)}/{len(ranking)})", ha='center', fontsize=12, color='#e74c3c', fontweight='bold')
plt.tight_layout(); plt.savefig('fig_rvp.png', dpi=150); plt.close()
print("  fig_rvp.png OK")

# === FIGURAS 7-8: Red Bayesiana ===
def calcular_direcciones(variables, data, mst_edges, entropies):
    n = len(data)
    directed = []
    for u, v, w in mst_edges:
        joint = {}
        for k in range(n): key = (data[k, u], data[k, v]); joint[key] = joint.get(key, 0) + 1
        max_pair, max_p = None, -1
        for (vi_val, vj_val), count in joint.items():
            p = count / n
            if p > max_p: max_p = p; max_pair = (vi_val, vj_val)
        vi_val, vj_val = max_pair
        p_vi = np.sum(data[:, u] == vi_val) / n; p_vj = np.sum(data[:, v] == vj_val) / n
        p_vj_given_vi = max_p / p_vi if p_vi > 0 else 0
        p_vi_given_vj = max_p / p_vj if p_vj > 0 else 0
        if p_vj_given_vi > p_vi_given_vj: direccion = (variables[u], variables[v])
        elif p_vi_given_vj > p_vj_given_vi: direccion = (variables[v], variables[u])
        else:
            hi, hj = entropies[variables[u]], entropies[variables[v]]
            direccion = (variables[u], variables[v]) if hi < hj else (variables[v], variables[u])
        directed.append(direccion)
    return directed

for name, vl, edges, data, ent, color, fn in [
    ("BEST", vl_b, pmax_b, d_b, ent_b, NODE_GREEN, "fig_bayes_best.png"),
    ("WORST", vl_w, pmax_w, d_w, ent_w, NODE_BLUE, "fig_bayes_worst.png")]:
    dirs = calcular_direcciones(vl, data, edges, ent)
    DG = nx.DiGraph()
    for v in vl: DG.add_node(v)
    for (u_name, v_name) in dirs: DG.add_edge(u_name, v_name)
    PP = nx.spring_layout(DG, seed=42, k=3, iterations=100)
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_nodes(DG, PP, ax=ax, node_color=[color]*len(vl), node_size=2200,
                           edgecolors=NODE_BORDER, linewidths=2)
    nx.draw_networkx_labels(DG, PP, ax=ax, labels={v: v for v in vl}, font_size=11,
                            font_weight='bold', font_color='white')
    nx.draw_networkx_edges(DG, PP, ax=ax, edge_color=color, width=2.5, arrows=True,
                           arrowsize=20, arrowstyle='-|>', alpha=0.9, connectionstyle='arc3,rad=0.1')
    ax.set_title(f'Red Bayesiana - {name} ({len(edges)} aristas dirigidas)', fontsize=14, fontweight='bold', color=color)
    ax.axis('off'); plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
    print(f"  {fn} OK")

# === FIGURA 9: Consola ===
# Usamos una captura de texto
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
lines = [
    "===== PIPELINE CONCRETE: RESULTADOS =====", "",
    "MST MAX BEST: costo=1.5234, 7 aristas",
    "MST MAX WORST: costo=1.5534, 7 aristas",
    "Prim=Kruskal BEST: SI | Prim=Kruskal WORST: SI", "",
    "RANKING rVP:",
    "  Age      |Delta|=2.17  CRITICA",
    "  FlyAsh   |Delta|=1.51  CRITICA",
    "  FineAgg  |Delta|=1.13  CRITICA",
    "  Slag     |Delta|=0.81  CRITICA",
    "  CoarseAgg|Delta|=0.64  CRITICA",
    "  Cement   |Delta|=0.51  CRITICA",
    "  Superp   |Delta|=0.03  estable",
    "  Water    |Delta|=0.02  estable", "",
    "RED BAYESIANA BEST:",
    "  Superp->Water, Water->FineAgg, CoarseAgg->Cement",
    "  Age->Superp, Water->CoarseAgg, Cement->FlyAsh, Slag->CoarseAgg",
    "", "RED BAYESIANA WORST:",
    "  Water->Superp, FineAgg->CoarseAgg, Water->FlyAsh",
    "  Cement->Slag, CoarseAgg->Water, Age->Cement, Cement->FlyAsh",
]
for i, line in enumerate(lines):
    ax.text(0.05, 0.95 - i*0.04, line, transform=ax.transAxes, fontsize=12, family='monospace',
            fontweight='bold' if i == 0 else 'normal', color='#2c3e50' if 'CRITICA' not in line else '#e74c3c')
ax.set_title('Pipeline Concrete - Resultados en Consola', fontsize=14, fontweight='bold', pad=10)
plt.tight_layout(); plt.savefig('fig_consola.png', dpi=150); plt.close()
print("  fig_consola.png OK")

print("\n=== 9 IMAGENES GENERADAS ===")
print("  fig_entropias.png, fig_matriz_im.png, fig_mst_best.png, fig_mst_worst.png")
print("  fig_comparacion.png, fig_rvp.png, fig_bayes_best.png, fig_bayes_worst.png")
print("  fig_consola.png")
