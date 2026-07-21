"""
=====================================================================
  pipeline_rvp.py - Pipeline MAX (MI) + rVP para Concrete dataset
  BEST = 507 registros  |  WORST = 295 registros  |  8 variables
=====================================================================
"""

import numpy as np
import networkx as nx
from collections import Counter
from itertools import combinations

def entropy(probs):
    return -sum(p * np.log2(p) for p in probs if p > 0)

def mutual_info(col_x, col_y, n):
    jc = {}
    for i in range(n): k = (col_x[i], col_y[i]); jc[k] = jc.get(k, 0) + 1
    mi = 0.0
    for (vx, vy), c in jc.items():
        p_xy = c / n; p_x = np.sum(col_x == vx) / n; p_y = np.sum(col_y == vy) / n
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

def load_dataset(fn):
    with open(fn) as f: lines = f.readlines()
    h = lines[0].strip().split(',')
    d = np.array([[int(x) for x in l.strip().split(',')] for l in lines[1:]])
    return h, d

def run_pipeline(header, data, label):
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

    # Prim MAX (mayor MI primero)
    visited = [False] * n_cols; visited[0] = True
    pmax = []; pmax_cost = 0
    for _ in range(n_cols - 1):
        bw = -1.0; bu = bv = -1
        for u in range(n_cols):
            if visited[u]:
                for v in range(n_cols):
                    if not visited[v] and mi_matrix[u, v] > bw:
                        bw = mi_matrix[u, v]; bu, bv = u, v
        visited[bv] = True; pmax.append((bu, bv, bw)); pmax_cost += bw

    # Kruskal MAX
    all_max = [(mi_matrix[i, j], i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    all_max.sort(reverse=True)
    uf = UnionFind(n_cols); kmax = []; kmax_cost = 0
    for w, u, v in all_max:
        if uf.union(u, v):
            kmax.append((u, v, w)); kmax_cost += w
            if len(kmax) == n_cols - 1: break

    return {
        'label': label, 'n_rows': n_rows, 'variables': variables,
        'entropies': entropies, 'mi_matrix': mi_matrix,
        'prim_max': pmax, 'prim_max_cost': pmax_cost,
        'kruskal_max': kmax, 'kruskal_max_cost': kmax_cost,
    }

def rVP(mi_matrix, variables):
    n = len(variables)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=mi_matrix[i, j])
    G_neg = nx.Graph()
    for u, v, data in G.edges(data=True):
        G_neg.add_edge(u, v, weight=-data['weight'])
    mst_neg = nx.minimum_spanning_tree(G_neg)
    G_mst = nx.Graph()
    for u, v in mst_neg.edges():
        G_mst.add_edge(variables[u], variables[v], weight=mi_matrix[u, v])
    path_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                p = nx.shortest_path(G_mst, variables[i], variables[j], weight='weight')
                path_matrix[i, j] = sum(G_mst[p[k]][p[k+1]]['weight'] for k in range(len(p)-1))
    return {i: sum(path_matrix[i]) for i in range(n)}

# =============================================================================
print("\n" + "=" * 70)
print("  PIPELINE CONCRETE - MAX (MI) + rVP")
print("=" * 70)

h_b, d_b = load_dataset("d9_concrete_B.csv")
h_w, d_w = load_dataset("d9_concrete_W.csv")
print(f"\n  BEST:  {len(d_b)} filas x {len(h_b)} variables")
print(f"  WORST: {len(d_w)} filas x {len(h_w)} variables")

# Pipeline
r_b = run_pipeline(h_b, d_b, "BEST")
r_w = run_pipeline(h_w, d_w, "WORST")

print(f"\n  Entropias BEST:  " + ", ".join(f"{v}={r_b['entropies'][v]:.4f}" for v in h_b))
print(f"  Entropias WORST: " + ", ".join(f"{v}={r_w['entropies'][v]:.4f}" for v in h_w))

print(f"\n  MST MAX BEST:  costo={r_b['prim_max_cost']:.4f}, {len(h_b)-1} aristas")
for u, v, w in sorted(r_b['prim_max'], key=lambda x: -x[2]):
    print(f"    {h_b[u]} -- {h_b[v]:<20} MI={w:.6f}")

print(f"\n  MST MAX WORST: costo={r_w['prim_max_cost']:.4f}, {len(h_w)-1} aristas")
for u, v, w in sorted(r_w['prim_max'], key=lambda x: -x[2]):
    print(f"    {h_w[u]} -- {h_w[v]:<20} MI={w:.6f}")

print(f"\n  Prim=Kruskal BEST: {'SI' if abs(r_b['prim_max_cost']-r_b['kruskal_max_cost'])<1e-6 else 'NO'}")
print(f"  Prim=Kruskal WORST: {'SI' if abs(r_w['prim_max_cost']-r_w['kruskal_max_cost'])<1e-6 else 'NO'}")

# rVP
print(f"\n  ETAPA rVP - Seleccion de Variables Criticas")
sum_b = rVP(r_b['mi_matrix'], r_b['variables'])
sum_w = rVP(r_w['mi_matrix'], r_w['variables'])

ranking = []
for i, var in enumerate(r_b['variables']):
    sb = sum_b[i]; sw = sum_w[i]; delta = abs(sb - sw)
    ranking.append((var, sb, sw, delta))

ranking.sort(key=lambda x: -x[3])
umbral = 0.5

print(f"\n  {'Variable':<20} {'SUM BEST':<12} {'SUM WORST':<12} {'|Delta|':<10} {'Critica?'}")
print(f"  {'-'*60}")
for var, sb, sw, d in ranking:
    c = 'SI' if d > umbral else 'No'
    print(f"  {var:<20} {sb:<12.4f} {sw:<12.4f} {d:<10.4f} {c}")

criticas = [(v, f'{d:.4f}') for v, _, _, d in ranking if d > umbral]
print(f"\n  Variables CRITICAS (|Delta| > {umbral}): {criticas}")

# =============================================================================
# ETAPA 5: RED BAYESIANA - Orientar aristas del MST con BIC
# =============================================================================

print("\n\n" + "=" * 70)
print("  ETAPA 5: RED BAYESIANA (BIC)")
print("  Metodo: Probar dos modelos Xi->Xj y Xj->Xi en cada arista MST")
print("  Elegir direccion con MENOR BIC = -2*log(L) + k*log(n)")
print("=" * 70)

d_b_raw, d_w_raw = load_dataset("d9_concrete_B.csv"), load_dataset("d9_concrete_W.csv")

def orientar_bic(header, data, mst_edges, label):
    variables = header
    n = len(data)
    n_cols = len(variables)

    print(f"\n  {label} (n={n}):")
    print(f"  {'Arista':<35} {'BIC(A->B)':<10} {'BIC(B->A)':<10} {'AIC(A->B)':<10} {'AIC(B->A)':<10} {'Direccion'}")
    print(f"  {'-'*86}")

    directed = []
    for u, v, w in sorted(mst_edges, key=lambda x: -x[2]):
        vi_var, vj_var = variables[u], variables[v]

        # Tabla de contingencia para las condicionales
        joint = {}
        for k in range(n):
            key = (data[k, u], data[k, v])
            joint[key] = joint.get(key, 0) + 1

        # Calcular distribuciones condicionales P(Xj | Xi) y P(Xi | Xj)
        vals_i = sorted(set(data[:, u]))
        vals_j = sorted(set(data[:, v]))

        # Modelo A: Xi -> Xj. L_A = prod P(Xj | Xi)
        log_L_A = 0.0
        for (vi, vj), count in joint.items():
            p_cond = count / np.sum(data[:, u] == vi)  # P(Xj=vj | Xi=vi)
            if p_cond > 0:
                log_L_A += count * np.log(p_cond)
        k_A = (len(vals_i) - 1) * len(vals_j)
        BIC_A = -2 * log_L_A + k_A * np.log(n)
        AIC_A = -2 * log_L_A + 2 * k_A

        # Modelo B: Xj -> Xi. L_B = prod P(Xi | Xj)
        log_L_B = 0.0
        for (vi, vj), count in joint.items():
            p_cond = count / np.sum(data[:, v] == vj)  # P(Xi=vi | Xj=vj)
            if p_cond > 0:
                log_L_B += count * np.log(p_cond)
        k_B = (len(vals_j) - 1) * len(vals_i)
        BIC_B = -2 * log_L_B + k_B * np.log(n)
        AIC_B = -2 * log_L_B + 2 * k_B

        # Elegir menor BIC (AIC como respaldo si BIC empata)
        bic_wins_B = BIC_A < BIC_B  # BIC dice A->B
        bic_wins_A = BIC_B < BIC_A  # BIC dice B->A
        aic_wins_B = AIC_A < AIC_B
        aic_wins_A = AIC_B < AIC_A

        if bic_wins_B and aic_wins_B:
            direccion = f"{vi_var} -> {vj_var}  (BIC y AIC)"
        elif bic_wins_A and aic_wins_A:
            direccion = f"{vj_var} -> {vi_var}  (BIC y AIC)"
        elif bic_wins_B:
            direccion = f"{vi_var} -> {vj_var}  (BIC)"
        elif bic_wins_A:
            direccion = f"{vj_var} -> {vi_var}  (BIC)"
        elif aic_wins_B:
            direccion = f"{vi_var} -> {vj_var}  (AIC)"
        elif aic_wins_A:
            direccion = f"{vj_var} -> {vi_var}  (AIC)"
        else:
            hi = entropies_b[vi_var] if 'entropies_b' in dir() else 0
            hj = entropies_b[vj_var] if 'entropies_b' in dir() else 0
            direccion = f"{vi_var} -> {vj_var}" if hi < hj else f"{vj_var} -> {vi_var}"

        directed.append((vi_var, vj_var, direccion))

        print(f"  {vi_var:<6} -- {vj_var:<20} {BIC_A:<10.2f} {BIC_B:<10.2f} {AIC_A:<10.2f} {AIC_B:<10.2f} {direccion}")

    return directed

dir_b = orientar_bic(h_b, d_b, r_b['prim_max'], "BEST")
dir_w = orientar_bic(h_w, d_w, r_w['prim_max'], "WORST")

print(f"\n  RED BAYESIANA BEST:  " + ",  ".join(d for _, _, d in dir_b))
print(f"  RED BAYESIANA WORST: " + ",  ".join(d for _, _, d in dir_w))

print("\n" + "=" * 70)
print("  PIPELINE COMPLETADO")
print("=" * 70)
