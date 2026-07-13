"""
=====================================================================
  visual_concrete.py - Visual MST (MAX) + rVP para Concrete
  8 variables, BEST vs WORST, paso a paso con ENTER
=====================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from itertools import combinations
import time

plt.ion()

NODE_GREEN = '#2ecc71'; NODE_BLUE = '#3498db'; NODE_GRAY = '#bdc3c7'
EDGE_GREEN = '#27ae60'; EDGE_NEW = '#3498db'
EDGE_ORANGE = '#f39c12'; EDGE_RED = '#e74c3c'; NODE_BORDER = '#2c3e50'

def wait_for_enter():
    fig = plt.gcf(); fig.canvas.draw(); fig.canvas.flush_events(); fig.show()
    pressed = []
    def on_key(e):
        if e.key == 'enter': pressed.append(True)
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    t0 = time.time()
    while not pressed and (time.time()-t0)<120: plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid); plt.close(fig)

def entropy(probs):
    return -sum(p*np.log2(p) for p in probs if p>0)

def mutual_info(cx, cy, n):
    jc = {}
    for i in range(n): k=(cx[i],cy[i]); jc[k]=jc.get(k,0)+1
    mi = 0.0
    for (vx,vy),c in jc.items():
        p_xy=c/n; p_x=np.sum(cx==vx)/n; p_y=np.sum(cy==vy)/n
        mi+=p_xy*np.log2(p_xy/(p_x*p_y))
    return mi

class UnionFind:
    def __init__(s,n): s.p=list(range(n)); s.r=[0]*n
    def find(s,x):
        if s.p[x]!=x: s.p[x]=s.find(s.p[x])
        return s.p[x]
    def union(s,x,y):
        rx,ry=s.find(x),s.find(y)
        if rx==ry: return False
        if s.r[rx]<s.r[ry]: s.p[rx]=ry
        elif s.r[rx]>s.r[ry]: s.p[ry]=rx
        else: s.p[ry]=rx; s.r[rx]+=1
        return True

def run_pipeline(header, data):
    n_rows, n_cols = len(data), len(header)
    variables = header
    prob_tables = {}
    for j, var in enumerate(variables):
        col=data[:,j]; counts=Counter(col)
        prob_tables[var]={k:v/n_rows for k,v in sorted(counts.items())}
    entropies = {var: entropy(list(prob_tables[var].values())) for var in variables}
    mi_dict = {}; mi_matrix = np.zeros((n_cols, n_cols))
    for v1,v2 in combinations(range(n_cols),2):
        m=mutual_info(data[:,v1],data[:,v2],n_rows)
        mi_dict[(v1,v2)]=m; mi_dict[(v2,v1)]=m; mi_matrix[v1,v2]=m; mi_matrix[v2,v1]=m

    # Prim MAX
    visited=[False]*n_cols; visited[0]=True; pmax=[]; pmax_cost=0
    for _ in range(n_cols-1):
        bw=-1.0; bu=bv=-1
        for u in range(n_cols):
            if visited[u]:
                for v in range(n_cols):
                    if not visited[v] and mi_matrix[u,v]>bw:
                        bw=mi_matrix[u,v]; bu,bv=u,v
        visited[bv]=True; pmax.append((bu,bv,bw)); pmax_cost+=bw

    return variables, pmax, pmax_cost, mi_matrix

def draw_prim_step(visited, prev, new_e, step, total, vl, G, POS):
    plt.figure(figsize=(14,10))
    nc=len(vl)
    nc_colors=[NODE_GREEN if visited[i] else NODE_GRAY for i in range(nc)]
    nx.draw_networkx_nodes(G,POS,node_color=nc_colors,node_size=2200,edgecolors=NODE_BORDER,linewidths=2.5)
    nx.draw_networkx_labels(G,POS,{v:v for v in vl},font_size=12,font_weight='bold',font_color='white')
    el=[(vl[u],vl[v]) for u,v,_ in prev]; ew={}
    for u,v,d in prev: ew[(vl[u],vl[v])]=f"{d:.4f}"
    if el: nx.draw_networkx_edges(G,POS,edgelist=el,edge_color=EDGE_GREEN,width=3.5,alpha=0.8)
    if new_e:
        u,v,d=new_e; nx.draw_networkx_edges(G,POS,edgelist=[(vl[u],vl[v])],edge_color=EDGE_NEW,width=6,alpha=0.9)
        ew[(vl[u],vl[v])]=f"{d:.4f}"
    if ew: nx.draw_networkx_edge_labels(G,POS,edge_labels=ew,font_size=9,font_weight='bold',
                                          bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.85))
    plt.title(f"PRIM MAX - Paso {step}/{total}",fontsize=18,fontweight='bold',color='#2c3e50',pad=20)
    if new_e:
        u,v,d=new_e; plt.suptitle(f"{vl[u]} ------ {vl[v]}   (MI = {d:.4f})",fontsize=14,color=NODE_GREEN,y=0.82)
    plt.figtext(0.5,0.01,"PRESIONE ENTER en esta ventana para continuar",ha='center',fontsize=11,color='#7f8c8d',fontweight='bold')
    plt.axis('off'); plt.tight_layout()

def draw_kruskal_step(acc, cur, is_acc, step, conn, n_acc, tot, vl, G, POS):
    plt.figure(figsize=(14,10))
    nc=len(vl)
    nc_colors=[NODE_GREEN if conn[i] else NODE_GRAY for i in range(nc)]
    nx.draw_networkx_nodes(G,POS,node_color=nc_colors,node_size=2200,edgecolors=NODE_BORDER,linewidths=2.5)
    nx.draw_networkx_labels(G,POS,{v:v for v in vl},font_size=12,font_weight='bold',font_color='white')
    ew={}
    if acc:
        al=[(vl[u],vl[v]) for u,v,_ in acc]
        for u,v,d in acc: ew[(vl[u],vl[v])]=f"{d:.4f}"
        nx.draw_networkx_edges(G,POS,edgelist=al,edge_color=EDGE_GREEN,width=3.5,alpha=0.8)
    u,v,d=cur; cl=(vl[u],vl[v])
    if is_acc: nx.draw_networkx_edges(G,POS,edgelist=[cl],edge_color=EDGE_NEW,width=6,alpha=0.9); ew[cl]=f"{d:.4f}"
    else: nx.draw_networkx_edges(G,POS,edgelist=[cl],edge_color=EDGE_RED,width=3,alpha=0.9,style='dashed')
    if ew: nx.draw_networkx_edge_labels(G,POS,edge_labels=ew,font_size=9,font_weight='bold',
                                          bbox=dict(boxstyle='round,pad=0.3',facecolor='white',alpha=0.85))
    vd="ACEPTADA" if is_acc else "RECHAZADA (ciclo)"
    vc=NODE_GREEN if is_acc else EDGE_RED
    plt.title(f"KRUSKAL MAX - Arista #{step}  [{n_acc}/{tot}]",fontsize=18,fontweight='bold',color='#2c3e50',pad=20)
    plt.suptitle(f"{vl[u]} ------ {vl[v]}   (MI = {d:.4f})",fontsize=14,color=vc,y=0.82)
    plt.figtext(0.5,0.06,vd,ha='center',fontsize=16,fontweight='bold',color=vc)
    plt.figtext(0.5,0.01,"PRESIONE ENTER en esta ventana para continuar",ha='center',fontsize=11,color='#7f8c8d',fontweight='bold')
    plt.axis('off'); plt.tight_layout()

def rVP(mi_matrix, variables):
    n=len(variables); G=nx.Graph()
    for i in range(n):
        for j in range(i+1,n): G.add_edge(i,j,weight=mi_matrix[i,j])
    G_neg=nx.Graph()
    for u,v,data in G.edges(data=True): G_neg.add_edge(u,v,weight=-data['weight'])
    mst_neg=nx.minimum_spanning_tree(G_neg)
    G_mst=nx.Graph()
    for u,v in mst_neg.edges(): G_mst.add_edge(variables[u],variables[v],weight=mi_matrix[u,v])
    pm=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i!=j:
                p=nx.shortest_path(G_mst,variables[i],variables[j],weight='weight')
                pm[i,j]=sum(G_mst[p[k]][p[k+1]]['weight'] for k in range(len(p)-1))
    return {i:sum(pm[i]) for i in range(n)}

# =============================================================================
print("\n"+"="*70)
print("  VISUAL: CONCRETE - MAX (MI) + rVP")
print("="*70)

all_data = {}
for name, file in [("BEST","d9_concrete_B.csv"),("WORST","d9_concrete_W.csv")]:
    with open(file) as f: lines=f.readlines()
    h=lines[0].strip().split(','); d=np.array([[int(x) for x in l.strip().split(',')] for l in lines[1:]])
    vl,pmax,pmax_c,mi_mat=run_pipeline(h,d)
    all_data[name]={'vars':vl,'pmax':pmax,'pmax_cost':pmax_c,'mi_mat':mi_mat}

# PANTALLA 1-2: Matrices MI
for name in ["BEST","WORST"]:
    dd=all_data[name]; vl=dd['vars']; mi_mat=dd['mi_mat']
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,8))
    fig.suptitle(f"MATRIZ IM 8x8 - {name}",fontsize=16,fontweight='bold',color='#2c3e50',y=0.98)
    vmax=np.max(mi_mat)*1.05 if np.max(mi_mat)>0 else 1
    im1=ax1.imshow(mi_mat,cmap='YlOrRd',vmin=0,vmax=vmax,aspect='equal')
    ax1.set_xticks(range(len(vl))); ax1.set_xticklabels(vl,fontsize=9,fontweight='bold',rotation=45)
    ax1.set_yticks(range(len(vl))); ax1.set_yticklabels(vl,fontsize=9,fontweight='bold')
    for i in range(len(vl)):
        for j in range(len(vl)):
            val=mi_mat[i,j]; c='white' if val>vmax*0.35 else '#2c3e50'
            ax1.text(j,i,f"{val:.4f}",ha='center',va='center',fontsize=8,fontweight='bold',color=c)
    ax1.set_title("Informacion Mutua I(X;Y)",fontsize=12,fontweight='bold',color='#e74c3c')
    fig.colorbar(im1,ax=ax1,shrink=0.82)
    im2=ax2.imshow(mi_mat>mi_mat.mean(),cmap='Greens',aspect='equal')
    ax2.set_xticks(range(len(vl))); ax2.set_xticklabels(vl,fontsize=9,fontweight='bold',rotation=45)
    ax2.set_yticks(range(len(vl))); ax2.set_yticklabels(vl,fontsize=9,fontweight='bold')
    ax2.set_title("Relaciones por encima de la media",fontsize=12,fontweight='bold',color='#27ae60')
    plt.figtext(0.5,0.01,"PRESIONE ENTER en esta ventana para continuar",ha='center',fontsize=11,color='#7f8c8d',fontweight='bold')
    plt.tight_layout(); wait_for_enter()

# PANTALLA 3-6: Prim y Kruskal paso a paso
for name in ["BEST","WORST"]:
    dd=all_data[name]; vl=dd['vars']; nv=len(vl); ts=nv-1
    mi_mat=dd['mi_mat']

    G=nx.Graph()
    for v in vl: G.add_node(v)
    for i in range(nv):
        for j in range(i+1,nv): G.add_edge(vl[i],vl[j])
    POS=nx.spring_layout(G,seed=42,k=3,iterations=100)

    # Prim MAX
    print(f"\n  >>> {name} - PRIM MAX")
    visited=[False]*nv; visited[0]=True; pe=[]; pc=0
    draw_prim_step(visited,[],None,0,ts,vl,G,POS); wait_for_enter()
    for st in range(1,ts+1):
        bw=-1.0; bu=bv=-1
        for u in range(nv):
            if visited[u]:
                for v in range(nv):
                    if not visited[v] and mi_mat[u,v]>bw: bw=mi_mat[u,v]; bu,bv=u,v
        visited[bv]=True; ne=(bu,bv,bw); pe.append(ne); pc+=bw
        print(f"    Paso {st}: {vl[bu]} -- {vl[bv]}  MI={bw:.4f}")
        draw_prim_step(visited,pe,ne,st,ts,vl,G,POS); wait_for_enter()

    # Kruskal MAX
    print(f"\n  >>> {name} - KRUSKAL MAX")
    all_e=[]
    for i in range(nv):
        for j in range(i+1,nv): all_e.append((mi_mat[i,j],i,j))
    all_e.sort(reverse=True)
    uf=UnionFind(nv); ke=[]; kc=0; conn=[False]*nv; sk=0
    draw_kruskal_step([],(0,0,0),True,0,conn,0,ts,vl,G,POS); wait_for_enter()
    for w,u,v in all_e:
        sk+=1; acc=uf.union(u,v)
        if acc: ke.append((u,v,w)); kc+=w; conn[u]=True; conn[v]=True
        vd="ACEPTADA" if acc else "RECHAZADA"
        print(f"    Eval #{sk}: {vl[u]} -- {vl[v]}  MI={w:.4f}  {vd}  ({len(ke)}/{ts})")
        draw_kruskal_step(ke,(u,v,w),acc,sk,conn,len(ke),ts,vl,G,POS); wait_for_enter()
        if acc and len(ke)==ts: break

# PANTALLA 7: Comparacion 2x2
fig,axes=plt.subplots(2,2,figsize=(18,14))
fig.suptitle("COMPARACION: BEST vs WORST - MAX (MI)",fontsize=18,fontweight='bold',color='#2c3e50',y=0.98)
v0=all_data['BEST']['vars']
for ax,name,color,title in [(axes[0,0],'BEST',NODE_GREEN,"BEST - MAX (MI)"),
                               (axes[0,1],'BEST',NODE_GREEN,"BEST - Estructura"),
                               (axes[1,0],'WORST',NODE_BLUE,"WORST - MAX (MI)"),
                               (axes[1,1],'WORST',NODE_BLUE,"WORST - Estructura")]:
    dd=all_data[name]; vl=dd['vars']; edges=dd['pmax']; cost=dd['pmax_cost']
    GG=nx.Graph()
    for v in vl: GG.add_node(v)
    for i in range(len(vl)):
        for j in range(i+1,len(vl)): GG.add_edge(vl[i],vl[j])
    PP=nx.spring_layout(GG,seed=42,k=3,iterations=100)
    nx.draw_networkx_nodes(GG,PP,ax=ax,node_color=[color]*len(vl),node_size=1800,edgecolors=NODE_BORDER,linewidths=2)
    nx.draw_networkx_labels(GG,PP,ax=ax,labels={v:v for v in vl},font_size=11,font_weight='bold',font_color='white')
    el=[(vl[u],vl[v]) for u,v,_ in edges]
    ew={(vl[u],vl[v]):f"{d:.4f}" for u,v,d in edges}
    nx.draw_networkx_edges(GG,PP,ax=ax,edgelist=el,edge_color=color,width=3.5,alpha=0.9)
    nx.draw_networkx_edge_labels(GG,PP,ax=ax,edge_labels=ew,font_size=7,font_weight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2',facecolor='white',alpha=0.85))
    ax.set_title(f"{title}\nCosto = {cost:.4f}",fontsize=14,fontweight='bold',color=color); ax.axis('off')
plt.figtext(0.5,0.01,"PRESIONE ENTER para continuar",ha='center',fontsize=11,color='#7f8c8d',fontweight='bold')
plt.tight_layout(); wait_for_enter()

# PANTALLA 8: rVP
print("\n  >>> Mostrando rVP...")
sb=rVP(all_data['BEST']['mi_mat'],v0); sw=rVP(all_data['WORST']['mi_mat'],v0)
ranking=[]
for i,var in enumerate(v0):
    delta=abs(sb[i]-sw[i]); ranking.append((var,sb[i],sw[i],delta))
ranking.sort(key=lambda x:-x[3])
umbral=0.5

fig,(ax_bar,ax_table)=plt.subplots(1,2,figsize=(18,9),gridspec_kw={'width_ratios':[1.2,1]})
fig.suptitle("SELECCION rVP - Concrete",fontsize=16,fontweight='bold',color='#2c3e50',y=0.98)
vr=[r[0] for r in ranking[::-1]]; dr=[r[3] for r in ranking[::-1]]
colors=['#e74c3c' if d>umbral else '#27ae60' for d in dr]
ax_bar.barh(range(len(vr)),dr,color=colors,edgecolor='white')
for i,(v,d) in enumerate(zip(vr,dr)): ax_bar.text(d+0.05,i,f"{v} ({d:.4f})",va='center',fontsize=11,fontweight='bold')
ax_bar.set_yticks(range(len(vr))); ax_bar.set_yticklabels(vr,fontsize=12,fontweight='bold')
ax_bar.set_xlabel('|Delta|',fontsize=11)
ax_bar.set_title(f'Rojo = CRITICA (|D|>{umbral})',fontsize=12,fontweight='bold',color='#2c3e50')
ax_table.axis('off')
td=[['Var','SUM BEST','SUM WORST','|Delta|','Critica?']]
for var,sb_,sw_,d in ranking:
    td.append([var,f'{sb_:.4f}',f'{sw_:.4f}',f'{d:.4f}','SI' if d>umbral else 'No'])
tbl=ax_table.table(cellText=td,cellLoc='center',loc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1,1.5)
for key,cell in tbl.get_celld().items():
    cell.set_edgecolor('#2c3e50'); cell.set_linewidth(1)
    if key[0]==0: cell.set_facecolor('#2c3e50'); cell.get_text().set_color('white')
ax_table.set_title('Ranking rVP',fontsize=13,fontweight='bold',pad=15)
criticas=[v for v,_,_,d in ranking if d>umbral]
plt.figtext(0.5,0.01,f"CRITICAS: {criticas}  ({len(criticas)}/{len(ranking)})",ha='center',fontsize=12,color='#e74c3c',fontweight='bold')
plt.tight_layout(); wait_for_enter()

print("\n"+"="*70)
print("  VISUALIZACION COMPLETADA")
print("="*70)
