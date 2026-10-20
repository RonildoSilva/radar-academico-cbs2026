"""Verifica o número de componentes conexos (P) dos grafos por caso e recalcula a variante filtrada com P explícito."""
import os, pandas as pd, numpy as np, networkx as nx
from scipy import stats
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
log = pd.read_pickle(f"{OUT}/log.pkl"); F = pd.read_pickle(f"{OUT}/final.pkl")
def collapse(seq):
    out=[]
    for a in seq:
        if not out or out[-1]!=a: out.append(a)
    return out
rows=[]
for (sess, stu), g in log.groupby(["session","student"]):
    g = g.sort_values(["start","end"])
    seqs = {"L1": g.activity.tolist(), "L2": g[g.in_task].activity.tolist(), "L3": g.meta.tolist()}
    r={"session":sess,"student":stu}
    for nm, seq in seqs.items():
        seq = collapse(seq); G = nx.DiGraph(); G.add_nodes_from(seq)
        ed = {}
        for a,b in zip(seq[:-1],seq[1:]): ed[(a,b)] = ed.get((a,b),0)+1
        G.add_edges_from(ed)
        r[f"P_{nm}"] = nx.number_weakly_connected_components(G) if len(G) else 0
        if nm=="L2":
            H = nx.DiGraph(); H.add_edges_from([e for e,v in ed.items() if v>=2])
            P = nx.number_weakly_connected_components(H) if len(H) else 0
            r["P_L2_f2"]=P; r["CM_L2_f2_P"] = H.number_of_edges()-H.number_of_nodes()+P
    rows.append(r)
R = pd.DataFrame(rows)
print("casos:", len(R))
for c in ["P_L1","P_L2","P_L3","P_L2_f2"]:
    print(c, R[c].value_counts().sort_index().to_dict())
S = R.groupby("student").CM_L2_f2_P.mean().rename("cm").to_frame().join(F.set_index("student").final_best).dropna()
print("CM_L2 filtrada (P explícito) x nota final:", stats.spearmanr(S.cm, S.final_best))
