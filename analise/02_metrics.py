import os, pandas as pd, numpy as np
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get("EPM_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "epm", "EPM Dataset 2", "Data"))
log = pd.read_pickle(f"{OUT}/log.pkl")
neg = (log.dur < 0).sum(); print("duracoes negativas:", neg, f"({neg/len(log):.4%})")
log["dur"] = log["dur"].clip(lower=0)

def collapse(seq):
    out=[]; 
    for a in seq:
        if not out or out[-1]!=a: out.append(a)
    return out

def dfg_metrics(seq, min_freq=1):
    seq = collapse(seq)
    edges = {}
    for a,b in zip(seq[:-1], seq[1:]):
        edges[(a,b)] = edges.get((a,b),0)+1
    E = {k for k,v in edges.items() if v>=min_freq}
    V = set(a for e in E for a in e) if min_freq>1 else set(seq)
    if not V: return 0,0,0,len(seq)
    cm = len(E)-len(V)+1
    return cm, len(V), len(E), len(seq)

def entropy(s):
    p = s.value_counts(normalize=True).values
    return float(-(p*np.log2(p)).sum())

rec=[]
for (sess, stu), g in log.groupby(["session","student"], sort=True):
    g = g.sort_values(["start","end"])
    r = {"session":sess, "student":stu}
    l1 = g["activity"].tolist()
    gi = g[g.in_task]
    l2 = gi["activity"].tolist()
    l3 = g["meta"].tolist()
    for nm, seq in [("L1",l1),("L2",l2),("L3",l3)]:
        cm,v,e,n = dfg_metrics(seq)
        r[f"CM_{nm}"]=cm; r[f"V_{nm}"]=v; r[f"E_{nm}"]=e; r[f"N_{nm}"]=n
    r["CM_L2_f2"] = dfg_metrics(l2, min_freq=2)[0]
    tot = g.dur.sum()
    r["tempo_total_min"] = tot/60
    r["tempo_tarefa_min"] = gi.dur.sum()/60
    r["pct_fora"] = 1 - gi.dur.sum()/tot if tot>0 else np.nan
    r["eventos"] = len(g)
    r["entropia_L3"] = entropy(g["meta"])
    r["idle_ratio"] = (g.idle.sum()/1000)/tot if tot>0 else np.nan
    r["teclas"] = g["keys"].sum(); r["cliques"] = g["click_l"].sum()
    r["exercicios"] = g.loc[g.exercise.str.match(r"Es_\d+_\d+"), "exercise"].nunique()
    for m in ["Estudo","Construção","Simulação","Relato"]:
        r[f"t_{m}"] = g.loc[g.meta==m,"dur"].sum()/tot if tot>0 else 0
    rec.append(r)
M = pd.DataFrame(rec)

# notas intermediarias (normalizadas pelo maximo observado na sessao)
ig = pd.read_excel(f"{D}/intermediate_grades.xlsx").rename(columns={"Student Id":"student"})
ig_long = ig.melt(id_vars="student", var_name="sess", value_name="ig")
ig_long["session"] = ig_long["sess"].str.extract(r"(\d+)").astype(int)
mx = ig_long.groupby("session").ig.transform("max")
ig_long["ig_norm"] = ig_long.ig/mx
M = M.merge(ig_long[["student","session","ig","ig_norm"]], on=["student","session"], how="left")

# nota final: primeira e melhor tentativa
fg = pd.read_excel(f"{D}/final_grades.xlsx", sheet_name=None)
sh = list(fg.values())
def tot(df):
    c=[c for c in df.columns if str(c).startswith("TOTAL")][0]
    return df[["Student ID",c]].rename(columns={"Student ID":"student", c:"final"})
f1, f2 = tot(sh[0]), tot(sh[1])
both = f1.merge(f2, on="student", how="outer", suffixes=("_1","_2"))
both["final_best"] = both[["final_1","final_2"]].max(axis=1)
both["final_first"] = both["final_1"].fillna(both["final_2"])
both["n_tentativas"] = both[["final_1","final_2"]].notna().sum(axis=1)
print("alunos com nota final:", len(both), "| duas tentativas:", (both.n_tentativas==2).sum(), "| ids > 115:", (both.student>115).sum())
M.to_pickle(f"{OUT}/metrics.pkl"); both.to_pickle(f"{OUT}/final.pkl")
print(M.describe().T[["mean","std","min","50%","max"]].round(2).to_string())
