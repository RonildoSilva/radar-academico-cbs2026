import os, pandas as pd, numpy as np
from scipy import stats
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
M = pd.read_pickle(f"{OUT}/metrics.pkl"); F = pd.read_pickle(f"{OUT}/final.pkl")
def sp(x,y):
    m = x.notna()&y.notna(); r,p = stats.spearmanr(x[m],y[m]); return r,p,m.sum()
def partial_spearman(x,y,z):
    d = pd.DataFrame({"x":x,"y":y,"z":z}).dropna().rank()
    rx = d.x - np.polyval(np.polyfit(d.z,d.x,1), d.z)
    ry = d.y - np.polyval(np.polyfit(d.z,d.y,1), d.z)
    r,p = stats.pearsonr(rx,ry); return r,p,len(d)

print("== A. Por sessão: médias")
A = M.groupby("session").agg(casos=("student","size"), CM_L1=("CM_L1","mean"), CM_L2=("CM_L2","mean"), CM_L3=("CM_L3","mean"),
     eventos=("eventos","mean"), tempo=("tempo_tarefa_min","mean"), pct_fora=("pct_fora","mean"), ig_norm=("ig_norm","mean"))
print(A.round(3).to_string())
A2 = A.loc[2:6]
for c in ["CM_L1","CM_L2","CM_L3","eventos"]:
    r,p = stats.spearmanr(A2[c], A2.ig_norm); rp,pp = stats.pearsonr(A2[c], A2.ig_norm)
    print(f"sessões 2-6 {c} vs nota interm.: spearman={r:.3f} (p={p:.3f}) pearson={rp:.3f} (p={pp:.3f})")

print("\n== B. Aluno-sessão (sessões 2-6): CM vs nota intermediária")
B = M[M.session>=2]
for c in ["CM_L1","CM_L2","CM_L3","CM_L2_f2","eventos","tempo_tarefa_min","pct_fora","entropia_L3"]:
    r,p,n = sp(B[c],B.ig_norm); print(f"{c:18s} rho={r:+.3f} p={p:.4f} n={n}")
print("-- por sessão (CM_L2)")
for s,g in B.groupby("session"):
    r,p,n = sp(g.CM_L2,g.ig_norm); r2,p2,_=sp(g.eventos,g.ig_norm); print(f"S{s}: CM_L2 rho={r:+.3f} p={p:.3f} | eventos rho={r2:+.3f} p={p2:.3f} n={n}")
print("-- parcial CM_L2 | eventos:", [round(v,4) for v in partial_spearman(B.CM_L2,B.ig_norm,B.eventos)])
print("-- parcial CM_L2 | N_L2:", [round(v,4) for v in partial_spearman(B.CM_L2,B.ig_norm,B.N_L2)])
print("-- corr CM_L2 vs N_L2:", sp(B.CM_L2,B.N_L2))

print("\n== C. Nível aluno: médias de processo vs nota final")
S = M.groupby("student").agg(sessoes=("session","nunique"), CM_L1=("CM_L1","mean"), CM_L2=("CM_L2","mean"), CM_L3=("CM_L3","mean"),
     eventos=("eventos","mean"), N_L2=("N_L2","mean"), tempo=("tempo_tarefa_min","mean"), pct_fora=("pct_fora","mean"),
     entropia=("entropia_L3","mean"), ig=("ig_norm","mean")).reset_index()
S = S.merge(F[["student","final_best","final_first","n_tentativas"]], on="student", how="left")
S.to_pickle(f"{OUT}/students.pkl")
print("alunos com log:", len(S), "| com nota final:", S.final_best.notna().sum(), "| sem nota final:", S.final_best.isna().sum())
for tgt in ["final_best","final_first"]:
    print("alvo", tgt)
    for c in ["sessoes","CM_L1","CM_L2","CM_L3","eventos","tempo","pct_fora","entropia","ig"]:
        r,p,n = sp(S[c],S[tgt]); print(f"   {c:10s} rho={r:+.3f} p={p:.4f} n={n}")
print("parcial CM_L2|eventos (final_best):", [round(v,4) for v in partial_spearman(S.CM_L2,S.final_best,S.eventos)])
print("parcial CM_L2|sessoes (final_best):", [round(v,4) for v in partial_spearman(S.CM_L2,S.final_best,S.sessoes)])
print("parcial pct_fora|sessoes:", [round(v,4) for v in partial_spearman(S.pct_fora,S.final_best,S.sessoes)])
# fez ou não exame
r = stats.mannwhitneyu(S.loc[S.final_best.notna(),"sessoes"], S.loc[S.final_best.isna(),"sessoes"])
print("sessões: com exame", S.loc[S.final_best.notna(),"sessoes"].mean().round(2), "sem exame", S.loc[S.final_best.isna(),"sessoes"].mean().round(2), "MWU p=",round(r.pvalue,4))

print("\n== D. Clusters mu +- n sigma (nota final best)")
T = S.dropna(subset=["final_best"]); mu, sd = T.final_best.mean(), T.final_best.std()
print(f"mu={mu:.2f} sd={sd:.2f}")
for n in [0,0.5,1.0]:
    hi = T[T.final_best>mu+n*sd]; lo = T[T.final_best<mu-n*sd]
    for c in ["CM_L1","CM_L2","CM_L3","pct_fora"]:
        u = stats.mannwhitneyu(hi[c],lo[c]); 
        print(f"n={n} {c}: alto={hi[c].mean():.2f} (n={len(hi)}) baixo={lo[c].mean():.2f} (n={len(lo)}) MWU p={u.pvalue:.4f}")
# por sessão cluster n=0
print("-- por sessão, CM_L2 alto vs baixo (n=0)")
MM = M.merge(T[["student","final_best"]], on="student")
for s,g in MM.groupby("session"):
    hi=g[g.final_best>mu]; lo=g[g.final_best<mu]
    u=stats.mannwhitneyu(hi.CM_L2,lo.CM_L2)
    print(f"S{s}: alto={hi.CM_L2.mean():.2f} dp={hi.CM_L2.std():.2f} (n={len(hi)}) baixo={lo.CM_L2.mean():.2f} dp={lo.CM_L2.std():.2f} (n={len(lo)}) p={u.pvalue:.4f}")
