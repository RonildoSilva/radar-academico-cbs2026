import os, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, permutation_test_score
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
LIMIAR = int(os.environ.get("LIMIAR_APROVACAO", 70))
M = pd.read_pickle(f"{OUT}/metrics.pkl"); F = pd.read_pickle(f"{OUT}/final.pkl"); S = pd.read_pickle(f"{OUT}/students.pkl")
rng = np.random.default_rng(42)

print("== 1. Bootstrap (5000) IC95% para correlações do nível do estudante")
T = S.dropna(subset=["final_best"]).reset_index(drop=True)
def pspear(d, x, y, z):
    r = d[[x,y,z]].rank()
    rx = r[x]-np.polyval(np.polyfit(r[z],r[x],1),r[z]); ry = r[y]-np.polyval(np.polyfit(r[z],r[y],1),r[z])
    return np.corrcoef(rx,ry)[0,1]
for nome, fn in [("rho CM_L2 x final", lambda d: stats.spearmanr(d.CM_L2,d.final_best)[0]),
                 ("rho CM_L1 x final", lambda d: stats.spearmanr(d.CM_L1,d.final_best)[0]),
                 ("parcial CM_L2|eventos", lambda d: pspear(d,"CM_L2","final_best","eventos"))]:
    est = fn(T); bs = [fn(T.iloc[rng.integers(0,len(T),len(T))]) for _ in range(5000)]
    print(f"{nome}: {est:.3f} IC95%=[{np.percentile(bs,2.5):.3f}; {np.percentile(bs,97.5):.3f}]")

print("== 2. Poder (Fisher z, bilateral, alfa 0,05)")
for r in [0.34, 0.26]:
    z = np.arctanh(r)*np.sqrt(len(T)-3); pw = 1-stats.norm.cdf(1.96-z)+stats.norm.cdf(-1.96-z)
    print(f"rho={r} n={len(T)} poder={pw:.3f}")
# n minimo para detectar rho com poder 0,80
for r in [0.34,0.26,0.20]:
    n = ((1.96+0.8416)/np.arctanh(r))**2+3; print(f"n minimo p/ rho={r}: {np.ceil(n):.0f}")

def features(upto):
    E = M[M.session<=upto]
    agg = E.groupby("student").agg(presenca=("session","nunique"), eventos=("eventos","mean"), tempo=("tempo_tarefa_min","mean"),
        CM_L1=("CM_L1","mean"), CM_L2=("CM_L2","mean"), CM_L3=("CM_L3","mean"), pct_fora=("pct_fora","mean"), entropia=("entropia_L3","mean"),
        t_estudo=("t_Estudo","mean"), t_simul=("t_Simulação","mean"))
    ig = E[E.session>=2].groupby("student").ig_norm.mean().rename("nota_interm")
    X = pd.DataFrame(index=range(1,116)).join(agg).join(ig)
    X["presenca"] = X.presenca.fillna(0)
    return X.fillna(X.median())
PROC = ["presenca","CM_L1","CM_L2","CM_L3","pct_fora","entropia","t_estudo","t_simul"]
RL = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
RF = lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=1)
fb = F.set_index("student").final_best

print("== 3. Testes de permutação (500 permutações, RL, 5 partições)")
cvp = StratifiedKFold(5, shuffle=True, random_state=42)
for nome, upto, alvo, cols in [("presença -> desengajamento", 3, "des", ["presenca"]),
                               ("processo -> nota", 3, "nota", PROC),
                               ("notas -> nota", 3, "nota", ["nota_interm"]),
                               ("processo+notas -> nota", 6, "nota", PROC+["nota_interm"])]:
    X = features(upto)
    if alvo=="des": XX, y = X[cols], (~X.index.isin(F.student)).astype(int)
    else:
        idx=[i for i in X.index if i in fb.index]; XX, y = X.loc[idx,cols], (fb.loc[idx]>=LIMIAR).astype(int).values
    sc, perm, p = permutation_test_score(RL(), XX, y, cv=cvp, scoring="roc_auc", n_permutations=500, random_state=42, n_jobs=-1)
    print(f"S1-S{upto} {nome}: AUC={sc:.3f} media_perm={perm.mean():.3f} p={p:.4f}")

print("== 4. Curva de antecipação: AUC por nº de sessões observadas (RL; RF para processo)")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
rows=[]
for k in range(1,7):
    X = features(k); y1 = (~X.index.isin(F.student)).astype(int)
    idx=[i for i in X.index if i in fb.index]; y2=(fb.loc[idx]>=LIMIAR).astype(int).values
    for alvo, XX, yy in [("Desengajamento", X, y1), (f"Nota >= {LIMIAR}", X.loc[idx], y2)]:
        for nome, cols in [("Presença",["presenca"]),("Processo",PROC),("Notas intermediárias",["nota_interm"])]:
            if nome.startswith("Notas") and k<2: continue
            best=max(cross_val_score(m(), XX[cols], yy, cv=cv, scoring="roc_auc", n_jobs=-1).mean() for m in ([RL] if nome=="Presença" else [RL,RF]))
            rows.append({"sessoes":k,"alvo":alvo,"conjunto":nome,"AUC":best})
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/curva_antecipacao.csv", index=False)
print(C.pivot_table(index=["alvo","conjunto"], columns="sessoes", values="AUC").round(3).to_string())

print("== 5. Presença por sessão: quem faltou à sessão k fez o exame?")
pres = M.pivot_table(index="student", columns="session", values="eventos", aggfunc="size").reindex(range(1,116)).notna()
fez = pd.Series(pres.index.isin(F.student), index=pres.index)
for k in range(1,7):
    print(f"S{k}: presentes={pres[k].sum()} exame entre presentes={fez[pres[k]].mean():.3f} | ausentes={(~pres[k]).sum()} exame entre ausentes={fez[~pres[k]].mean():.3f}")
aus2 = (~pres[[1,2,3]]).sum(axis=1)
for a in range(4): 
    m=aus2==a; print(f"faltas S1-S3={a}: n={m.sum()} sem exame={1-fez[m].mean():.3f}")

print("== 6. Padrão temporal da presença (trajetória de presença como processo)")
P = pres.astype(int)
def classe(r):
    s = r.values; idx = np.where(s==1)[0]
    if len(idx)==0: return "Sem registro"
    first, last = idx[0]+1, idx[-1]+1
    if s.sum()==6: return "Contínua"
    if last <= 4: return "Saída antecipada"          # último registro até a sessão 4
    if first >= 3: return "Entrada tardia"           # primeiro registro na sessão 3 ou depois
    return "Intermitente"
cl = P.apply(classe, axis=1)
tab = pd.DataFrame({"classe":cl, "sem_exame":~fez}).groupby("classe").agg(n=("sem_exame","size"), sem_exame=("sem_exame","mean"))
print(tab.round(3).to_string())
ct = pd.crosstab(cl, ~fez); print(ct)
from scipy.stats import fisher_exact
a = ct.loc["Saída antecipada"].values if "Saída antecipada" in ct.index else None
# saída antecipada vs demais
m = cl=="Saída antecipada"
print("Fisher saída vs demais:", fisher_exact([[ (~fez)[m].sum(), m.sum()-(~fez)[m].sum()], [(~fez)[~m].sum(), (~m).sum()-(~fez)[~m].sum()]]))
# Na janela S1-S3: 'saída' = presente em S1 ou S2 e ausente em S3; 'entrada tardia' = ausente em S1 e S2
e3 = pd.Series(np.where((P[3]==0)&((P[1]==1)|(P[2]==1)),"Presente e depois ausente (até S3)",
           np.where((P[1]==0)&(P[2]==0),"Ausente em S1 e S2", "Presente em S3")), index=P.index)
print(pd.DataFrame({"p":e3,"sem":~fez}).groupby("p").agg(n=("sem","size"), sem_exame=("sem","mean")).round(3))
# AUC de um indicador de ruptura (S1-S3) vs contagem de presença
from sklearn.metrics import roc_auc_score
ruptura = ((P[3]==0)&((P[1]==1)|(P[2]==1))).astype(int)
cont = P[[1,2,3]].sum(axis=1)
X3 = pd.DataFrame({"presenca":cont, "ruptura":ruptura, "ultima_presente":P[3]})
y = (~fez).astype(int)
for cols in [["presenca"],["ultima_presente"],["presenca","ruptura"]]:
    s = cross_val_score(RL(), X3[cols], y, cv=cv, scoring="roc_auc").mean(); print("S1-S3", cols, round(s,3))

print("== 7. Regra em tempo real: duas ausências consecutivas após ter estado presente")
def flag(r, k=2):
    s = r.values; seen=False; run=0
    for i,v in enumerate(s):
        if v==1: seen=True; run=0
        elif seen:
            run+=1
            if run>=k: return i+1
    return np.nan
for k in [1,2]:
    fl = P.apply(lambda r: flag(r,k), axis=1)
    y = ~fez; pred = fl.notna()
    tp=(pred&y).sum(); fp=(pred&~y).sum(); fn=(~pred&y).sum()
    print(f"k={k}: sinalizados={pred.sum()} VP={tp} FP={fp} FN={fn} precisão={tp/max(1,tp+fp):.3f} sensibilidade={tp/(tp+fn):.3f}")
    print("   sessão do alerta (VP):", fl[pred&y].value_counts().sort_index().to_dict(), "| (FP):", fl[pred&~y].value_counts().sort_index().to_dict())
