"""Respostas ao parecer final do CBS 2026 (versão 16), ponto 3.

1. Imputação dentro do pipeline: a mediana passa a ser calculada só na partição
   de treino (SimpleImputer), junto com a padronização. A presença continua com
   zero para quem não tem registro, regra que não depende de outros estudantes.
2. Modelo fixo: a Tabela 1 passa a reportar a regressão logística em todas as
   células; a floresta aleatória vai para o material suplementar
   (early_warning_pipeline.csv), sem escolha do melhor modelo por célula.
3. Testes de permutação com o pipeline completo (imputação, padronização e
   modelo reajustados em cada partição e em cada permutação).
Requer os arquivos gerados por 01 e 02 em EPM_OUT.
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, permutation_test_score
warnings.filterwarnings("ignore")
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
LIMIAR = int(os.environ.get("LIMIAR_APROVACAO", 70))
M = pd.read_pickle(f"{OUT}/metrics.pkl"); F = pd.read_pickle(f"{OUT}/final.pkl")

def features(upto):
    """Como em 04_early.py, mas sem imputar: os NaN ficam para o pipeline."""
    E = M[M.session <= upto]
    agg = E.groupby("student").agg(presenca=("session", "nunique"), eventos=("eventos", "mean"), tempo=("tempo_tarefa_min", "mean"),
        CM_L1=("CM_L1", "mean"), CM_L2=("CM_L2", "mean"), CM_L3=("CM_L3", "mean"), pct_fora=("pct_fora", "mean"), entropia=("entropia_L3", "mean"),
        t_estudo=("t_Estudo", "mean"), t_simul=("t_Simulação", "mean"))
    ig = E[E.session >= 2].groupby("student").ig_norm.mean().rename("nota_interm")
    X = pd.DataFrame(index=range(1, 116)).join(agg).join(ig)
    X["presenca"] = X.presenca.fillna(0)
    return X
PROC = ["presenca", "CM_L1", "CM_L2", "CM_L3", "pct_fora", "entropia", "t_estudo", "t_simul"]
SETS = {"Presença": ["presenca"], "Presença e volume": ["presenca", "eventos", "tempo"],
        "Presença e processo": PROC, "Notas intermediárias": ["nota_interm"], "Processo e notas": PROC + ["nota_interm"]}
RL = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
RF = lambda: make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=1))
fb = F.set_index("student").final_best
def alvo(X, nome):
    if nome == "des": return X, (~X.index.isin(F.student)).astype(int)
    idx = [i for i in X.index if i in fb.index]; return X.loc[idx], (fb.loc[idx] >= LIMIAR).astype(int).values

print("== 1 e 2. AUC com imputação dentro do pipeline, por modelo (5 partições x 10 repetições)")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
res = []
for upto in [3, 6]:
    X = features(upto)
    for nome in ["des", "nota"]:
        XX, y = alvo(X, nome)
        for sname, cols in SETS.items():
            for mname, mdl in [("RL", RL), ("RF", RF)]:
                s = cross_val_score(mdl(), XX[cols], y, cv=cv, scoring="roc_auc", n_jobs=-1)
                res.append({"janela": f"S1-S{upto}", "alvo": nome, "conjunto": sname, "modelo": mname,
                            "AUC": s.mean(), "dp": s.std(), "n": len(y), "positivos": int(np.sum(y))})
R = pd.DataFrame(res); R.to_csv(f"{OUT}/early_warning_pipeline.csv", index=False)
pd.set_option("display.width", 200)
tab = R.assign(cel=R.AUC.map("{:.2f}".format) + " ± " + R.dp.map("{:.2f}".format))
print(tab.pivot_table(index=["modelo", "conjunto"], columns=["alvo", "janela"], values="cel", aggfunc="first").to_string())
d = R.pivot_table(index=["alvo", "janela", "conjunto"], columns="modelo", values="AUC")
print("diferença RF - RL: média", round((d.RF - d.RL).mean(), 3), "| máx. abs.", round((d.RF - d.RL).abs().max(), 3))

print("\n== 3. Testes de permutação com o pipeline completo (RL, 500 permutações, 5 partições)")
cvp = StratifiedKFold(5, shuffle=True, random_state=42)
perm = []
for upto in [3, 6]:
    X = features(upto)
    for nome in ["des", "nota"]:
        XX, y = alvo(X, nome)
        for sname, cols in SETS.items():
            sc, pm, p = permutation_test_score(RL(), XX[cols], y, cv=cvp, scoring="roc_auc", n_permutations=500, random_state=42, n_jobs=-1)
            perm.append({"janela": f"S1-S{upto}", "alvo": nome, "conjunto": sname, "AUC": sc, "media_perm": pm.mean(), "p": p})
            print(f"S1-S{upto} {nome:4s} {sname:22s} AUC={sc:.3f} média perm.={pm.mean():.3f} p={p:.4f}", flush=True)
pd.DataFrame(perm).to_csv(f"{OUT}/permutacao_pipeline.csv", index=False)

print("\n== 4. Curva de antecipação com o pipeline completo (RL, 5 partições x 10 repetições)")
rows = []
for k in range(1, 7):
    X = features(k)
    for nome in ["des", "nota"]:
        XX, y = alvo(X, nome)
        for sname in ["Presença", "Presença e processo", "Notas intermediárias"]:
            if sname.startswith("Notas") and k < 2: continue
            s = cross_val_score(RL(), XX[SETS[sname]], y, cv=cv, scoring="roc_auc", n_jobs=-1)
            rows.append({"sessoes": k, "alvo": nome, "conjunto": sname, "AUC": s.mean()})
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/curva_antecipacao_pipeline.csv", index=False)
print(C.pivot_table(index=["alvo", "conjunto"], columns="sessoes", values="AUC").round(2).to_string())
