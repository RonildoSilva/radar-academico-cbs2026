"""Respostas à quarta revisão externa (versão 11).

1. Modelo nulo da complexidade: embaralha a ordem das atividades de cada caso,
   preservando o repertório e o número de visitas, e mede a complexidade que
   excede a esperada pelo acaso (seção 4.3).
2. Desbalanceamento de classes: ponderação de classes (seção 4.4).
3. Hiperparâmetros: busca em grade aninhada na validação cruzada (seção 4.4).
4. Circularidade da regra de ruptura: escolha do número de ausências dentro da
   validação cruzada e intervalo de Wilson para a precisão (seção 4.5).
Requer os arquivos gerados por 01 a 03 em EPM_OUT.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV,
                                     cross_val_score)
warnings.filterwarnings("ignore")
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
LIMIAR = int(os.environ.get("LIMIAR_APROVACAO", 70))
log = pd.read_pickle(f"{OUT}/log.pkl"); M = pd.read_pickle(f"{OUT}/metrics.pkl")
F = pd.read_pickle(f"{OUT}/final.pkl"); S = pd.read_pickle(f"{OUT}/students.pkl")
rng = np.random.default_rng(42)

def proportion_confint(k, n, z=1.959964, method="wilson"):
    p = k / n; d = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h

def collapse(seq):
    out = []
    for a in seq:
        if not out or out[-1] != a: out.append(a)
    return out
def cm(seq):
    seq = collapse(seq)
    E = set(zip(seq[:-1], seq[1:]))
    return len(E) - len(set(seq)) + 1 if seq else 0

print("== 1. Modelo nulo: ordem das atividades embaralhada (50 permutações por caso, nível 2)")
R = 50; rows = []
for (sess, stu), g in log[log.in_task].sort_values(["session", "student", "start", "end"]).groupby(["session", "student"]):
    seq = collapse(g.activity.tolist())
    obs = cm(seq); arr = np.array(seq, dtype=object)
    null = np.mean([cm(list(rng.permutation(arr))) for _ in range(R)])
    rows.append({"session": sess, "student": stu, "cm_obs": obs, "cm_nulo": null, "visitas": len(seq)})
N = pd.DataFrame(rows)
N["excesso"] = N.cm_obs - N.cm_nulo            # negativo: mais estruturado que o acaso
N["razao"] = N.cm_obs / N.cm_nulo
print(N[["cm_obs", "cm_nulo", "razao"]].describe().round(3).to_string())
print("casos com complexidade abaixo da nula:", f"{(N.excesso < 0).mean():.4f}")
SN = N.groupby("student").agg(cm_obs=("cm_obs", "mean"), cm_nulo=("cm_nulo", "mean"), razao=("razao", "mean"),
                              excesso=("excesso", "mean"), visitas=("visitas", "mean")).reset_index()
SN = SN.merge(S[["student", "final_best", "eventos"]], on="student").dropna(subset=["final_best"])
def pspear(d, x, y, z):
    r = d[[x, y, z]].rank()
    rx = r[x] - np.polyval(np.polyfit(r[z], r[x], 1), r[z]); ry = r[y] - np.polyval(np.polyfit(r[z], r[y], 1), r[z])
    return stats.pearsonr(rx, ry)
for c in ["cm_obs", "cm_nulo", "razao", "excesso"]:
    r, p = stats.spearmanr(SN[c], SN.final_best)
    bs = [stats.spearmanr(*SN.iloc[rng.integers(0, len(SN), len(SN))][[c, "final_best"]].values.T)[0] for _ in range(2000)]
    print(f"{c:8s} x nota final: rho={r:+.3f} p={p:.4f} IC95%=[{np.percentile(bs,2.5):+.3f}; {np.percentile(bs,97.5):+.3f}]")
r, p = pspear(SN, "cm_obs", "final_best", "cm_nulo"); print(f"parcial cm_obs | cm_nulo: rho={r:+.3f} p={p:.4f}")
r, p = pspear(SN, "cm_obs", "final_best", "visitas"); print(f"parcial cm_obs | visitas (sequência fundida): rho={r:+.3f} p={p:.4f}")
r, p = stats.spearmanr(SN.cm_nulo, SN.visitas); print(f"cm_nulo x visitas: rho={r:+.3f}")
VD = M.groupby("student").V_L2.mean().rename("vertices")
SN = SN.merge(VD, on="student")
r, p = pspear(SN, "cm_obs", "final_best", "vertices"); print(f"parcial cm_obs | atividades distintas: rho={r:+.3f} p={p:.4f}")
N.to_csv(f"{OUT}/modelo_nulo.csv", index=False)

def features(upto):
    E = M[M.session <= upto]
    agg = E.groupby("student").agg(presenca=("session", "nunique"), CM_L1=("CM_L1", "mean"), CM_L2=("CM_L2", "mean"),
        CM_L3=("CM_L3", "mean"), pct_fora=("pct_fora", "mean"), entropia=("entropia_L3", "mean"),
        t_estudo=("t_Estudo", "mean"), t_simul=("t_Simulação", "mean"))
    X = pd.DataFrame(index=range(1, 116)).join(agg)
    X["presenca"] = X.presenca.fillna(0)
    return X  # a mediana é imputada dentro do pipeline, só com a partição de treino
PROC = ["presenca", "CM_L1", "CM_L2", "CM_L3", "pct_fora", "entropia", "t_estudo", "t_simul"]
fb = F.set_index("student").final_best
def alvo(X, nome):
    if nome == "des": return X, (~X.index.isin(F.student)).astype(int)
    idx = [i for i in X.index if i in fb.index]; return X.loc[idx], (fb.loc[idx] >= LIMIAR).astype(int).values

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
print("\n== 2. Ponderação de classes (presença e processo)")
for upto in [3, 6]:
    for nome in ["des", "nota"]:
        XX, y = alvo(features(upto), nome); out = []
        for cw in [None, "balanced"]:
            rl = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("m", LogisticRegression(max_iter=2000, class_weight=cw))])
            rf = Pipeline([("i", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight=cw, random_state=42, n_jobs=1))])
            a = [cross_val_score(m, XX[PROC], y, cv=cv, scoring="roc_auc", n_jobs=-1).mean() for m in (rl, rf)]
            out.append(f"{cw or 'sem'}: RL={a[0]:.3f} RF={a[1]:.3f}")
        print(f"S1-S{upto} {nome} (positivos={int(np.sum(y))}/{len(y)}): " + " | ".join(out))

print("\n== 3. Busca em grade aninhada (externa 5x5, interna 5 partições), presença e processo")
cvo = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42); cvi = StratifiedKFold(5, shuffle=True, random_state=7)
GRADES = {
    "RL": (Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000, solver="liblinear"))]),
           {"m__C": [0.01, 0.1, 1, 10, 100], "m__penalty": ["l1", "l2"], "m__class_weight": [None, "balanced"]}),
    "RF": (Pipeline([("i", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=1))]),
           {"m__min_samples_leaf": [1, 3, 5, 10], "m__max_depth": [None, 3, 5], "m__max_features": ["sqrt", None],
            "m__class_weight": [None, "balanced"]}),
}
for upto in [3, 6]:
    for nome in ["des", "nota"]:
        XX, y = alvo(features(upto), nome); out = []
        for mn, (est, grid) in GRADES.items():
            gs = GridSearchCV(est, grid, cv=cvi, scoring="roc_auc", n_jobs=-1)
            s = cross_val_score(gs, XX[PROC], y, cv=cvo, scoring="roc_auc", n_jobs=1)
            out.append(f"{mn}={s.mean():.3f} (dp {s.std():.3f})")
        print(f"S1-S{upto} {nome}: " + " | ".join(out), flush=True)

print("\n== 4. Regra de ruptura: circularidade")
pres = M.pivot_table(index="student", columns="session", values="eventos", aggfunc="size").reindex(range(1, 116)).notna().astype(int)
sem = pd.Series(~pres.index.isin(F.student), index=pres.index)
def sinal(r, k):
    seen = False; run = 0
    for i, v in enumerate(r):
        if v: seen = True; run = 0
        elif seen:
            run += 1
            if run >= k: return i + 1
    return 0
FL = {k: pres.apply(lambda r: sinal(r.values, k), axis=1) for k in [1, 2, 3]}
def met(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    return tp, fp, fn
for k, fl in FL.items():
    tp, fp, fn = met(fl > 0, sem); lo, hi = proportion_confint(tp, tp + fp, method="wilson")
    print(f"k={k}: sinalizados={tp+fp} VP={tp} precisão={tp/(tp+fp):.4f} IC95% Wilson=[{lo:.4f}; {hi:.4f}] sensibilidade={tp/(tp+fn):.4f}")
lo, hi = proportion_confint(int(sem.sum()), len(sem), method="wilson"); print(f"taxa base={sem.mean():.4f} IC95%=[{lo:.4f}; {hi:.4f}]")
# k escolhido dentro das partições de treino (F1), avaliado na partição de teste
tot = np.zeros(3, int); escolhas = []
for rep in range(100):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(pres, sem):
        f1 = {}
        for k, fl in FL.items():
            tp, fp, fn = met(fl.iloc[tr] > 0, sem.iloc[tr]); f1[k] = 2 * tp / max(1, 2 * tp + fp + fn)
        kb = max(f1, key=f1.get); escolhas.append(kb)
        tot += met(FL[kb].iloc[te] > 0, sem.iloc[te])
tp, fp, fn = tot
print(f"k escolhido no treino: {pd.Series(escolhas).value_counts(normalize=True).round(3).to_dict()}")
print(f"fora da amostra (100 repetições x 5 partições): precisão={tp/(tp+fp):.4f} sensibilidade={tp/(tp+fn):.4f}")
