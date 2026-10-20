import os, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__)))
M = pd.read_pickle(f"{OUT}/metrics.pkl"); F = pd.read_pickle(f"{OUT}/final.pkl")
D = os.environ.get("EPM_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "epm", "EPM Dataset 2", "Data"))
logs = None


# Variância (Levene) alto vs baixo por nível
T = M.merge(F[["student","final_best"]], on="student")
mu = F.final_best.mean()
print("== Levene: dispersão CM, alto vs baixo (n=0)")
for c in ["CM_L1","CM_L2","CM_L3"]:
    hi=T[T.final_best>mu]; lo=T[T.final_best<mu]
    L=stats.levene(hi[c],lo[c]); print(f"{c}: dp alto={hi[c].std():.2f} dp baixo={lo[c].std():.2f} p={L.pvalue:.4f} | CV alto={hi[c].std()/hi[c].mean():.3f} baixo={lo[c].std()/lo[c].mean():.3f}")

def features(upto):
    E = M[M.session<=upto]
    agg = E.groupby("student").agg(presenca=("session","nunique"), eventos=("eventos","mean"), tempo=("tempo_tarefa_min","mean"),
        CM_L1=("CM_L1","mean"), CM_L2=("CM_L2","mean"), CM_L3=("CM_L3","mean"), pct_fora=("pct_fora","mean"), entropia=("entropia_L3","mean"),
        t_estudo=("t_Estudo","mean"), t_simul=("t_Simulação","mean"))
    ig = E[E.session>=2].groupby("student").ig_norm.mean().rename("nota_interm")
    X = pd.DataFrame(index=range(1,116)).join(agg).join(ig)
    X["presenca"] = X.presenca.fillna(0)
    return X.fillna(X.median())
SETS = {"Presença":["presenca"], "Volume":["presenca","eventos","tempo"],
        "Processo (CM + fora da tarefa + entropia)":["presenca","CM_L1","CM_L2","CM_L3","pct_fora","entropia","t_estudo","t_simul"],
        "Notas intermediárias":["nota_interm"],
        "Processo + notas":["presenca","CM_L1","CM_L2","CM_L3","pct_fora","entropia","t_estudo","t_simul","nota_interm"]}
LIMIAR = int(os.environ.get("LIMIAR_APROVACAO", 70))
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
models = {"RL": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
          "RF": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=1)}
res=[]
for upto in [3,6]:
    X = features(upto)
    fb = F.set_index("student").final_best
    # Alvo 1: não realizou exame final (desengajamento) — todos os 115
    y1 = (~X.index.isin(F.student)).astype(int)
    # Alvo 2: entre quem fez exame, nota >= LIMIAR
    idx = [i for i in X.index if i in fb.index]
    y2 = (fb.loc[idx] >= LIMIAR).astype(int).values
    for tgt, XX, yy in [("Não realizou o exame", X, y1), (f"Nota final >= {LIMIAR}", X.loc[idx], y2)]:
        for sname, cols in SETS.items():
            if upto==3 and sname.startswith("Notas") and False: pass
            for mname, mdl in models.items():
                if sname=="Presença" and mname=="RF": continue
                s = cross_val_score(mdl, XX[cols], yy, cv=cv, scoring="roc_auc", n_jobs=-1)
                res.append({"janela":f"S1–S{upto}","alvo":tgt,"conjunto":sname,"modelo":mname,"AUC":s.mean(),"dp":s.std(),"n":len(yy),"positivos":int(yy.sum())})
R = pd.DataFrame(res)
pd.set_option("display.width",200)
print(R.round(3).to_string())
R.to_csv(f"{OUT}/early_warning.csv", index=False)
# Coeficientes RL para processo, janela 3, alvo nota
X = features(3); fb = F.set_index("student").final_best; idx=[i for i in X.index if i in fb.index]
cols = SETS["Processo + notas"]; m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X.loc[idx,cols], (fb.loc[idx]>=LIMIAR).astype(int))
print(dict(zip(cols, m[-1].coef_[0].round(3))))
print("taxa nota>=LIMIAR:", (fb>=LIMIAR).mean().round(3))
