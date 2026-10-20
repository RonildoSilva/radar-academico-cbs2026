"""Concorrência de disciplinas no nível do histórico acadêmico: grafo ingênuo vs. grafo por semestre vs. ordem parcial (POWL)."""
import os, sys, time, pandas as pd, numpy as np
ROOT = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
GERADOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerador_artigo")
L = pd.read_csv(f"{GERADOR}/saida/Academic_Mock_Log_Multiyear.csv", parse_dates=["time:timestamp"])
from importlib.util import spec_from_file_location, module_from_spec
sp = spec_from_file_location("cfg", f"{GERADOR}/config.py"); cfg = module_from_spec(sp); sp.loader.exec_module(cfg)
per = {d:k for k,v in cfg.MATRIZ.items() for d in v}
# o gerador grava o semestre cursado; dentro do semestre, a ordem das linhas é arbitrária
L = L.sort_values(["case:concept:name","semestre"], kind="stable").copy()
def cm(E): 
    V = {a for e in E for a in e}; return len(E)-len(V)+1, len(V), len(E)
# A. ingênuo: sucessão direta por timestamp
L["prox"] = L.groupby("case:concept:name")["concept:name"].shift(-1)
L["sem_prox"] = L.groupby("case:concept:name")["semestre"].shift(-1)
T = L.dropna(subset=["prox"]); T = T[T["concept:name"]!=T["prox"]]
intra = (T["semestre"] == T["sem_prox"]).mean()
EA = set(zip(T["concept:name"], T["prox"]))
confA = (T["concept:name"].map(per) <= T["prox"].map(per)).mean()
print(f"A ingênuo: transições={len(T)} intra-semestre={intra:.4f} CM,V,E={cm(EA)} conformidade={confA:.4f}")
# B. grafo por semestre: arestas de toda disciplina do semestre t para toda do semestre t+1
pairs=[]
for c,g in L.groupby("case:concept:name"):
    S = g.groupby("semestre")["concept:name"].apply(list).sort_index()
    ks = list(S.index)
    for a,b in zip(ks[:-1], ks[1:]):
        if b==a+1:
            for x in S[a]:
                for y in S[b]: pairs.append((x,y))
P = pd.DataFrame(pairs, columns=["a","b"])
EB = set(map(tuple, P.values))
confB = (P.a.map(per) < P.b.map(per)).mean()
confB_eq = (P.a.map(per) <= P.b.map(per)).mean()
print(f"B por semestre: pares={len(P)} CM,V,E={cm(EB)} conformidade estrita={confB:.4f} (<=: {confB_eq:.4f})")
# C. POWL (ordem parcial) no pm4py
import pm4py
log = L.rename(columns={"case:concept:name":"case:concept:name"})[["case:concept:name","concept:name","time:timestamp"]]
log = pm4py.format_dataframe(log, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
def resumo(pw):
    s=str(pw); return f"len={len(s)} PO={s.count('PO=')} X={s.count('X(')} loops={s.count('*(')} tau={s.count('tau')}"
# conjuntos: log completo; só concluintes (trajetória completa)
A = pd.read_csv(f"{GERADOR}/saida/Academic_Mock_Alunos.csv")
concl = set(A.loc[A["desfecho"]=="Concluinte","case:concept:name"])  # integralizaram a matriz
for nome, lg in [("completo", log), ("concluintes", log[log["case:concept:name"].isin(concl)])]:
    for fw in [0.0, 0.2]:
        t0=time.time()
        try:
            pw = pm4py.discover_powl(lg, filtering_weight_factor=fw)
            print(f"C POWL {nome} filtro={fw}: {time.time()-t0:.1f}s {resumo(pw)}")
            if nome=="concluintes" and fw==0.2: print(str(pw)[:2500])
        except Exception as e: print("POWL falhou:", nome, fw, repr(e))

print("== D. Matriz como modelo de referência em POWL: períodos em sequência, disciplinas concorrentes")
import pm4py.objects.powl.obj as po
blocos = [po.StrictPartialOrder(nodes=[po.Transition(label=d) for d in cfg.MATRIZ[k]]) for k in sorted(cfg.MATRIZ)]
ref = po.Sequence(nodes=blocos)
net, im, fm = pm4py.convert_to_petri_net(ref)
print("rede:", len(net.places), "lugares", len(net.transitions), "transições")
for nome, lg in [("todos", log), ("concluintes", log[log["case:concept:name"].isin(concl)])]:
    t0=time.time()
    f = pm4py.fitness_token_based_replay(lg, net, im, fm)
    print(f"replay {nome}: {time.time()-t0:.1f}s", {k: round(v,4) for k,v in f.items()})
# mesma rede, mas com a ordem dentro do semestre forçada (modelo sequencial estrito) para contraste
seq = po.Sequence(nodes=[po.Transition(label=d) for k in sorted(cfg.MATRIZ) for d in cfg.MATRIZ[k]])
n2,i2,f2 = pm4py.convert_to_petri_net(seq)
lgc = log[log["case:concept:name"].isin(concl)]
print("replay concluintes, modelo totalmente sequencial:", {k: round(v,4) for k,v in pm4py.fitness_token_based_replay(lgc, n2, i2, f2).items()})
