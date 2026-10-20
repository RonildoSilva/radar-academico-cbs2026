"""Experimento controlado: conformidade POWL (ordem parcial) vs. sequência estrita, variando a taxa de trocas do gerador.
Sem retenção (nenhuma reprovação, trancamento ou evasão), só a taxa de trocas varia. Concluintes = históricos que
integralizaram a matriz. Não grava nada em gerador_artigo/saida."""
import os, sys, io, contextlib, pandas as pd
from datetime import datetime
ROOT = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerador_artigo"))
import pm4py, pm4py.objects.powl.obj as po
import config as cfg
import mock_data_generator as gen
blocos = [po.StrictPartialOrder(nodes=[po.Transition(label=d) for d in cfg.MATRIZ[k]]) for k in sorted(cfg.MATRIZ)]
netP, imP, fmP = pm4py.convert_to_petri_net(po.Sequence(nodes=blocos))
netS, imS, fmS = pm4py.convert_to_petri_net(po.Sequence(nodes=[po.Transition(label=d) for k in sorted(cfg.MATRIZ) for d in cfg.MATRIZ[k]]))
ing, _ = gen.carregar_inep()
corte = datetime(max(ing), 12, 31)
rows = []
for nivel in [0.0, 0.10, 0.25, 0.50]:
    L, A = gen.simular(ing, corte, nivel=nivel, retencao=False)
    concl = A.loc[A["desfecho"] == "Concluinte", "case:concept:name"]
    L = L[L["case:concept:name"].isin(concl)].sort_values(["case:concept:name", "semestre"])
    lg = pm4py.format_dataframe(L[["case:concept:name", "concept:name", "time:timestamp"]].copy(),
        case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
    with contextlib.redirect_stderr(io.StringIO()):
        fP = pm4py.fitness_token_based_replay(lg, netP, imP, fmP); fS = pm4py.fitness_token_based_replay(lg, netS, imS, fmS)
    esperado = (1 - nivel) ** 7
    rows.append({"nivel_trocas": nivel, "concluintes": len(concl), "fitness_ordem_parcial": fP["log_fitness"], "aderentes_ordem_parcial_%": fP["percentage_of_fitting_traces"],
                 "fitness_sequencial": fS["log_fitness"], "aderentes_sequencial_%": fS["percentage_of_fitting_traces"], "sem_troca_esperado_%": 100 * esperado})
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False))
R.to_csv(os.path.join(os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__))), "powl_controle.csv"), index=False)
