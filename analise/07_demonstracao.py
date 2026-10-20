"""Demonstração do protótipo: nível do curso (INEP real) e verificação do gerador sintético (nível do histórico acadêmico).
Executar depois de gerador_artigo/mock_data_generator.py (o run_all.sh já faz isso).
Usa só volumes agregados e públicos do Censo da Educação Superior (data/processed/)."""
import os, sys, io, contextlib, pandas as pd, numpy as np
ROOT = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
GERADOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gerador_artigo")
sys.path.insert(0, GERADOR)
import config as cfg
from calibracao import anos_validos, razao_conclusao, intervalo_wilson, janelas_moveis, series_sinteticas

inep = pd.read_csv(f"{ROOT}/data/processed/cursos_sistemas_informacao_historico.csv", sep=";")
A = inep.groupby("NU_ANO_CENSO")[["QT_VG_TOTAL", "QT_INSCRITO_TOTAL", "QT_ING", "QT_MAT", "QT_CONC"]].sum()
A[f"razao_t-{cfg.DEFASAGEM_CONCLUSAO}"] = A.QT_CONC / A.QT_ING.shift(cfg.DEFASAGEM_CONCLUSAO)
print(A.to_string())
A.to_csv(os.path.join(os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__))), "inep_macro.csv"))
ing_r, conc_r = A.QT_ING.astype(int).to_dict(), A.QT_CONC.astype(int).to_dict()
anos = anos_validos(ing_r, conc_r)
r_real, num, den = razao_conclusao(ing_r, conc_r, anos)
print(f"razão ano a ano: máx {A.iloc[:, -1].max():.4f} (instável em volumes pequenos)")

L = pd.read_csv(f"{GERADOR}/saida/Academic_Mock_Log_Multiyear.csv", parse_dates=["time:timestamp"])
S = pd.read_csv(f"{GERADOR}/saida/Academic_Mock_Alunos.csv")
ing_s, conc_s = series_sinteticas(S)
r_sim, _, _ = razao_conclusao(ing_s, conc_s, anos)
print("\n1. Volume: anos de ingresso simulados sem âncora:", sorted(set(ing_s) - set(ing_r)),
      "| volumes iguais aos do INEP:", all(ing_s[a] == ing_r[a] for a in ing_r))
ic = intervalo_wilson(num, den)
print(f"2/3. Razão defasada ({cfg.DEFASAGEM_CONCLUSAO} anos, {anos[0]}-{anos[-1]}): sintética {r_sim:.4f} | INEP {r_real:.4f} "
      f"(IC95% {ic[0]:.4f}-{ic[1]:.4f}) | diferença {abs(r_sim - r_real) * 100:.2f} p.p.")
jr, js = janelas_moveis(ing_r, conc_r, anos), janelas_moveis(ing_s, conc_s, anos)
print("   janelas móveis real:", jr.razao.round(3).tolist(), "\n   janelas móveis sint.:", js.razao.round(3).tolist())
print("   máximo das janelas reais:", round(jr.razao.max(), 4))
c = S[S.desfecho == "Concluinte"]
print("4. Concluintes por ano de ingresso:", c.groupby("ano_ingresso").size().to_dict(),
      f"| menor intervalo ingresso-conclusão: {(c.ano_desfecho - c.ano_ingresso).min():.0f} anos | todos integralizaram:",
      bool((c.disciplinas_aprovadas == sum(len(v) for v in cfg.MATRIZ.values())).all()))
print("   desfechos:", S.desfecho.value_counts().to_dict(), f"| taxa de conclusão, ingressantes até {max(ing_r) - cfg.DEFASAGEM_CONCLUSAO - 1}:",
      round((S[S.ano_ingresso <= max(ing_r) - cfg.DEFASAGEM_CONCLUSAO - 1].desfecho == "Concluinte").mean(), 4))
ult = L[L.semestre == L.groupby("case:concept:name").semestre.transform("max")].drop_duplicates("case:concept:name")
print(f"5. Atraso crítico (alunos com atraso > 2 no último semestre): {(ult.lag_academico > 2).mean():.4f} | "
      f"alunos com algum atraso: {(ult.lag_academico >= 1).mean():.4f} | trancamentos: {S.trancamentos.sum()} | "
      f"semestres até concluir: média {c.semestres_cursados.mean():.2f}, máx {c.semestres_cursados.max()}")
# 6. Concorrência: transições ingênuas intra-semestre vs. grafo por semestre e conformidade por replay de tokens
Ls = L.sort_values(["case:concept:name", "time:timestamp", "semestre"], kind="stable").copy()
Ls["prox"] = Ls.groupby("case:concept:name")["concept:name"].shift(-1)
Ls["sem_prox"] = Ls.groupby("case:concept:name")["semestre"].shift(-1)
T = Ls.dropna(subset=["prox"])
print(f"6. Grafo ingênuo por instante: {(T.semestre == T.sem_prox).mean():.4f} das transições dentro do mesmo semestre; "
      "grafo por semestre: 0 por construção")
import pm4py, pm4py.objects.powl.obj as po
net, im, fm = pm4py.convert_to_petri_net(po.Sequence(nodes=[po.StrictPartialOrder(nodes=[po.Transition(label=d) for d in cfg.MATRIZ[k]]) for k in sorted(cfg.MATRIZ)]))
lg = pm4py.format_dataframe(L.sort_values(["case:concept:name", "semestre"])[["case:concept:name", "concept:name", "time:timestamp"]].copy(),
                            case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
with contextlib.redirect_stderr(io.StringIO()):
    f = pm4py.fitness_token_based_replay(lg, net, im, fm)
print(f"   conformidade em ordem parcial (replay): {f['log_fitness']:.4f} | históricos aderentes: {f['percentage_of_fitting_traces']:.2f}%")
print(f"eventos {len(L)} alunos {L['case:concept:name'].nunique()} disciplinas {L['concept:name'].nunique()}")
