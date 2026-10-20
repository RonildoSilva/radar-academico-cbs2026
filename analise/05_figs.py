import os, sys, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = os.environ.get("EPM_OUT", os.path.dirname(os.path.abspath(__file__))); FIG = sys.argv[1]; os.makedirs(f"{FIG}/nao_usadas", exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.formatter.use_locale":False})
fmt = lambda v,_=None: f"{v:.2f}".replace(".",",")
M = pd.read_pickle(f"{OUT}/metrics.pkl"); F = pd.read_pickle(f"{OUT}/final.pkl")
NAVY, ORANGE, GRAY = "#003350", "#F37021", "#8A8F98"

# Figura 2: CM médio (L2) por sessão x nota intermediária normalizada
A = M.groupby("session").agg(CM=("CM_L2","mean"), ig=("ig_norm","mean"))
fig, ax1 = plt.subplots(figsize=(6.3,3.6)); ax2 = ax1.twinx()
ax1.plot(A.index, A.CM, "o-", color=NAVY, label="Complexidade ciclomática média (nível 2)")
ax2.plot(A.index[1:], A.ig[1:], "s--", color=ORANGE, label="Nota intermediária média (normalizada)")
ax1.set_xlabel("Sessão de laboratório"); ax1.set_ylabel("Complexidade ciclomática")
ax2.set_ylabel("Nota intermediária (0 a 1)")
ax1.yaxis.set_major_formatter(plt.FuncFormatter(fmt)); ax2.yaxis.set_major_formatter(plt.FuncFormatter(fmt))
ax1.set_xticks(range(1,7))
h1,l1 = ax1.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5,-0.22), ncol=2, fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/nao_usadas/figura_cm_por_sessao.png", dpi=300); plt.close(fig)

# Figura 3: CM L2 alto vs baixo por sessão (mu)
T = M.merge(F[["student","final_best"]], on="student"); mu = F.final_best.mean()
g = T.assign(grupo=np.where(T.final_best>mu,"Acima da média","Abaixo da média")).groupby(["session","grupo"]).CM_L2.agg(["mean","std","count"]).reset_index()
fig, ax = plt.subplots(figsize=(6.3,3.2)); w=0.38
for i,(gr,c) in enumerate([("Acima da média",NAVY),("Abaixo da média",ORANGE)]):
    d = g[g.grupo==gr]; se = d["std"]/np.sqrt(d["count"])
    ax.bar(d.session+(i-0.5)*w, d["mean"], w, yerr=1.96*se, color=c, capsize=3, label=f"Nota final {gr.lower()}")
ax.set_xlabel("Sessão de laboratório"); ax.set_ylabel("Complexidade ciclomática média")
ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt)); ax.set_xticks(range(1,7))
ax.legend(fontsize=8, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/nao_usadas/figura_cm_por_grupo.png", dpi=300); plt.close(fig)

# Figura 1: arquitetura sistêmica em três níveis
fig, ax = plt.subplots(figsize=(6.5,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,7.2); ax.axis("off")
def box(x,y,w,h,t,c,fs=8.5,tc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.15",fc=c,ec="none"))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=tc,fontsize=fs,wrap=True)
def arr(x1,y1,x2,y2,t="",c=GRAY,style="-|>"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle=style,color=c,lw=1.3))
    if t: ax.text((x1+x2)/2+0.1,(y1+y2)/2,t,fontsize=7,color="#333")
box(0.2,5.6,3.0,1.2,"Nível do curso\nCenso da Educação Superior\n(ingressantes, concluintes)",NAVY)
box(0.2,3.4,3.0,1.2,"Nível do histórico acadêmico\nregistros de matrícula\n(disciplina, período,\nsituação)",NAVY)
box(0.2,1.2,3.0,1.2,"Nível da atividade\nde aprendizagem\nrastros de interação\n(atividade, tempo, sessão)",NAVY)
box(4.0,1.2,2.4,5.6,"Camada analítica\n\nregistro de eventos\n(caso, atividade,\ninstante)\n\ngrafo de sucessão\ndireta\n\nconformidade com\na matriz\n\nmétricas: complexidade,\npresença, fora da tarefa\n\nmodelos de alerta",ORANGE,fs=8)
box(7.2,4.6,2.6,2.2,"Painel de decisão\nindicadores com nota\ntécnica e fila de\nintervenção",NAVY)
box(7.2,1.2,2.6,2.2,"Ação pedagógica\ntutoria, reforço,\nrevisão curricular,\ncontato ativo",NAVY)
for y in [6.2,4.0,1.8]: arr(3.25,y,3.95,y if y!=6.2 else 5.8)
arr(6.45,5.7,7.15,5.7); arr(8.5,4.55,8.5,3.45,"decisão")
ax.plot([8.5,8.5,1.7,1.7],[1.15,0.55,0.55,1.1],color=ORANGE,lw=1.5); ax.annotate("",xy=(1.7,1.15),xytext=(1.7,0.9),arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=1.5))
ax.text(1.9,0.15,"ciclo de retroalimentação: a ação altera a trajetória, que gera novos rastros",fontsize=7.5,color="#333")
ax.text(0.2,5.2,"calibração de volume e desfecho",fontsize=7,color="#333",style="italic")
fig.tight_layout(); fig.savefig(f"{FIG}/figura1_arquitetura.png", dpi=300); plt.close(fig)
print("ok")

# Figura 3: grafos de sucessão direta (nível 2) de dois estudantes na sessão 6.
# Escolha: no quartil superior e no inferior da nota final, o estudante cuja
# complexidade está mais próxima da mediana do seu quartil.
import re
from matplotlib.patches import FancyArrowPatch
log = pd.read_pickle(f"{OUT}/log.pkl")
SESS = 6
m6 = M[M.session == SESS].merge(F[["student", "final_best"]], on="student")
q1, q3 = m6.final_best.quantile([.25, .75])
def representante(d):
    return d.assign(dist=(d.CM_L2 - d.CM_L2.median()).abs()).sort_values(["dist", "student"]).iloc[0]
alto, baixo = representante(m6[m6.final_best >= q3]), representante(m6[m6.final_best <= q1])
COLS = ["Study_Es", "Study_Materials", "Deeds_Es", "Deeds", "Properties", "FSM_Es", "FSM_Related", "Diagram",
        "TextEditor_Es", "TextEditor", "Aulaweb"]
GRUPOS = [("Estudo", 0, 1), ("Construção", 2, 6), ("Simul.", 7, 7), ("Relato", 8, 9), ("Plat.", 10, 10)]
def pos(a):
    b = re.sub(r"_\d+_\d+$", "", a); mt = re.search(r"_(\d+)_(\d+)$", a)
    return COLS.index(b), (int(mt.group(2)) if mt else 0)
seqs = {}
for r in (alto, baixo):
    sq = log[(log.session == SESS) & (log.student == r.student) & log.in_task].sort_values(["start", "end"]).activity.tolist()
    seqs[r.student] = [a for i, a in enumerate(sq) if i == 0 or sq[i - 1] != a]
YMAX = max(pos(a)[1] for sq in seqs.values() for a in sq)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
for ax, r, tit in [(axes[0], alto, "Nota final no quartil superior"), (axes[1], baixo, "Nota final no quartil inferior")]:
    seq = seqs[r.student]; E = {}
    for a, b in zip(seq[:-1], seq[1:]): E[(a, b)] = E.get((a, b), 0) + 1
    V = sorted(set(seq)); P = {v: pos(v) for v in V}
    for k, (nome, x0, x1) in enumerate(GRUPOS):
        ax.axvspan(x0 - 0.45, x1 + 0.45, color="#EEF1F4" if k % 2 == 0 else "white", zorder=0)
        ax.text((x0 + x1) / 2, -0.85, nome, ha="center", va="bottom", fontsize=6.5)
    for (a, b), w in E.items():
        ax.add_patch(FancyArrowPatch(P[a], P[b], connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=5,
                                     lw=0.4 + 0.25 * min(w, 8), color=NAVY, alpha=0.35, shrinkA=4, shrinkB=4, zorder=2))
    for v, (x, y) in P.items(): ax.plot(x, y, "o", ms=5.5, color=ORANGE, zorder=3)
    ax.set_xlim(-0.6, len(COLS) - 0.4); ax.set_ylim(YMAX + 0.9, -0.9)
    ax.set_yticks(range(0, YMAX + 1)); ax.set_yticklabels(["sem exercício"] + [f"exercício {k}" for k in range(1, YMAX + 1)], fontsize=6.5)
    ax.set_xticks([]); ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)
    if ax is axes[1]: ax.set_yticklabels([])
    ax.set_title(f"{tit}: {len(seq)} atividades,\n{len(V)} vértices, {len(E)} arestas, CM = {len(E) - len(V) + 1}", fontsize=7.5, pad=12)
    print(tit, "estudante", int(r.student), "nota", r.final_best, "CM", len(E) - len(V) + 1, "eventos", int(r.eventos))
fig.tight_layout(); fig.savefig(f"{FIG}/figura3_grafos_estudantes.png", dpi=300); plt.close(fig)
print("figura 3 ok")
