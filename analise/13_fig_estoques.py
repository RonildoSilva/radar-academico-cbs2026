"""Figura 1: ciclo de gestão da permanência em estoques e fluxos (seção 2.1).

Não usa dados. Uso: python 13_fig_estoques.py ../figuras
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Ellipse
FIG = sys.argv[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
NAVY, ORANGE, GRAY = "#003350", "#F37021", "#8A8F98"

fig, ax = plt.subplots(figsize=(6.5, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0.2, 6.2); ax.axis("off")

def estoque(x, y, t, w=1.6, h=0.9):
    ax.add_patch(Rectangle((x, y), w, h, fc=NAVY, ec="none"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=8.5)
def nuvem(x, y):
    for dx, dy in [(-0.12, 0), (0.1, 0.06), (0.02, -0.08)]:
        ax.add_patch(Ellipse((x + dx, y + dy), 0.32, 0.24, fc="white", ec=GRAY, lw=0.9))
def fluxo(x1, y1, x2, y2, rotulo, lado=(0, 0.2), ha="center"):
    """Cano duplo com válvula no meio, como na notação de estoques e fluxos."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="simple,head_width=0.9,head_length=0.7,tail_width=0.35", fc="white", ec=NAVY, lw=0.9))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2; s = 0.09
    if y1 == y2: tri = [[(mx - s, my + s), (mx + s, my - s), (mx + s, my + s), (mx - s, my - s)]]
    else: tri = [[(mx - s, my + s), (mx + s, my - s), (mx - s, my - s), (mx + s, my + s)]]
    ax.add_patch(Polygon(tri[0], closed=True, fc=NAVY, ec=NAVY))
    ax.text(mx + lado[0], my + lado[1], rotulo, ha=ha, va="center", fontsize=7.2, color="#222")
    return mx, my
def caixa(x, y, w, h, t, c, tc="white", fs=7.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=c, ec="none"))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=tc, fontsize=fs)
def info(x1, y1, x2, y2, c=GRAY, rad=0.0, lw=1.1, ls="--"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))
def atraso(x, y, ang=0):
    for d in (-0.05, 0.05):
        ax.plot([x + d - 0.06, x + d + 0.06], [y - 0.12, y + 0.12], color="#222", lw=1.3)

# Estoques do sistema acadêmico
estoque(1.0, 3.9, "Ingressantes", w=1.5, h=0.8); estoque(3.5, 3.9, "Ativos em\nfluxo regular", w=1.5, h=0.8)
estoque(6.0, 3.9, "Concluintes", w=1.3, h=0.8)
estoque(1.0, 1.5, "Evadidos", w=1.5, h=0.8); estoque(3.5, 1.5, "Retidos", w=1.5, h=0.8)
nuvem(0.3, 4.3)
fluxo(0.47, 4.3, 0.98, 4.3, "ingresso", lado=(-0.25, 0.42))
fluxo(2.52, 4.3, 3.48, 4.3, "progressão", lado=(0, 0.28))
fluxo(5.02, 4.3, 5.98, 4.3, "conclusão", lado=(0, 0.28))
fluxo(3.9, 3.88, 3.9, 2.32, "retenção", lado=(-0.17, 0), ha="right")
vx, vy = fluxo(4.6, 2.32, 4.6, 3.88, "recuperação", lado=(0.12, 0.33), ha="left")
fluxo(1.75, 3.88, 1.75, 2.32, "evasão\nprecoce", lado=(0.17, 0), ha="left")
ex, ey = fluxo(3.48, 1.9, 2.52, 1.9, "evasão", lado=(0, 0.25))

# Canais de informação e decisão
caixa(7.9, 4.7, 2.0, 0.95, "Indicadores tradicionais\n(taxas de conclusão\ne de evasão)", GRAY, fs=7)
caixa(7.9, 2.65, 1.6, 1.15, "Radar acadêmico\n(trajetória de\npresença, notas\nparciais)", ORANGE, fs=7)
caixa(7.9, 0.35, 2.0, 0.95, "Coordenação:\ndecisão e ação\npedagógica", NAVY, fs=7)
info(6.65, 4.72, 7.87, 5.15, rad=-0.3)                          # concluintes e evadidos -> indicadores
atraso(7.15, 5.12); ax.text(7.0, 5.75, "atraso longo:\nsemestres a anos", ha="center", va="center", fontsize=6.6, color="#222")
info(5.04, 3.95, 7.87, 3.4, c=ORANGE, lw=1.3)                    # rastros dos ativos -> radar
atraso(7.1, 3.54); ax.text(7.0, 3.3, "atraso curto: semanas", ha="center", va="top", fontsize=6.6, color="#222")
info(9.72, 4.67, 9.72, 1.33, c=GRAY, lw=1.0)                     # indicadores -> decisão
info(8.7, 2.62, 8.7, 1.33, c=ORANGE, lw=1.3)                     # radar -> decisão
# ação pedagógica sobre as válvulas da evasão e da recuperação
ax.plot([7.87, 2.98, 2.98], [0.8, 0.8, 1.72], color=ORANGE, lw=1.3)
ax.annotate("", xy=(2.98, 1.78), xytext=(2.98, 1.5), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))
ax.plot([5.45, 5.45, 4.8], [0.8, 3.1, 3.1], color=ORANGE, lw=1.3)
ax.annotate("", xy=(4.74, 3.1), xytext=(4.95, 3.1), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3))
ax.text(4.2, 0.55, "tutoria, reforço, contato ativo", ha="center", va="center", fontsize=6.6, color=ORANGE)
ax.text(6.6, 1.05, "B", ha="center", va="center", fontsize=8.5, color=ORANGE, fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=ORANGE))
ax.text(6.78, 2.85, "★ ponto de alavancagem:\nestrutura do fluxo de\ninformação (MEADOWS, 2008)", ha="center", va="top",
        fontsize=6.4, color=ORANGE)
# laço de reforço da retenção (Tinto)
ax.text(5.75, 1.9, "R", ha="center", va="center", fontsize=8.5, color=NAVY, fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=NAVY))
ax.text(6.0, 1.9, "reprovação → atraso na turma →\nmenor integração (TINTO, 1975)\n→ nova retenção",
        ha="left", va="center", fontsize=6.2, color="#222")
fig.tight_layout(); fig.savefig(f"{FIG}/figura1_estoques_fluxos.png", dpi=300); plt.close(fig)
print("ok")
