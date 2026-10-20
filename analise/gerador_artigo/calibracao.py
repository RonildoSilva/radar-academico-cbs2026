"""Indicador de conclusão com defasagem, calculado da mesma forma no INEP e no dado sintético.

Razão agregada: soma dos concluintes do ano t dividida pela soma dos ingressantes do ano
t - defasagem, na janela de anos em que as duas séries existem. Somar antes de dividir evita a
instabilidade da razão ano a ano em volumes pequenos (que chega a passar de 100%).
"""
import numpy as np
import pandas as pd

from config import DEFASAGEM_CONCLUSAO, ANOS_AQUECIMENTO, JANELA_MOVEL


def anos_validos(ingressantes, concluintes, defasagem=DEFASAGEM_CONCLUSAO, aquecimento=ANOS_AQUECIMENTO):
    """Anos t do Censo com ingressantes em t - defasagem, descartando os primeiros anos da série."""
    if not ingressantes:
        return []
    inicio = min(ingressantes) + defasagem + aquecimento
    return [t for t in sorted(concluintes) if t >= inicio and (t - defasagem) in ingressantes]


def razao_conclusao(ingressantes, concluintes, anos, defasagem=DEFASAGEM_CONCLUSAO):
    """Retorna (razão, concluintes somados, ingressantes defasados somados) para a lista de anos."""
    num = sum(concluintes.get(t, 0) for t in anos)
    den = sum(ingressantes.get(t - defasagem, 0) for t in anos)
    return (num / den if den > 0 else np.nan), num, den


def intervalo_wilson(num, den, z=1.96):
    """Intervalo de 95% de Wilson, tratando a razão agregada como proporção."""
    if den <= 0:
        return (np.nan, np.nan)
    p = min(num / den, 1.0)
    centro = (p + z**2 / (2 * den)) / (1 + z**2 / den)
    margem = z * np.sqrt(p * (1 - p) / den + z**2 / (4 * den**2)) / (1 + z**2 / den)
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def janelas_moveis(ingressantes, concluintes, anos, janela=JANELA_MOVEL, defasagem=DEFASAGEM_CONCLUSAO):
    """Razão agregada em janelas móveis de `janela` anos, indexada pelo último ano da janela."""
    linhas = []
    for i in range(janela - 1, len(anos)):
        bloco = anos[i - janela + 1:i + 1]
        r, num, den = razao_conclusao(ingressantes, concluintes, bloco, defasagem)
        linhas.append({"ano": anos[i], "razao": r, "concluintes": num, "ingressantes_defasados": den})
    return pd.DataFrame(linhas)


def series_sinteticas(alunos):
    """Ingressantes por ano de ingresso e concluintes por ano de conclusão a partir da tabela de alunos."""
    ing = alunos.groupby("ano_ingresso").size().to_dict()
    conc = alunos[alunos["desfecho"] == "Concluinte"].groupby("ano_desfecho").size().to_dict()
    return {int(k): int(v) for k, v in ing.items()}, {int(k): int(v) for k, v in conc.items()}
