"""Gerador de históricos acadêmicos sintéticos calibrados pelo INEP (nível meso do radar).

Cópia congelada do gerador usado no artigo CBS 2026 (commit 8c79a90 da branch artigo-cbs-2026).
O motor do projeto em pipelines/synthetic_engine/ evoluiu de forma independente; esta versão
existe para que os números do artigo (scripts 07, 08 e 10) continuem reproduzíveis.

- Volume: cada ano de ingresso com dado do Censo recebe exatamente QT_ING alunos. Anos sem âncora não são
  simulados (sem o CSV do INEP, o modo de desenvolvimento gera turmas marcadas como não ancoradas).
- Desfecho: a habilidade média é ajustada por bisseção até que a razão de conclusão defasada do
  sintético iguale a do INEP (ver calibracao.py), calculada da mesma forma nas duas séries.
- Conclusão: integralização da matriz (todas as disciplinas aprovadas), e não uma disciplina isolada.
- Progressão: reprovações viram pendências, a carga por semestre é limitada, há trancamentos,
  evasão e desligamento por prazo, de modo que o atraso curricular emerge da simulação.
- Concorrência: as disciplinas de um semestre recebem o mesmo instante (fechamento do semestre)
  e a coluna `semestre` permite ordenar o registro de eventos por semestre, e não por disciplina.

Saídas: saida/Academic_Mock_Log_Multiyear.csv (eventos) e saida/Academic_Mock_Alunos.csv (desfechos),
nesta pasta, para não sobrescrever os artefatos do motor do projeto em data/synthetic.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as cfg
from calibracao import anos_validos, razao_conclusao, intervalo_wilson, series_sinteticas

NIVEL_SPAGHETTI = cfg.NIVEL_SPAGHETTI  # alterável por quem importa o módulo (ver analise/10_powl_controle.py)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAIDA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida")
DATA_INEP_PATH = os.path.join(ROOT, "data", "processed", "cursos_sistemas_informacao_historico.csv")
TOTAL_DISCIPLINAS = sum(len(v) for v in cfg.MATRIZ.values())
MEDIA_POR_PERIODO = TOTAL_DISCIPLINAS / len(cfg.MATRIZ)
ESPERADAS_ATE = np.cumsum([0] + [len(cfg.MATRIZ[p]) for p in sorted(cfg.MATRIZ)])  # índice: semestres cursados
ADIANTAVEIS = [d for d in cfg.PERIODO if d not in cfg.PRE_REQUISITOS]


def carregar_inep(path=DATA_INEP_PATH):
    """Ingressantes e concluintes por ano do Censo, ou (None, None) se o CSV não existir."""
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path, sep=";")
    serie = df.groupby("NU_ANO_CENSO")[["QT_ING", "QT_CONC"]].sum()
    return ({int(k): int(v) for k, v in serie["QT_ING"].items()},
            {int(k): int(v) for k, v in serie["QT_CONC"].items()})


def gerar_perfil_aluno(ano_ingresso, rng, ajuste):
    trabalha = int(rng.random() < cfg.P_TRABALHA)
    viaja = int(rng.random() < cfg.P_DESLOCAMENTO)
    tipo_ingresso = rng.choice(["Ampla", "Cota Renda", "Cota Escola Pública"], p=[0.5, 0.25, 0.25])
    tem_bolsa = int(tipo_ingresso != "Ampla" and rng.random() < 0.8)
    responsavel_familiar = int(rng.random() < 0.25)
    idade_ingresso = int(rng.integers(17, 42))
    intervalo_em = 0 if idade_ingresso < 20 else (idade_ingresso - 18 - int(rng.integers(0, 3)))

    # Nota ENEM (Garcia, 2020)
    nota_enem = np.clip(rng.normal(600, 80), 400, 980)
    # Cursou na Pandemia (Leonel Jr, 2024)
    cursou_na_pandemia = int(ano_ingresso in [2020, 2021, 2022])
    distancia = rng.uniform(20, 65) if viaja else rng.uniform(1, 12)

    # `ajuste` é o deslocamento da habilidade média, obtido pela calibração de desfecho
    habilidade_base = rng.normal(8.0 + ajuste, 0.8)

    return {
        "ano_ingresso": ano_ingresso, "tipo_ingresso": tipo_ingresso, "distancia": round(distancia, 1),
        "trabalha": trabalha, "viaja": viaja, "tem_bolsa": tem_bolsa, "responsavel_familiar": responsavel_familiar,
        "idade_ingresso": idade_ingresso, "intervalo_em": max(0, intervalo_em),
        "habilidade": habilidade_base, "nota_enem": round(nota_enem, 2), "cursou_na_pandemia": cursou_na_pandemia
    }


def simular_desempenho(disciplina, perfil_aluno, rng):
    base = perfil_aluno["habilidade"]

    if perfil_aluno["trabalha"]: base -= 0.5
    if perfil_aluno["intervalo_em"] > 3 and disciplina in ["Matemática", "Lógica de Programação"]: base -= 0.8
    if perfil_aluno["cursou_na_pandemia"]: base -= 0.4  # Impacto da pandemia no foco acadêmico
    if disciplina in cfg.GARGALOS: base -= 0.6

    nota = np.clip(rng.normal(base, 0.7), 0, 10)
    freq_base = 90 - (perfil_aluno["distancia"] / 5) - (5 if perfil_aluno["trabalha"] else 0)
    frequencia = np.clip(rng.normal(freq_base, 5), 40, 100)

    status = "Aprovado" if (nota >= 7.0 and frequencia >= 75) else "Reprovado"
    return round(float(nota), 1), round(float(frequencia), 1), status


def data_semestre(ano_ingresso, s):
    """Fechamento do s-ésimo semestre desde o ingresso (s = 1 é o primeiro semestre do ano de ingresso)."""
    ano = ano_ingresso + (s - 1) // 2
    return (datetime(ano, 6, 30) if (s - 1) % 2 == 0 else datetime(ano, 12, 15)), f"{ano}.{(s - 1) % 2 + 1}"


def simular_aluno(aluno_id, ano, perfil, rng, data_corte, nivel, retencao, eventos):
    """Simula a trajetória até a conclusão, a evasão, o desligamento ou a data de corte (aluno ativo)."""
    aprovadas, pendentes = set(), []
    periodo = 1                      # próximo período da matriz a entrar na matrícula
    s = cursados = trancamentos = 0
    media_anterior = None
    desfecho, data_desfecho = "Ativo", None

    elegivel = lambda d: cfg.PRE_REQUISITOS.get(d) is None or cfg.PRE_REQUISITOS[d] in aprovadas

    while True:
        s += 1
        data, letivo = data_semestre(ano, s)
        if data > data_corte:
            break
        if cursados >= cfg.PRAZO_MAXIMO:
            desfecho, data_desfecho = "Desligado", data
            break

        if retencao and cursados > 0:
            p_tranc = cfg.P_TRANCAMENTO + (cfg.P_TRANCAMENTO_TRABALHA if perfil["trabalha"] else 0)
            if rng.random() < p_tranc:
                trancamentos += 1
                continue

        # Matrícula: pendências elegíveis primeiro, depois o próximo período até a carga máxima
        matricula = [d for d in pendentes if elegivel(d)][:cfg.CARGA_MAXIMA]
        if periodo <= len(cfg.MATRIZ):
            novas = [d for d in cfg.MATRIZ[periodo] if d not in aprovadas and d not in pendentes]
            futuras = [d for d in ADIANTAVEIS if cfg.PERIODO[d] > periodo and d not in aprovadas and d not in pendentes]
            if novas and futuras and rng.random() < nivel:
                # Adiantamento: troca uma disciplina do período por uma futura; a preterida fica pendente
                origem = novas[int(rng.integers(len(novas)))]
                novas.remove(origem)
                pendentes.append(origem)
                novas.append(futuras[int(rng.integers(len(futuras)))])
            for d in novas:
                if len(matricula) < cfg.CARGA_MAXIMA and elegivel(d):
                    matricula.append(d)
                else:
                    pendentes.append(d)
            periodo += 1
        pendentes = [d for d in pendentes if d not in matricula]
        if not matricula:
            continue
        cursados += 1

        registros, notas, reprovas = [], [], 0
        for d in matricula:
            nota, freq, status = simular_desempenho(d, perfil, rng)
            if not retencao:
                status = "Aprovado"
            if status == "Aprovado":
                aprovadas.add(d)
            else:
                pendentes.append(d)
                reprovas += 1
            notas.append(nota)
            registros.append((d, nota, freq, status))

        media = float(np.mean(notas))
        esperadas = ESPERADAS_ATE[min(s, len(cfg.MATRIZ))]
        atraso = max(0, int(np.floor((esperadas - len(aprovadas)) / MEDIA_POR_PERIODO + 1e-9)))
        desordenada = int(len({cfg.PERIODO[d] for d in matricula}) > 1)
        # Disciplinas do semestre são concorrentes: a ordem de gravação é arbitrária, não a da matriz
        for j in rng.permutation(len(registros)):
            d, nota, freq, status = registros[j]
            eventos.append({
                "case:concept:name": aluno_id, "concept:name": d, "time:timestamp": data,
                "semestre": s, "semestre_letivo": letivo, "periodo_disciplina": cfg.PERIODO[d],
                "nota_final": nota, "frequencia": freq, "status": status,
                "tipo_ingresso": perfil["tipo_ingresso"], "idade_ingresso": perfil["idade_ingresso"],
                "intervalo_em": perfil["intervalo_em"], "trabalha": perfil["trabalha"],
                "bolsa": perfil["tem_bolsa"], "distancia_km": perfil["distancia"],
                "delta_cr": round(media - media_anterior, 2) if media_anterior is not None else 0.0,
                "lag_academico": atraso,
                "ano_ingresso": ano, "nota_enem": perfil["nota_enem"],
                "cursou_na_pandemia": perfil["cursou_na_pandemia"],
                "matricula_desordenada": desordenada,
                "responsavel_familiar": perfil["responsavel_familiar"],
            })
        media_anterior = media

        if len(aprovadas) == TOTAL_DISCIPLINAS:
            desfecho, data_desfecho = "Concluinte", data
            break
        if retencao:
            p_evasao = min(cfg.P_EVASAO_MAX, cfg.P_EVASAO_BASE + cfg.P_EVASAO_POR_REPROVA * reprovas)
            if rng.random() < p_evasao:
                desfecho, data_desfecho = "Evadido", data
                break

    return {
        "case:concept:name": aluno_id, "ano_ingresso": ano, "desfecho": desfecho,
        "ano_desfecho": data_desfecho.year if data_desfecho else np.nan,
        "semestres_cursados": cursados, "trancamentos": trancamentos,
        "disciplinas_aprovadas": len(aprovadas),
        "atraso_final": eventos[-1]["lag_academico"] if cursados and eventos[-1]["case:concept:name"] == aluno_id else 0,
        "trabalha": perfil["trabalha"], "bolsa": perfil["tem_bolsa"], "distancia_km": perfil["distancia"],
    }


def simular(volumes, data_corte, ajuste=0.0, nivel=None, retencao=True, semente=cfg.SEMENTE):
    """Simula todas as turmas de ingresso. Cada aluno tem sua própria semente, derivada de (semente, ano, índice),
    de modo que as chamadas da calibração usam números aleatórios comuns."""
    nivel = NIVEL_SPAGHETTI if nivel is None else nivel
    eventos, alunos = [], []
    for ano in sorted(volumes):
        for i in range(volumes[ano]):
            rng = np.random.default_rng([semente, ano, i])
            perfil = gerar_perfil_aluno(ano, rng, ajuste)
            alunos.append(simular_aluno(f"ALU_{ano}_{i:03d}", ano, perfil, rng, data_corte, nivel, retencao, eventos))
    return pd.DataFrame(eventos), pd.DataFrame(alunos)


def calibrar_desfecho(ing_inep, conc_inep, data_corte, iteracoes=30):
    """Bisseção do deslocamento de habilidade até a razão de conclusão sintética igualar a do INEP."""
    anos = anos_validos(ing_inep, conc_inep)
    alvo, _, _ = razao_conclusao(ing_inep, conc_inep, anos)
    razao = lambda ajuste: razao_conclusao(*series_sinteticas(simular(ing_inep, data_corte, ajuste)[1]), anos)[0]
    lo, hi = -3.0, 3.0
    for _ in range(iteracoes):
        meio = (lo + hi) / 2
        if razao(meio) < alvo:
            lo = meio
        else:
            hi = meio
    return (lo + hi) / 2, alvo, anos


def gerar_log_academico(saida_dir=SAIDA_DIR, nivel=None, retencao=True, verbose=True):
    log = print if verbose else (lambda *a, **k: None)
    ing_inep, conc_inep = carregar_inep()
    if ing_inep:
        volumes, ancorado = ing_inep, True
        data_corte = datetime(max(ing_inep), 12, 31)
        log(f">>> Volumes do INEP: {len(volumes)} anos de ingresso ({min(volumes)} a {max(volumes)}). Anos sem Censo não são simulados.")
    else:
        volumes, ancorado = {a: cfg.NUM_ALUNOS_POR_TURMA for a in cfg.ANOS_SEM_INEP}, False
        data_corte = datetime(max(volumes), 12, 31)
        log(f">>> Aviso: {DATA_INEP_PATH} não encontrado. Modo de desenvolvimento: turmas NÃO ancoradas no Censo.")

    ajuste = 0.0
    if ancorado and retencao:
        ajuste, alvo, anos = calibrar_desfecho(ing_inep, conc_inep, data_corte)
        log(f">>> Calibração de desfecho: ajuste de habilidade {ajuste:+.3f}")

    df, alunos = simular(volumes, data_corte, ajuste, nivel, retencao)
    df["ingresso_ancorado"] = int(ancorado)
    alunos["ingresso_ancorado"] = int(ancorado)

    os.makedirs(saida_dir, exist_ok=True)
    df.to_csv(os.path.join(saida_dir, "Academic_Mock_Log_Multiyear.csv"), index=False)
    alunos.to_csv(os.path.join(saida_dir, "Academic_Mock_Alunos.csv"), index=False)

    log(f"\n>>> Geração concluída: {len(alunos)} alunos, {len(df)} eventos (corte em {data_corte:%d/%m/%Y}).")
    log(">>> Desfechos:", alunos["desfecho"].value_counts().to_dict())
    if ancorado and retencao:
        r_sim, _, _ = razao_conclusao(*series_sinteticas(alunos), anos)
        _, num, den = razao_conclusao(ing_inep, conc_inep, anos)
        ic = intervalo_wilson(num, den)
        log(f">>> Razão de conclusão defasada ({cfg.DEFASAGEM_CONCLUSAO} anos, {anos[0]} a {anos[-1]}): "
            f"sintética {r_sim:.2%} | INEP {alvo:.2%} (IC95% {ic[0]:.2%} a {ic[1]:.2%})")
    return df, alunos


if __name__ == "__main__":
    gerar_log_academico()
