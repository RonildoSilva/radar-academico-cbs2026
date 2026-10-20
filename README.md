# Radar acadêmico de evasão: reprodução do artigo do CBS 2026

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23516271.svg)](https://doi.org/10.5281/zenodo.23516271)

Código, dados derivados e figuras que reproduzem os resultados do artigo:

> SILVA, R. O. da; SILVA, F. B. da; VIANA, I. da S. Da complexidade do processo à trajetória de presença: sinais de mineração de processos educacionais para um radar de evasão no ensino superior. In: CONGRESSO BRASILEIRO DE SISTEMAS, 2026.

## Como reproduzir

Requer Python 3.12, `curl` e `unzip`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh                 # tudo, incluindo a busca em grade (cerca de 1 h em 8 núcleos)
PULAR_GRADE=1 ./run_all.sh   # sem a busca em grade da seção 4.4
```

O `run_all.sh` baixa o EPM, executa os scripts em ordem e grava as tabelas em `resultados/` e as figuras em `figuras/`. Todas as rotinas aleatórias usam semente 42.

## Conteúdo

| Caminho | O que é |
|---|---|
| `analise/01_build.py` a `03_stats.py` | Leitura do EPM, grafos de sucessão direta, complexidade ciclomática e correlações (seções 4.2 e 4.3) |
| `analise/04_early.py`, `06_robustez.py` | Avaliação preditiva da primeira execução, bootstrap, poder, trajetória de presença e regra de ruptura (seção 4.5) |
| `analise/12_parecer_cbs.py` | Tabela 1 da versão final: pipeline com imputação e padronização dentro da validação cruzada, testes de permutação e curva de antecipação (seção 4.4) |
| `analise/11_revisao.py` | Modelo nulo da complexidade, ponderação de classes, busca em grade aninhada e validação da regra de ruptura |
| `analise/09_componentes.py` | Componentes conexos dos grafos (seções 3.4 e 4.3) |
| `analise/gerador_artigo/` | Gerador sintético de históricos acadêmicos calibrado pelo INEP, na versão usada no artigo |
| `analise/07_demonstracao.py`, `08_concorrencia.py`, `10_powl_controle.py` | Demonstração do protótipo com a série do INEP e o gerador sintético (seções 4.1 e 4.6) |
| `analise/05_figs.py`, `13_fig_estoques.py` | Figuras 1 a 3 |
| `data/processed/cursos_sistemas_informacao_historico.csv` | Série agregada do Censo da Educação Superior (INEP, 2010 a 2024) do curso-alvo |
| `figuras/` | Figuras como publicadas no artigo |
| `scripts/baixar_epm.sh` | Download do EPM com verificação SHA-256 |

Em `04_early.py` e nas seções 3 e 4 de `06_robustez.py`, a imputação pela mediana é feita antes da validação cruzada. Os números da Tabela 1, da curva de antecipação e do texto da seção 4.4 vêm de `12_parecer_cbs.py`, que substitui esses resultados.

## Dados

- **Educational Process Mining (EPM)**: VAHDAT, M. et al., 2015. UCI Machine Learning Repository, DOI [10.24432/C5NP5K](https://doi.org/10.24432/C5NP5K), licença CC BY 4.0. O conjunto não é redistribuído aqui: o script o baixa direto da UCI.
- **Censo da Educação Superior**: microdados públicos do INEP, filtrados para o curso de Sistemas de Informação da instituição. O arquivo contém só volumes agregados, sem nenhum dado pessoal.
- Os históricos acadêmicos do gerador são **sintéticos**.

## Licença

- Código: [MIT](LICENSE).
- Dados derivados, figuras e documentação: [CC-BY-4.0](LICENSES/CC-BY-4.0.txt).

## Como citar

Use o botão "Cite this repository" do GitHub ([`CITATION.cff`](CITATION.cff)) ou o DOI do Zenodo acima.

## Autores

Centro Universitário Católica de Quixadá (UniCatólica):

- Ronildo Oliveira da Silva. ORCID [0009-0008-4629-117X](https://orcid.org/0009-0008-4629-117X).
- Francisca Bianca da Silva. ORCID [0009-0008-4503-6337](https://orcid.org/0009-0008-4503-6337).
- Isaac da Silva Viana. ORCID [0009-0006-5307-6553](https://orcid.org/0009-0006-5307-6553).
