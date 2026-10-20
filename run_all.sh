#!/usr/bin/env bash
# Reproduz todos os números e figuras do artigo. Resultados em resultados/ e figuras em figuras/.
# Use PULAR_GRADE=1 para pular 11_revisao.py (busca em grade aninhada, cerca de uma hora em 8 núcleos).
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")" && pwd)"
cd "$RAIZ"

bash scripts/baixar_epm.sh

export EPM_OUT="$RAIZ/resultados"
mkdir -p "$EPM_OUT"

cd analise
# Avaliação no EPM (seções 3.3 a 4.5)
python 01_build.py
python 02_metrics.py
python 03_stats.py
python 04_early.py
python 06_robustez.py
python 09_componentes.py
python 12_parecer_cbs.py
if [ "${PULAR_GRADE:-0}" != "1" ]; then
    python 11_revisao.py
fi

# Demonstração do protótipo com o INEP e o gerador sintético (seções 4.1 e 4.6)
python gerador_artigo/mock_data_generator.py
python 07_demonstracao.py
python 08_concorrencia.py
python 10_powl_controle.py

# Figuras
python 05_figs.py "$RAIZ/figuras"
python 13_fig_estoques.py "$RAIZ/figuras"
