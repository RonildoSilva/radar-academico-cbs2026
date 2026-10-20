#!/usr/bin/env bash
# Baixa o conjunto Educational Process Mining (EPM) do repositório da UCI e o extrai em data/epm/.
# Fonte: https://archive.ics.uci.edu/dataset/346 (VAHDAT et al., 2015). O conjunto não é redistribuído aqui.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="$RAIZ/data/epm"
URL="https://archive.ics.uci.edu/static/public/346/educational+process+mining+epm+a+learning+analytics+data+set.zip"
SHA256="8a176960e28771c75365a3f1a50a9ad4a2888606e5003e5e64cb8dbfa8d42160"

if [ -d "$DESTINO/EPM Dataset 2/Data" ]; then
    echo "EPM já está em $DESTINO"
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Baixando EPM da UCI..."
curl -fsSL -o "$TMP/epm.zip" "$URL"

if command -v sha256sum >/dev/null; then
    echo "$SHA256  $TMP/epm.zip" | sha256sum -c -
else
    echo "$SHA256  $TMP/epm.zip" | shasum -a 256 -c -
fi

# O zip da UCI contém um segundo zip ("EPMDataset .zip") com a pasta "EPM Dataset 2".
unzip -q "$TMP/epm.zip" -d "$TMP"
mkdir -p "$DESTINO"
unzip -q "$TMP/EPMDataset .zip" -d "$DESTINO"
echo "EPM extraído em $DESTINO/EPM Dataset 2"
