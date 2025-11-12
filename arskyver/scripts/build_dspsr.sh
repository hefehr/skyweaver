#!/usr/bin/env bash
set -euo pipefail

: "${SOFTWARE_DIR:?}"

# Requires: HEALPix (optional), CALCEPH, TEMPO2, PGPLOT env, numpy present in venv

cd "${SOFTWARE_DIR}/src"
git clone --recursive git://git.code.sf.net/p/dspsr/code dspsr
cd dspsr

echo "kat fits sigproc" > backends.list
./bootstrap

./configure \
    --prefix="${SOFTWARE_DIR}"

make -j"$(nproc)"
make
make install
make clean



