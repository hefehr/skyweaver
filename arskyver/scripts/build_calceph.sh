#!/usr/bin/env bash
set -euo pipefail

: "${SOFTWARE_DIR:?}"

cd "${SOFTWARE_DIR}/src"
wget -q https://www.imcce.fr/content/medias/recherche/equipes/asd/calceph/calceph-3.5.3.tar.gz
tar -xzf calceph-3.5.3.tar.gz
rm -f calceph-3.5.3.tar.gz
cd "${SOFTWARE_DIR}/src/calceph-3.5.3"
./configure --prefix="${SOFTWARE_DIR}" --with-pic --enable-shared --enable-static --enable-fortran --enable-thread
make -j"$(nproc)"
make check
make install
make clean






