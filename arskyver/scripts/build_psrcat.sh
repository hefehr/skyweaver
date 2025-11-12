#!/usr/bin/env bash
set -euo pipefail

: "${SOFTWARE_DIR:?}"

cd "${SOFTWARE_DIR}/src"
wget -q https://www.atnf.csiro.au/research/pulsar/psrcat/downloads/psrcat_pkg.v2.7.0.tar.gz
tar -xzf psrcat_pkg.v2.7.0.tar.gz --no-same-owner
rm -f psrcat_pkg.v2.7.0.tar.gz
cd "${SOFTWARE_DIR}/src/psrcat_tar"
/bin/sh makeit
