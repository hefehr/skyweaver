#!/usr/bin/env bash
set -euo pipefail

: "${SOFTWARE_DIR:?}"
: "${TEMPO2:?}"

cd "${SOFTWARE_DIR}/src"
git clone --depth=1 https://bitbucket.org/psrsoft/tempo2.git
cd tempo2
sync && perl -pi -e 's/chmod \+x/#chmod +x/' bootstrap
./bootstrap

cp -r T2runtime "${TEMPO2}/"

X11LIB="/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)"
./configure --prefix="${SOFTWARE_DIR}" --x-libraries="${X11LIB}" --enable-shared --enable-static --with-pic

make -j"$(nproc)"
make install
make plugins-install
make clean
rm -rf .git




