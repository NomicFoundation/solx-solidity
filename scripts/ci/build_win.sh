#!/usr/bin/env bash
set -euo pipefail

(( $# <= 1 )) || { >&2 echo "Usage: $0 [PRERELEASE_SOURCE]"; exit 1; }
prerelease_source="${1:-ci}"
ROOTDIR="$(realpath "$(dirname "$0")/../../")"
FORCE_RELEASE="${FORCE_RELEASE:-}"
CIRCLE_TAG="${CIRCLE_TAG:-}"

cd "$ROOTDIR"
"${ROOTDIR}/scripts/prerelease_suffix.sh" "$prerelease_source" "$CIRCLE_TAG" > prerelease.txt

if [[ "${CCACHE_ENABLED:-}" == "1" ]]; then
  export CCACHE_BASEDIR="$ROOTDIR"
  export CCACHE_NOHASHDIR=1
  export CCACHE_COMPILERTYPE=msvc
  # Hard-coded MSVC cl.exe path
  CCACHE_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe"
  export CCACHE_COMPILER
  PATH="$ROOTDIR/deps/ccache:$(dirname "$CCACHE_COMPILER"):$PATH"
  export PATH
  # Use a Windows-style path for Visual Studio globals.
  WIN_PWD="$(pwd -W)"
  # Visual Studio generator doesn't use compiler launchers; use ccache masquerade.
  # See https://github.com/ccache/ccache/wiki/MS-Visual-Studio
  CMAKE_OPTIONS="${CMAKE_OPTIONS:-} -DCMAKE_VS_GLOBALS=CLToolPath=${WIN_PWD}/deps/ccache;UseMultiToolTask=true"
fi

mkdir -p build/
cd build/

# NOTE: Using an array to force Bash to do wildcard expansion
boost_dir=("${ROOTDIR}/deps/boost/lib/cmake/Boost-"*)

 # shellcheck disable=SC2086
 "${ROOTDIR}/deps/cmake/bin/cmake" \
    -G "Visual Studio 16 2019" \
    -DBoost_DIR="${boost_dir[*]}" \
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
    -DCMAKE_INSTALL_PREFIX="${ROOTDIR}/uploads/" \
    ${CMAKE_OPTIONS:-} \
    ..
if [[ "${CCACHE_ENABLED:-}" == "1" ]]; then
  ccache -z
fi
MSBuild.exe solidity.sln \
    -property:Configuration=Release \
    -maxCpuCount:10 \
    -verbosity:minimal
"${ROOTDIR}/deps/cmake/bin/cmake" \
    --build . \
    -j 10 \
    --target install \
    --config Release
if [[ "${CCACHE_ENABLED:-}" == "1" ]]; then
  ccache -s
fi
