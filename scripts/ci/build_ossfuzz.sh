#!/usr/bin/env bash
set -ex

ROOTDIR="$(realpath "$(dirname "$0")/../..")"
BUILDDIR="${ROOTDIR}/build"
mkdir -p "${BUILDDIR}" && mkdir -p "$BUILDDIR/deps"

function generate_protobuf_bindings
{
  cd "${ROOTDIR}"/test/tools/ossfuzz
  # Generate protobuf C++ bindings
  for protoName in yul abiV2 sol;
  do
    protoc "${protoName}"Proto.proto --cpp_out .
  done
}

function build_fuzzers
{
  cd "${BUILDDIR}"
  if [[ "${CCACHE_ENABLED:-}" == "1" ]]; then
    export CCACHE_DIR="$HOME/.ccache"
    export CCACHE_BASEDIR="$ROOTDIR"
    export CCACHE_NOHASHDIR=1
    CMAKE_OPTIONS="${CMAKE_OPTIONS:-} -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
    mkdir -p "$CCACHE_DIR"
  fi
  # shellcheck disable=SC2086
  cmake .. -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
    -DCMAKE_TOOLCHAIN_FILE="${ROOTDIR}"/cmake/toolchains/libfuzzer.cmake \
    $CMAKE_OPTIONS
  if [[ "${CCACHE_ENABLED:-}" == "1" ]]; then
    ccache -z
  fi
  make ossfuzz ossfuzz_proto ossfuzz_abiv2 -j 4
  if [[ "${CCACHE_ENABLED:-}" == "1" ]]; then
    ccache -s
  fi
}

generate_protobuf_bindings
build_fuzzers
