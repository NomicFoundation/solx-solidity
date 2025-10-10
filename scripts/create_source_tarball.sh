#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(dirname "$0")"/..
cd "$REPO_ROOT"
version=$(scripts/get_version.sh)
commit_hash=$(git rev-parse --short=8 HEAD)
commit_date=$(git show --format=%ci HEAD | head -n 1 | cut - -b1-10 | sed -e 's/-0?/./' | sed -e 's/-0?/./')

# File exists and has zero size -> not a prerelease
if [[ -e prerelease.txt && ! -s prerelease.txt ]]; then
    version_string="$version"
else
    version_string="${version}-nightly-${commit_date}-${commit_hash}"
fi

TEMPDIR=$(mktemp -d -t "solc-src-tarball-XXXXXX")
SOLDIR="${TEMPDIR}/solidity_${version_string}/"
mkdir "$SOLDIR"

# Ensure that submodules are initialized.
git submodule update --init --recursive
# Store the current source
git checkout-index --all --prefix="$SOLDIR"
# shellcheck disable=SC2016
SOLDIR="$SOLDIR" git submodule foreach 'git checkout-index --all --prefix="${SOLDIR}/${sm_path}/"'

# Include the commit hash and prerelease suffix in the tarball
echo "$commit_hash" > "${SOLDIR}/commit_hash.txt"
[[ -e prerelease.txt && ! -s prerelease.txt ]] && cp prerelease.txt "${SOLDIR}/"

mkdir -p "$REPO_ROOT/upload"
tar \
    --owner 0 \
    --group 0 \
    --create \
    --gzip \
    --file "${REPO_ROOT}/upload/solidity_${version_string}.tar.gz" \
    --directory "$TEMPDIR" \
    "solidity_${version_string}"
rm -r "$TEMPDIR"
