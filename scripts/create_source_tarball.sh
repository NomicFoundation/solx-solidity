#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(dirname "$0")"/..
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"

cd "$REPO_ROOT"
version=$(scripts/get_version.sh)
commit_hash=$(git rev-parse --short=8 HEAD)
commit_date=$(TZ=UTC git show --quiet --date="format-local:%Y.%-m.%-d" --format="%cd")

# File exists and has zero size -> not a prerelease
if [[ -e prerelease.txt && ! -s prerelease.txt ]]; then
    version_string="$version"
elif [[ -e prerelease.txt ]]; then
    version_string="${version}-$(cat prerelease.txt)-${commit_hash}"
else
    version_string="${version}-nightly-${commit_date}-${commit_hash}"
fi

# The only purpose of commit_hash.txt is to make it possible to build the compiler without git.
# It is not meant as an override of the real hash.
[[ ! -e commit_hash.txt ]] || \
    fail "commit_hash.txt is present in the repository root, but will not be used to override the commit hash for the source package."

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
