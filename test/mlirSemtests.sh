#!/usr/bin/env bash
#
# Run each semantic test via MLIR in its own soltest process, in parallel.
#
# Requires: GNU parallel
#
# Usage: ./mlirSemtests.sh --soltest <path> --testpath <path> --vm <path> [--baseline <file>] [filter]
# Examples:
#   ./mlirSemtests.sh --soltest build/dbg/test/soltest --testpath test --vm /path/to/libevmone.so
#   ./mlirSemtests.sh --soltest build/dbg/test/soltest --testpath test --vm /path/to/libevmone.so enums
#   ./mlirSemtests.sh --soltest build/dbg/test/soltest --testpath test --vm /path/to/libevmone.so --baseline test/mlirSemtestFailures.txt

set -eu
export LC_ALL=C

SOLTEST=""
TESTPATH=""
VM=""
BASELINE=""
FILTER=""
OUTDIR=$(mktemp -d -t mlir-semtest-XXXXXX)
JOBLOG=$(mktemp -t mlir-semtest-XXXXXX)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --soltest)   SOLTEST="$2"; shift 2 ;;
        --testpath)  TESTPATH="$2"; shift 2 ;;
        --vm)        VM="$2"; shift 2 ;;
        --baseline)  BASELINE="$2"; shift 2 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           FILTER="$1"; shift ;;
    esac
done

if [[ -z "$SOLTEST" || -z "$TESTPATH" || -z "$VM" ]]; then
    echo "Usage: $0 --soltest <path> --testpath <path> --vm <path> [filter]"
    exit 1
fi

SEMTEST_DIR="$TESTPATH/libsolidity/semanticTests"

collect_tests() {
    local target="$SEMTEST_DIR/$FILTER"
    if [[ -n "$FILTER" && -f "$target" ]]; then
        echo "semanticTests/${FILTER%.sol}"
    elif [[ -z "$FILTER" || -d "$target" ]]; then
        find "${target:-$SEMTEST_DIR}" -name '*.sol' | sort | \
            sed "s|^$TESTPATH/libsolidity/||; s|\.sol$||"
    else
        echo "Not a file or directory: $target" >&2
        exit 1
    fi
}

tests=()
while IFS= read -r line; do
    tests+=("$line")
done < <(collect_tests)
total=${#tests[@]}

if [[ $total -eq 0 ]]; then
    echo "No tests found for filter: $FILTER"
    exit 1
fi

trap 'rm -rf "$OUTDIR" "$JOBLOG"' EXIT

echo "Running $total tests in parallel..."

run_one_test() {
    "$SOLTEST" --catch_system_errors=no --log_level=error --report_level=no \
        -t "$1" \
        -- --testpath "$TESTPATH" --vm "$VM" --force-mlir
}
export -f run_one_test
export SOLTEST TESTPATH VM

printf '%s\n' "${tests[@]}" | \
    parallel --joblog "$JOBLOG" --results "$OUTDIR" \
    run_one_test {} || true

pass=$(tail -n+2 "$JOBLOG" | awk -F'\t' '$7==0 && $8==0' | wc -l)
abort=$(tail -n+2 "$JOBLOG" | awk -F'\t' '$8!=0' | wc -l)
fail=$(tail -n+2 "$JOBLOG" | awk -F'\t' '$7!=0 && $8==0' | wc -l)

echo ""
echo "=========================================="
echo "Results: $total tests"
echo "=========================================="
echo "  Pass:    $pass"
echo "  Fail:    $fail"
echo "  Abort:   $abort"
echo ""

# Write errors.txt — verbose per-test error output
: > errors.txt
failures=$((fail + abort))
if [[ $failures -gt 0 ]]; then
    tail -n+2 "$JOBLOG" | awk -F'\t' '$7!=0 || $8!=0 {print $NF}' | while read -r cmd; do
        test_name="${cmd##*run_one_test }"
        encoded=$(echo "$test_name" | sed 's|/|+z|g')
        stderr_file="$OUTDIR/1/$encoded/stderr"
        stdout_file="$OUTDIR/1/$encoded/stdout"

        echo "=== $test_name ==="
        if [[ -s "$stderr_file" ]]; then
            cat "$stderr_file"
        elif [[ -s "$stdout_file" ]]; then
            cat "$stdout_file"
        fi
        echo ""
    done > errors.txt
fi

# Write failures.txt — sorted list of failing test names
tail -n+2 "$JOBLOG" | awk -F'\t' '$7!=0 || $8!=0 {print $NF}' | \
    sed 's/run_one_test //' | sort > failures.txt

# Compare against baseline if provided
if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
    fixed=$(comm -23 "$BASELINE" failures.txt)
    regressions=$(comm -13 "$BASELINE" failures.txt)

    if [[ -n "$fixed" ]]; then
        echo "Fixed:"
        echo "$fixed"
    fi
    if [[ -n "$regressions" ]]; then
        echo "Regressions:"
        echo "$regressions"
        exit 1
    fi
fi
