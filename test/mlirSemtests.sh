#!/usr/bin/env bash
#
# Run semantic tests via MLIR.
#
# Two modes:
#   --gnu-parallel  Run each test in its own soltest process via GNU parallel (fast on Linux)
#   (default)       Run in a single soltest process using --isolate-tests (fast on macOS)
#
# Usage: ./mlirSemtests.sh --soltest <path> --testpath <path> --vm <path> [--baseline <file>] [--gnu-parallel] [filter]
# Examples:
#   ./mlirSemtests.sh --soltest <build>/test/soltest --testpath test --vm /path/to/libevmone.so
#   ./mlirSemtests.sh --soltest <build>/test/soltest --testpath test --vm /path/to/libevmone.so --gnu-parallel
#   ./mlirSemtests.sh --soltest <build>/test/soltest --testpath test --vm /path/to/libevmone.so --baseline test/mlirSemtestFailures.txt

set -eu
export LC_ALL=C

SOLTEST=""
TESTPATH=""
VM=""
BASELINE=""
FILTER=""
GNU_PARALLEL=false
OUTDIR=$(mktemp -d -t mlir-semtest-XXXXXX)
RAWLOG="$OUTDIR/soltest.raw.log"
CLEANLOG="$OUTDIR/soltest.clean.log"
FAILCOUNT="$OUTDIR/fail.count"
ABORTCOUNT="$OUTDIR/abort.count"
UNSORTED_FAILURES="$OUTDIR/failures.unsorted.txt"
JOBLOG=$(mktemp -t mlir-semtest-XXXXXX)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --soltest)       SOLTEST="$2"; shift 2 ;;
        --testpath)      TESTPATH="$2"; shift 2 ;;
        --vm)            VM="$2"; shift 2 ;;
        --baseline)      BASELINE="$2"; shift 2 ;;
        --gnu-parallel)  GNU_PARALLEL=true; shift ;;
        -*)              echo "Unknown option: $1"; exit 1 ;;
        *)               FILTER="$1"; shift ;;
    esac
done

if [[ -z "$SOLTEST" || -z "$TESTPATH" || -z "$VM" ]]; then
    echo "Usage: $0 --soltest <path> --testpath <path> --vm <path> [filter]"
    exit 1
fi

SEMTEST_DIR="$TESTPATH/libsolidity/semanticTests"

is_valid_semantic_test_name() {
    local name="$1"
    local suffix="${name#semanticTests/}"
    local IFS='/'
    read -r -a parts <<< "$suffix"
    for part in "${parts[@]}"; do
        [[ "$part" == _* ]] && return 1
    done
    return 0
}

collect_tests() {
    local target="$SEMTEST_DIR/$FILTER"
    if [[ -n "$FILTER" && -f "$target" ]]; then
        local test_name="semanticTests/${FILTER%.sol}"
        if is_valid_semantic_test_name "$test_name"; then
            echo "$test_name"
        fi
    elif [[ -z "$FILTER" || -d "$target" ]]; then
        find "${target:-$SEMTEST_DIR}" -name '*.sol' | sort | \
            sed "s|^$TESTPATH/libsolidity/||; s|\.sol$||" | \
            while IFS= read -r test_name; do
                if is_valid_semantic_test_name "$test_name"; then
                    echo "$test_name"
                fi
            done
    else
        echo "Not a file or directory: $target" >&2
        exit 1
    fi
}

suite_filter() {
    local target="$SEMTEST_DIR/$FILTER"
    if [[ -n "$FILTER" && -f "$target" ]]; then
        echo "semanticTests/${FILTER%.sol}"
    elif [[ -n "$FILTER" && -d "$target" ]]; then
        echo "semanticTests/${FILTER%/}/"
    else
        echo "semanticTests/"
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

echo "Running $total tests..."

if $GNU_PARALLEL; then
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

    # Write errors.txt
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

    # Write failures.txt
    tail -n+2 "$JOBLOG" | awk -F'\t' '$7!=0 || $8!=0 {print $NF}' | \
        sed 's/run_one_test //' | sort > failures.txt
else
    run_status=0
    "$SOLTEST" --catch_system_errors=no --log_level=error --report_level=no \
        -t "$(suite_filter)" \
        -- --testpath "$TESTPATH" --vm "$VM" --force-mlir --isolate-tests \
        >"$RAWLOG" 2>&1 || run_status=$?

    perl -pe 's/\e\[[0-9;]*[A-Za-z]//g; s/\r//g' "$RAWLOG" > "$CLEANLOG"

    : > errors.txt
    : > "$UNSORTED_FAILURES"

    awk \
        -v failures_file="$UNSORTED_FAILURES" \
        -v errors_file="errors.txt" \
        -v fail_count_file="$FAILCOUNT" \
        -v abort_count_file="$ABORTCOUNT" '
    function flush_block() {
        if (test_name == "")
            return;

        print test_name >> failures_file;
        print "=== " test_name " ===" >> errors_file;
        if (block != "")
            printf "%s", block >> errors_file;
        printf "\n" >> errors_file;

        if (kind == "abort")
            abort_count++;
        else
            fail_count++;

        test_name = "";
        block = "";
        kind = "";
    }

    {
        if (index($0, "error: in \"") > 0) {
            line = $0;
            start = index(line, "error: in \"") + length("error: in \"");
            remainder = substr(line, start);
            quote = index(remainder, "\"");

            if (quote > 0 && substr(remainder, quote + 1, 2) == ": ") {
                new_test_name = substr(remainder, 1, quote - 1);
                new_block_line = substr(remainder, quote + 3);
                new_kind = (new_block_line ~ /^Isolated test terminated by signal /) ? "abort" : "fail";

                if (test_name == "") {
                    test_name = new_test_name;
                    block = new_block_line "\n";
                    kind = new_kind;
                    next;
                }

                if (new_test_name == test_name) {
                    block = block $0 "\n";
                    if (kind != "abort" && new_kind == "abort")
                        kind = new_kind;
                    next;
                }

                flush_block();
                test_name = new_test_name;
                block = new_block_line "\n";
                kind = new_kind;
                next;
            }
        }
    }

    test_name != "" {
        if ($0 ~ /^\*\*\* [0-9]+ failures are detected in the test module "/) {
            flush_block();
            next;
        }

        block = block $0 "\n";
        next;
    }

    END {
        flush_block();
        print fail_count + 0 > fail_count_file;
        print abort_count + 0 > abort_count_file;
    }
    ' "$CLEANLOG"

    sort -u "$UNSORTED_FAILURES" > failures.txt

    fail=$(cat "$FAILCOUNT")
    abort=$(cat "$ABORTCOUNT")
    pass=$((total - fail - abort))
fi

echo ""
echo "=========================================="
echo "Results: $total tests"
echo "=========================================="
echo "  Pass:    $pass"
echo "  Fail:    $fail"
echo "  Abort:   $abort"
echo ""

# Compare against baseline if provided
if [[ -n "$BASELINE" && -f "$BASELINE" ]]; then
    SELECTED_TESTS="$OUTDIR/selected-tests.txt"
    FILTERED_BASELINE="$OUTDIR/baseline.filtered.txt"
    printf '%s\n' "${tests[@]}" | sort -u > "$SELECTED_TESTS"
    grep -F -x -f "$SELECTED_TESTS" "$BASELINE" | sort -u > "$FILTERED_BASELINE" || true

    fixed=$(comm -23 "$FILTERED_BASELINE" failures.txt)
    regressions=$(comm -13 "$FILTERED_BASELINE" failures.txt)

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

if ! $GNU_PARALLEL && [[ $run_status -ne 0 && $((fail + abort)) -eq 0 ]]; then
    exit "$run_status"
fi
