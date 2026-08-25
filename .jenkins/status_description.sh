#!/bin/sh
# Summarize one status file per matrix cell into a GitHub commit-status description.
#
#   status_description.sh [status-dir]
#
# Each matrix cell writes one file into <status-dir>, holding a single line:
#
#     STARTED     <label>                   cell began but never reported an outcome
#     CONFIGURE   <label>                   cmake configure failed
#     COMPILE     <label>                   cmake --build failed
#     TESTS       <label> <failed> <total>  ctest ran; counts read from the JUnit XML
#     TESTS_ERROR <label>                   ctest produced no parseable results
#
# Prints "<SUCCESS|FAILURE> <description>" on stdout, one line.
#
# STARTED exists because `post { failure { ... } }` does not fire when a stage is
# ABORTED (job cancelled, Slurm timeout). Without it a killed cell would vanish from
# the tally and a red build could be described as a green one.
#
# The description is clipped to GitHub's 140-character commit-status limit; anything
# longer is silently truncated by the API, so a too-long string loses the tail that
# usually carries the test counts.
#
# Deliberately free of `set -e`: the logic is mostly `[ ... ]` tests, whose non-zero
# exit at the end of a loop body would abort the script under errexit.

dir=${1:-status}

MAX_DESC=140
MAX_LABELS=2

conf_n=0; conf_l=''
comp_n=0; comp_l=''
inc_n=0;  inc_l=''
terr_n=0; terr_l=''
t_fail=0; t_total=0
cells=0

for f in "$dir"/*; do
    [ -f "$f" ] || continue
    cells=$((cells + 1))
    kind=''; label=''; a=''; b=''
    read -r kind label a b < "$f"
    case "$kind" in
        CONFIGURE)   conf_n=$((conf_n + 1)); conf_l="$conf_l $label" ;;
        COMPILE)     comp_n=$((comp_n + 1)); comp_l="$comp_l $label" ;;
        STARTED)     inc_n=$((inc_n + 1));   inc_l="$inc_l $label" ;;
        TESTS_ERROR) terr_n=$((terr_n + 1)); terr_l="$terr_l $label" ;;
        TESTS)
            # A TESTS line missing its counts means the XML was there but unparseable.
            # Treating it as success would report green on no evidence.
            case "$a$b" in
                ''|*[!0-9]*) terr_n=$((terr_n + 1)); terr_l="$terr_l $label" ;;
                *)           t_fail=$((t_fail + a)); t_total=$((t_total + b)) ;;
            esac
            ;;
    esac
done

# Render a label list as "A, B" or "A, B +3 more", so one broken axis value cannot
# crowd the test counts out of the 140-character budget.
fmt_labels() {
    n=0; out=''; extra=0
    for l in $1; do
        n=$((n + 1))
        if [ "$n" -le "$MAX_LABELS" ]; then
            if [ -z "$out" ]; then out="$l"; else out="$out, $l"; fi
        else
            extra=$((extra + 1))
        fi
    done
    if [ "$extra" -gt 0 ]; then out="$out +$extra more"; fi
    printf '%s' "$out"
}

desc=''
add() {
    if [ -z "$desc" ]; then desc="$1"; else desc="$desc; $1"; fi
}

fails=$((conf_n + comp_n + inc_n + terr_n + t_fail))

if [ "$cells" -eq 0 ]; then
    state=FAILURE
    desc='no results reported'
elif [ "$fails" -eq 0 ]; then
    state=SUCCESS
    if [ "$cells" -eq 1 ]; then noun='config'; else noun='configs'; fi
    if [ "$t_total" -gt 0 ]; then
        desc="$cells $noun OK; $t_total tests passed"
    else
        desc="$cells $noun OK"
    fi
else
    state=FAILURE
    # Priority order: an earlier phase failing explains why later ones have no data.
    if [ "$conf_n" -gt 0 ]; then add "configure failed: $(fmt_labels "$conf_l")"; fi
    if [ "$comp_n" -gt 0 ]; then add "compile failed: $(fmt_labels "$comp_l")"; fi
    if [ "$inc_n" -gt 0 ]; then
        if [ "$inc_n" -eq 1 ]; then noun='cell'; else noun='cells'; fi
        add "$inc_n $noun did not finish: $(fmt_labels "$inc_l")"
    fi
    if [ "$terr_n" -gt 0 ]; then add "no results: $(fmt_labels "$terr_l")"; fi
    if [ "$t_total" -gt 0 ]; then
        if [ "$t_fail" -gt 0 ]; then
            add "$t_fail/$t_total tests failed"
        else
            add "$t_total tests passed"
        fi
    fi
fi

if [ "${#desc}" -gt "$MAX_DESC" ]; then
    desc="$(printf '%s' "$desc" | cut -c1-$((MAX_DESC - 3)))..."
fi

printf '%s %s\n' "$state" "$desc"
