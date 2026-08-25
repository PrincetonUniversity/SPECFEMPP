#!/bin/sh
# Record one matrix cell's ctest outcome for status_description.sh to aggregate.
#
#   record_test_result.sh <src-xml> <dst-xml> <label> <status-file>
#
# <src-xml> is the JUnit file ctest wrote in the (GPFS) test directory; the compute
# node running ctest cannot see the Jenkins workspace, so it has to be copied back
# here rather than written straight to <dst-xml>.
#
# This lives in its own script rather than inline in the Jenkinsfile because the sed
# expressions below would otherwise have to survive Groovy triple-quote interpolation
# on top of shell quoting -- every backslash doubled, and no way to test it outside
# a Jenkins run.

src=$1
dst=$2
label=$3
out=$4

if [ ! -f "$src" ]; then
    # ctest died before writing results, or never ran at all.
    printf 'TESTS_ERROR %s\n' "$label" > "$out"
    exit 0
fi

mkdir -p "$(dirname "$dst")"
cp "$src" "$dst"

# Matrix cells share one workspace and one Jenkins test report, so identical gtest
# names from different cells would merge into a single confusing entry. Prefixing
# classname keeps them as separate packages. '|' delimits because $label contains '/'.
sed -i "s| classname=\"| classname=\"$label.|g" "$dst"

# ctest writes these on the <testsuite> element, one attribute per line.
failures=$(sed -n 's/.*failures="\([0-9]*\)".*/\1/p' "$dst" | head -1)
total=$(sed -n 's/.*tests="\([0-9]*\)".*/\1/p' "$dst" | head -1)

if [ -z "$failures" ] || [ -z "$total" ]; then
    printf 'TESTS_ERROR %s\n' "$label" > "$out"
else
    printf 'TESTS %s %s %s\n' "$label" "$failures" "$total" > "$out"
fi
