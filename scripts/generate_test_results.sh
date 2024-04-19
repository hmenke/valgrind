#!/bin/sh

# This script generates expected results for the following AVX-512 regression 
# tests:
# 
#
# Expected results are not provided with the patch because the file size
# exceeds limits allowed by bugtracking system

# Please run the script from Valgrind root directory before testing AVX-512 
# to avoid unexpected regression test failures

make check

FILE=none/tests/amd64/avx512-l1
if test -f "$FILE"; then
   ./$FILE > none/tests/amd64/avx512-l1.stdout.exp
   touch none/tests/amd64/avx512-l1.stderr.exp
   echo "Generated AVX-1 legacy test results"
else
   echo "Skipped AVX-1 legacy test results"
fi

FILE=none/tests/amd64/avx512-l2
if test -f "$FILE"; then
   ./$FILE > none/tests/amd64/avx512-l2.stdout.exp
   touch none/tests/amd64/avx512-l2.stderr.exp
   echo "Generated AVX-2 legacy test results"
else
   echo "Skipped AVX-2 legacy test results"
fi

FILE=none/tests/amd64/avx512-skx
if test -f "$FILE"; then
   ./$FILE > none/tests/amd64/avx512-skx.stdout.exp
   touch none/tests/amd64/avx512-skx.stderr.exp
   echo "Generated Skylake test results"
else
   echo "Skipped Skylake test results"
fi

FILE=none/tests/amd64/avx512-knl
if test -f "$FILE"; then
   ./$FILE > none/tests/amd64/avx512-knl.stdout.exp
   touch none/tests/amd64/avx512-knl.stderr.exp
   echo "Generated Knights Landing test results"
else
   echo "Skipped Knights Landing test results"
fi

FILE=./memcheck/tests/amd64/xsave-avx512
if test -f "$FILE"; then
   ./$FILE x 2>./memcheck/tests/amd64/xsave-avx512.stderr.exp
   touch ./memcheck/tests/amd64/xsave-avx512.stdout.exp
   echo "Generated XSAVE test results"
else
   echo "Skipped XSAVE test results"
fi
