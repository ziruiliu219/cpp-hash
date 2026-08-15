#!/bin/bash
# Build and run the TaperHashTable C++ benchmark
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "═══════════════════════════════════════════════════════════"
echo " TaperHashTable C++ Benchmark"
echo "═══════════════════════════════════════════════════════════"

# Build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo ""
echo "─── Running tests ───────────────────────────────────────"
./taper_test

echo ""
echo "─── Running benchmark ───────────────────────────────────"
# Default: run a representative subset
FILTER="${1:-.*4str_0int.*ht=65536.*}"
echo "Filter: $FILTER"
echo ""
./taper_bench --benchmark_filter="$FILTER" --benchmark_counters_tabular=true
