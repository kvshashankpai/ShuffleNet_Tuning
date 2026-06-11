#!/usr/bin/env bash
# scripts/check_hardware.sh
# --------------------------
# Pre-flight check: confirm your CPU vendor and core count before running
# the grid search. Determines which energy profiling backend to use.
#
# Run this once before starting experiments:
#   bash scripts/check_hardware.sh

echo ""
echo "============================================================"
echo "  ShuffleNetV2 CPU Hardware Check"
echo "============================================================"
echo ""

# Detect OS
OS_TYPE=$(uname -s)

if [ "$OS_TYPE" = "Darwin" ]; then
    # macOS CPU Check
    BRAND=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
    echo "  CPU Brand:         $BRAND"
    
    # Vendor determination
    if echo "$BRAND" | grep -iq "intel"; then
        VENDOR="GenuineIntel"
    elif echo "$BRAND" | grep -iq "amd"; then
        VENDOR="AuthenticAMD"
    elif echo "$BRAND" | grep -iq "apple"; then
        VENDOR="Apple Silicon"
    else
        VENDOR="Unknown"
    fi
    echo "  CPU Vendor:        $VENDOR"
    
    if [ "$VENDOR" = "GenuineIntel" ]; then
        echo "  Profiling backend: CodeCarbon or pyRAPL (Intel RAPL supported)"
    elif [ "$VENDOR" = "AuthenticAMD" ]; then
        echo "  Profiling backend: AMD µProf recommended"
        echo "  NOTE: Swap EmissionsTracker in engine/profiler.py for AMD µProf"
    elif [ "$VENDOR" = "Apple Silicon" ]; then
        echo "  Profiling backend: CodeCarbon (estimating for Apple Silicon)"
    else
        echo "  Profiling backend: CodeCarbon (software estimation fallback)"
    fi
    
    echo ""
    
    PHYSICAL_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null)
    LOGICAL_CORES=$(sysctl -n hw.logicalcpu 2>/dev/null)
    echo "  Physical cores:    $PHYSICAL_CORES"
    echo "  Logical cores:     $LOGICAL_CORES"
else
    # Linux/other CPU Check
    if command -v lscpu >/dev/null 2>&1; then
        VENDOR=$(lscpu | grep "Vendor ID" | awk '{print $3}')
        echo "  CPU Vendor:        $VENDOR"

        if [ "$VENDOR" = "GenuineIntel" ]; then
            echo "  Profiling backend: CodeCarbon or pyRAPL (Intel RAPL supported)"
        elif [ "$VENDOR" = "AuthenticAMD" ]; then
            echo "  Profiling backend: AMD µProf recommended"
            echo "  NOTE: Swap EmissionsTracker in engine/profiler.py for AMD µProf"
        else
            echo "  Profiling backend: CodeCarbon (software estimation fallback)"
        fi

        echo ""

        PHYSICAL_CORES=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')
        LOGICAL_CORES=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
        echo "  Physical cores:    $PHYSICAL_CORES"
        echo "  Logical cores:     $LOGICAL_CORES"
    else
        echo "  lscpu not found. Attempting fallback vendor check."
        VENDOR="Unknown"
        if [ -f /proc/cpuinfo ]; then
            VENDOR=$(grep -m1 "vendor_id" /proc/cpuinfo | awk '{print $3}')
        fi
        echo "  CPU Vendor:        $VENDOR"
        
        PHYSICAL_CORES=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "1")
        LOGICAL_CORES=$PHYSICAL_CORES
        echo "  Cores (estimated): $PHYSICAL_CORES"
    fi
fi

echo ""

# ── Thread Grid Advice ────────────────────────────────────────────────────────
echo "  Recommended thread values for base_config.py:"
if [ -z "$PHYSICAL_CORES" ] || [ "$PHYSICAL_CORES" -le 2 ]; then
    echo "    intra_op_threads: [1, 2]"
elif [ "$PHYSICAL_CORES" -le 4 ]; then
    echo "    intra_op_threads: [1, 2, 4]"
else
    echo "    intra_op_threads: [1, 2, 4, $PHYSICAL_CORES]"
fi

echo ""

# ── Python / PyTorch Check ───────────────────────────────────────────────────
echo "  Python version:    $(python3 --version 2>&1)"
echo "  PyTorch version:   $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "  CodeCarbon:        $(python3 -c 'import codecarbon; print(codecarbon.__version__)' 2>/dev/null || echo 'NOT FOUND — pip install codecarbon')"
echo "  MedMNIST:          $(python3 -c 'import medmnist; print(medmnist.__version__)' 2>/dev/null || echo 'NOT FOUND — pip install medmnist')"

echo ""
echo "============================================================"
echo ""
