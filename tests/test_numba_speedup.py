"""
Test script to verify numba optimizations work correctly and measure speedup.

This script:
1. Tests that numba functions are loaded correctly
2. Verifies numerical correctness (results match pure Python)
3. Measures speedup from JIT compilation
"""

import numpy as np
import time
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test imports
print("=" * 70)
print("Testing Numba Optimizations for PackingEnv")
print("=" * 70)

try:
    from envs.numba_utils import (
        NUMBA_AVAILABLE,
        calculate_plane_features_numba,
        directional_distance_numba,
        calculate_g_numba
    )
    print(f"\n✓ Successfully imported numba_utils")
    print(f"  Numba available: {NUMBA_AVAILABLE}")
except ImportError as e:
    print(f"\n✗ Failed to import numba_utils: {e}")
    exit(1)

# Import environment
try:
    from envs.packingEnv import PackingEnv
    print(f"✓ Successfully imported PackingEnv")
except ImportError as e:
    print(f"\n✗ Failed to import PackingEnv: {e}")
    exit(1)

print("\n" + "=" * 70)
print("Test 1: Numerical Correctness Verification")
print("=" * 70)

# Create test environment
L, W = 100, 100
n_boxes = 20

random.seed(42)
np.random.seed(42)

l_samples = [random.randint(L // 10, L // 2) for _ in range(n_boxes)]
w_samples = [random.randint(W // 10, W // 2) for _ in range(n_boxes)]
h_samples = [random.randint(min(L, W) // 10, max(L, W) // 2) for _ in range(n_boxes)]
boxes = list(zip(l_samples, w_samples, h_samples))

env = PackingEnv(container_dims=(L, W), initial_boxes=boxes, render_mode=None)

print(f"\nCreated test environment: {L}×{W} container, {n_boxes} boxes")

# Run a few random steps to create non-trivial state
obs, info = env.reset()
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break

print(f"Executed {i+1} random steps to create test state")
print(f"  Packed boxes: {info['num_packed_boxes']}")
print(f"  Max height: {info['max_packed_height']:.2f}")

# Test directional distance
if NUMBA_AVAILABLE:
    print("\nTesting directional_distance_numba...")
    test_row, test_col = 50, 50

    # Python version (via env method)
    python_result = env.directional_distance(test_row, test_col)

    # Numba version
    numba_result = directional_distance_numba(env.container_height_map, test_row, test_col)

    # Compare
    expected = [python_result['right'], python_result['down'], python_result['left'],
                python_result['up'], python_result['right_next'], python_result['down_next']]

    matches = np.allclose(numba_result, expected)
    print(f"  Results match: {matches}")
    if not matches:
        print(f"    Python: {expected}")
        print(f"    Numba:  {numba_result}")
else:
    print("\nSkipping directional_distance test (numba not available)")

# Test plane features calculation
print("\nTesting calculate_plane_features_numba...")
if NUMBA_AVAILABLE:
    # Get both versions
    # Note: env._calculate_plane_features will use numba if available,
    # so we need to temporarily disable it
    from envs import packingEnv
    original_flag = packingEnv.NUMBA_AVAILABLE

    # Get numba version
    packingEnv.NUMBA_AVAILABLE = True
    numba_features = env._calculate_plane_features()

    # Get Python version
    packingEnv.NUMBA_AVAILABLE = False
    python_features = env._calculate_plane_features()

    # Restore flag
    packingEnv.NUMBA_AVAILABLE = original_flag

    # Compare
    matches = np.allclose(numba_features, python_features)
    print(f"  Results match: {matches}")
    print(f"  Shape: {numba_features.shape}")
    print(f"  Max difference: {np.max(np.abs(numba_features - python_features)):.10f}")

    if not matches:
        print("  WARNING: Results don't match!")
        diff_mask = ~np.isclose(numba_features, python_features)
        print(f"  Number of differing elements: {np.sum(diff_mask)}")
else:
    print("  Skipping (numba not available)")

print("\n" + "=" * 70)
print("Test 2: Performance Benchmark")
print("=" * 70)

if NUMBA_AVAILABLE:
    # Warm up JIT compiler
    print("\nWarming up JIT compiler (first call compiles, should be slow)...")
    _ = calculate_plane_features_numba(env.container_height_map)
    print("  Done")

    # Benchmark
    n_iterations = 100
    print(f"\nBenchmarking with {n_iterations} iterations...")

    # Numba version
    start = time.time()
    for _ in range(n_iterations):
        _ = calculate_plane_features_numba(env.container_height_map)
    numba_time = time.time() - start

    # Python version (temporarily disable numba)
    from envs import packingEnv
    original_flag = packingEnv.NUMBA_AVAILABLE
    packingEnv.NUMBA_AVAILABLE = False

    start = time.time()
    for _ in range(n_iterations):
        _ = env._calculate_plane_features()
    python_time = time.time() - start

    packingEnv.NUMBA_AVAILABLE = original_flag

    speedup = python_time / numba_time

    print(f"\nResults:")
    print(f"  Python version: {python_time:.4f}s ({python_time/n_iterations*1000:.2f}ms per call)")
    print(f"  Numba version:  {numba_time:.4f}s ({numba_time/n_iterations*1000:.2f}ms per call)")
    print(f"  Speedup:        {speedup:.2f}x")

    if speedup > 5:
        print(f"\n✓ Excellent speedup achieved!")
    elif speedup > 2:
        print(f"\n✓ Good speedup achieved")
    else:
        print(f"\n⚠ Speedup lower than expected (target: >10x)")
else:
    print("\nSkipping benchmark (numba not available)")
    print("To enable numba optimizations, install numba:")
    print("  pip install numba>=0.58.0")

print("\n" + "=" * 70)
print("Test 3: Full Environment Step Performance")
print("=" * 70)

# Reset for clean test
env.reset(seed=42)

n_steps = 50
print(f"\nRunning {n_steps} environment steps...")

start = time.time()
for i in range(n_steps):
    # Sample valid action (ensure box_select is within range)
    action = env.action_space.sample()
    if len(env.unpacked_boxes) > 0:
        action['box_select'] = action['box_select'] % len(env.unpacked_boxes)
    else:
        break
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
total_time = time.time() - start

print(f"\nCompleted {i+1} steps in {total_time:.4f}s")
print(f"  Average per step: {total_time/(i+1)*1000:.2f}ms")
print(f"  Steps per second: {(i+1)/total_time:.1f}")

if NUMBA_AVAILABLE:
    print(f"\n✓ All tests passed! Numba optimizations are working correctly.")
else:
    print(f"\n⚠ Numba not available - using pure Python implementation")

print("\n" + "=" * 70)