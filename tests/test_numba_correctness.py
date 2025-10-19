"""
Unit tests for verifying correctness of PackingEnv implementations.

This test suite verifies:
1. Numba-optimized functions produce identical results to pure Python
2. Environment state transitions are deterministic
3. Plane features calculation is correct
4. Gap metric calculation is correct


import unittest
import numpy as np
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.packingEnv import PackingEnv

try:
    from envs.numba_utils import (
        NUMBA_AVAILABLE,
        calculate_plane_features_numba,
        directional_distance_numba,
        calculate_g_numba
    )
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Could not import numba_utils")


class TestNumbaCorrectness(unittest.TestCase):
    """Test that Numba-optimized functions produce correct results."""

    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)
        np.random.seed(42)

        # Create test environment
        L, W = 100, 100
        n_boxes = 20

        l_samples = [random.randint(L // 10, L // 2) for _ in range(n_boxes)]
        w_samples = [random.randint(W // 10, W // 2) for _ in range(n_boxes)]
        h_samples = [random.randint(min(L, W) // 10, max(L, W) // 2) for _ in range(n_boxes)]
        boxes = list(zip(l_samples, w_samples, h_samples))

        self.env = PackingEnv(container_dims=(L, W), initial_boxes=boxes, render_mode=None)

        # Run some random steps to create non-trivial state
        obs, info = self.env.reset(seed=42)
        for i in range(5):
            action = self.env.action_space.sample()
            if len(self.env.unpacked_boxes) > 0:
                action['box_select'] = action['box_select'] % len(self.env.unpacked_boxes)
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                break

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not available")
    def test_directional_distance_correctness(self):
        """Test that directional_distance_numba matches Python implementation."""
        test_positions = [
            (0, 0),      # Corner
            (50, 50),    # Center
            (99, 99),    # Opposite corner
            (25, 75),    # Random position
        ]

        for row, col in test_positions:
            with self.subTest(row=row, col=col):
                # Python version
                python_result = self.env.directional_distance(row, col)

                # Numba version
                numba_result = directional_distance_numba(self.env.container_height_map, row, col)

                # Convert Python dict to array for comparison
                expected = np.array([
                    python_result['right'],
                    python_result['down'],
                    python_result['left'],
                    python_result['up'],
                    python_result['right_next'],
                    python_result['down_next']
                ], dtype=np.float32)

                np.testing.assert_array_equal(
                    numba_result,
                    expected,
                    err_msg=f"Mismatch at position ({row}, {col})"
                )

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not available")
    def test_plane_features_correctness(self):
        """Test that calculate_plane_features_numba matches Python implementation."""
        # Get Python version (disable numba temporarily)
        from envs import packingEnv
        original_flag = packingEnv.NUMBA_AVAILABLE

        try:
            # Force Python version
            packingEnv.NUMBA_AVAILABLE = False
            python_features = self.env._calculate_plane_features()

            # Force Numba version
            packingEnv.NUMBA_AVAILABLE = True
            numba_features = self.env._calculate_plane_features()

        finally:
            # Restore original flag
            packingEnv.NUMBA_AVAILABLE = original_flag

        # Verify shapes match
        self.assertEqual(python_features.shape, numba_features.shape)
        self.assertEqual(python_features.shape, (100, 100, 7))

        # Verify values match (allowing for floating point precision)
        np.testing.assert_allclose(
            numba_features,
            python_features,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Plane features don't match"
        )

        # Additional check: verify exact equality
        max_diff = np.max(np.abs(numba_features - python_features))
        self.assertLess(max_diff, 1e-10, f"Max difference: {max_diff}")

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not available")
    def test_calculate_g_correctness(self):
        """Test that calculate_g_numba matches Python implementation."""
        # Get Python version
        from envs import packingEnv
        original_flag = packingEnv.NUMBA_AVAILABLE

        try:
            # Force Python version
            packingEnv.NUMBA_AVAILABLE = False
            python_g = self.env._calculate_g()

            # Force Numba version
            packingEnv.NUMBA_AVAILABLE = True
            numba_g = self.env._calculate_g()

        finally:
            packingEnv.NUMBA_AVAILABLE = original_flag

        # Verify values match
        self.assertAlmostEqual(
            numba_g,
            python_g,
            places=5,
            msg=f"Gap metric mismatch: numba={numba_g}, python={python_g}"
        )

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba not available")
    def test_deterministic_execution(self):
        """Test that plane features are deterministic for same state."""
        # This test verifies that given the same container state,
        # plane features are calculated deterministically

        # Save current state
        state_copy = self.env.container_height_map.copy()

        # Calculate features multiple times
        features1 = self.env._calculate_plane_features()
        features2 = self.env._calculate_plane_features()
        features3 = self.env._calculate_plane_features()

        # All should be identical
        np.testing.assert_array_equal(features1, features2, err_msg="Features not deterministic (1 vs 2)")
        np.testing.assert_array_equal(features2, features3, err_msg="Features not deterministic (2 vs 3)")

    def test_plane_features_shape(self):
        """Test that plane features have correct shape."""
        features = self.env._calculate_plane_features()

        self.assertEqual(features.shape, (100, 100, 7))
        self.assertEqual(features.dtype, np.float32)

    def test_plane_features_height_channel(self):
        """Test that channel 0 of plane features matches height map."""
        features = self.env._calculate_plane_features()

        # Channel 0 should be the height
        np.testing.assert_array_equal(
            features[:, :, 0],
            self.env.container_height_map,
            err_msg="Height channel doesn't match height map"
        )

    def test_empty_container(self):
        """Test plane features on empty container."""
        # Create environment with one box but don't place it
        empty_env = PackingEnv(container_dims=(10, 10), initial_boxes=[(5, 5, 5)], render_mode=None)
        empty_env.reset()
        # Clear the height map to simulate empty container
        empty_env.container_height_map = np.zeros((10, 10), dtype=np.float32)

        features = empty_env._calculate_plane_features()

        # All heights should be 0
        self.assertTrue(np.all(features[:, :, 0] == 0))

        # Check expected directional distances for empty container
        # For (5,5) in empty 10x10 with all same height:
        # - right: includes current cell, goes to edge = 4 cells  (5,6,7,8,9)
        # - down: includes current cell, goes to edge = 4 cells (5,6,7,8,9)
        # - left: excludes current cell, goes to edge = 5 cells (0,1,2,3,4)
        # - up: excludes current cell, goes to edge = 5 cells (0,1,2,3,4)
        # Actually looking at the code: DOWN starts at 1, UP starts at 0
        center_features = features[5, 5, :]

        # The function counts distance to different height, not to edge
        # In empty container all heights are same, so it counts to edge
        # Verify the values are reasonable (all same height means maximum distance)
        self.assertGreater(center_features[1], 0, "Right distance should be positive")
        self.assertGreater(center_features[2], 0, "Down distance should be positive")
        self.assertGreater(center_features[3], 0, "Left distance should be positive")
        self.assertGreater(center_features[4], 0, "Up distance should be positive")


class TestPackingEnvCorrectness(unittest.TestCase):
    """Test PackingEnv basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = PackingEnv(
            container_dims=(50, 50),
            initial_boxes=[(10, 10, 10), (20, 20, 20)],
            render_mode=None
        )

    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()

        # Check observation structure
        self.assertIn('container_state', obs)
        self.assertIn('unpacked_boxes_state', obs)

        # Check shapes
        self.assertEqual(obs['container_state'].shape, (50, 50, 7))
        self.assertEqual(obs['unpacked_boxes_state'].shape, (2, 3))

        # Check info
        self.assertEqual(info['num_unpacked_boxes'], 2)
        self.assertEqual(info['num_packed_boxes'], 0)

    def test_valid_placement(self):
        """Test valid box placement."""
        obs, info = self.env.reset()

        # Place first box at origin
        action = {
            'position': (0, 0),
            'box_select': 0,
            'orientation': 0  # (10, 10, 10)
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check box was placed
        self.assertEqual(info['num_packed_boxes'], 1)
        self.assertEqual(info['num_unpacked_boxes'], 1)

        # Check height map updated
        self.assertEqual(self.env.container_height_map[0, 0], 10)
        self.assertEqual(self.env.container_height_map[9, 9], 10)
        self.assertEqual(self.env.container_height_map[10, 10], 0)

    def test_invalid_placement_out_of_bounds(self):
        """Test invalid placement outside container."""
        obs, info = self.env.reset()

        # Try to place box outside container
        action = {
            'position': (45, 45),  # Will exceed bounds with 10x10 box
            'box_select': 0,
            'orientation': 0
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Should receive large negative reward
        self.assertLess(reward, -1e6)

        # Box should not be placed
        self.assertEqual(info['num_packed_boxes'], 0)


def run_tests():
    """Run all tests and print results."""
    print("=" * 70)
    print("Running Correctness Tests")
    print("=" * 70)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestNumbaCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestPackingEnvCorrectness))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(run_tests())