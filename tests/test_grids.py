"""Tests for src.grids — synthetic pd.Index lat/lon arrays, no I/O."""

import numpy as np
import pandas as pd

from src.grids import nearest_cell_indices


class TestNearestCellIndices:
    def test_exact_cell(self):
        lat_index = pd.Index([0.0, 1.0, 2.0, 3.0])
        lon_index = pd.Index([10.0, 11.0, 12.0])
        lat_pos, lon_pos = nearest_cell_indices(
            lat_index, lon_index, np.array([2.0]), np.array([11.0])
        )
        assert lat_pos.tolist() == [2]
        assert lon_pos.tolist() == [1]

    def test_between_cells_rounds_to_nearest(self):
        lat_index = pd.Index([0.0, 1.0, 2.0, 3.0])
        lon_index = pd.Index([10.0, 11.0, 12.0])
        # 1.6 is closer to 2.0 than 1.0; 10.4 is closer to 10.0 than 11.0.
        lat_pos, lon_pos = nearest_cell_indices(
            lat_index, lon_index, np.array([1.6]), np.array([10.4])
        )
        assert lat_pos.tolist() == [2]
        assert lon_pos.tolist() == [0]

    def test_outside_range_clamps_to_edge(self):
        lat_index = pd.Index([0.0, 1.0, 2.0, 3.0])
        lon_index = pd.Index([10.0, 11.0, 12.0])
        lat_pos, lon_pos = nearest_cell_indices(
            lat_index, lon_index,
            np.array([-10.0, 100.0]), np.array([-180.0, 180.0]),
        )
        assert lat_pos.tolist() == [0, 3]
        assert lon_pos.tolist() == [0, 2]

    def test_descending_index(self):
        # Some grids store latitude north-to-south (descending).
        lat_index = pd.Index([3.0, 2.0, 1.0, 0.0])
        lon_index = pd.Index([10.0, 11.0, 12.0])
        lat_pos, lon_pos = nearest_cell_indices(
            lat_index, lon_index, np.array([0.4]), np.array([11.9])
        )
        assert lat_pos.tolist() == [3]
        assert lon_pos.tolist() == [2]

    def test_multiple_points(self):
        lat_index = pd.Index([0.0, 1.0, 2.0])
        lon_index = pd.Index([0.0, 1.0, 2.0])
        lat_pos, lon_pos = nearest_cell_indices(
            lat_index, lon_index,
            np.array([0.0, 1.1, 2.0]), np.array([2.0, 0.9, 0.0]),
        )
        assert lat_pos.tolist() == [0, 1, 2]
        assert lon_pos.tolist() == [2, 1, 0]
