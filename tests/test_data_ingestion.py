"""Tests for data ingestion and validation module."""

import pytest
import pandas as pd
import numpy as np
from pycharting.data.ingestion import DataManager, DataValidationError, validate_input


class TestValidateInput:
    """Tests for the validate_input function."""
    
    def test_valid_pandas_input(self):
        """Test validation with valid Pandas Series input."""
        index = pd.date_range("2024-01-01", periods=5)
        open_data = pd.Series([100, 102, 101, 103, 102])
        high = pd.Series([105, 106, 105, 107, 106])
        low = pd.Series([99, 100, 99, 101, 100])
        close = pd.Series([104, 103, 104, 105, 104])
        
        result = validate_input(index, open_data, high, low, close)
        
        assert isinstance(result["index"], np.ndarray)
        assert isinstance(result["open"], np.ndarray)
        assert len(result["index"]) == 5
        assert np.array_equal(result["open"], [100, 102, 101, 103, 102])
    
    def test_valid_numpy_input(self):
        """Test validation with valid NumPy array input."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        result = validate_input(index, open_data, high, low, close)
        
        assert isinstance(result["index"], np.ndarray)
        assert len(result["index"]) == 5
        assert np.array_equal(result["close"], [104, 103, 104, 105, 104])
    
    def test_with_overlays(self):
        """Test validation with overlay data."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        overlays = {
            "SMA20": np.array([101, 102, 102, 103, 103]),
            "EMA10": np.array([100, 101, 101, 102, 102]),
        }

        result = validate_input(index, open_data, high, low, close, overlays=overlays)

        assert len(result["overlays"]) == 2
        assert "SMA20" in result["overlays"]
        assert "EMA10" in result["overlays"]
        # Overlays are now styled objects with 'data' and 'style' keys
        assert np.array_equal(result["overlays"]["SMA20"]["data"], [101, 102, 102, 103, 103])
        assert result["overlays"]["SMA20"]["style"] == "line"
    
    def test_with_subplots(self):
        """Test validation with subplot data."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        subplots = {
            "Volume": np.array([1000, 1200, 1100, 1300, 1150]),
            "RSI": np.array([55, 58, 52, 60, 57]),
        }
        
        result = validate_input(index, open_data, high, low, close, subplots=subplots)
        
        assert len(result["subplots"]) == 2
        assert "Volume" in result["subplots"]
        assert "RSI" in result["subplots"]
    
    def test_invalid_index_type(self):
        """Test that invalid index type raises error."""
        index = [1, 2, 3, 4, 5]  # List instead of Index or ndarray
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        with pytest.raises(DataValidationError, match="Index must be"):
            validate_input(index, open_data, high, low, close)
    
    def test_length_mismatch(self):
        """Test that mismatched lengths raise error."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101])  # Length 3
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        with pytest.raises(DataValidationError, match="does not match index length"):
            validate_input(index, open_data, high, low, close)
    
    def test_ohlc_constraint_high_violation(self):
        """Test that High < max(Open, Close) raises error."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([99, 106, 105, 107, 106])  # First high < open
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        with pytest.raises(DataValidationError, match="High must be >= max"):
            validate_input(index, open_data, high, low, close)
    
    def test_ohlc_constraint_low_violation(self):
        """Test that Low > min(Open, Close) raises error."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([101, 100, 99, 101, 100])  # First low > open
        close = np.array([104, 103, 104, 105, 104])
        
        with pytest.raises(DataValidationError, match="Low must be <= min"):
            validate_input(index, open_data, high, low, close)
    
    def test_overlay_length_mismatch(self):
        """Test that mismatched overlay length raises error."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        overlays = {"SMA20": np.array([101, 102, 102])}  # Length 3
        
        with pytest.raises(DataValidationError, match="Overlay.*does not match"):
            validate_input(index, open_data, high, low, close, overlays=overlays)


class TestDataManager:
    """Tests for the DataManager class."""
    
    def test_init_with_numpy_arrays(self):
        """Test initialization with NumPy arrays."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        assert len(dm) == 5
        assert dm.length == 5
        assert isinstance(dm.open, np.ndarray)
        assert np.array_equal(dm.open, open_data)
    
    def test_init_with_pandas_series(self):
        """Test initialization with Pandas Series."""
        index = pd.date_range("2024-01-01", periods=5)
        open_data = pd.Series([100, 102, 101, 103, 102])
        high = pd.Series([105, 106, 105, 107, 106])
        low = pd.Series([99, 100, 99, 101, 100])
        close = pd.Series([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        assert len(dm) == 5
        assert isinstance(dm.close, np.ndarray)
        assert dm.close[0] == 104
    
    def test_properties(self):
        """Test all property accessors."""
        index = np.arange(3)
        open_data = np.array([100, 102, 101])
        high = np.array([105, 106, 105])
        low = np.array([99, 100, 99])
        close = np.array([104, 103, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        assert np.array_equal(dm.index, index)
        assert np.array_equal(dm.open, open_data)
        assert np.array_equal(dm.high, high)
        assert np.array_equal(dm.low, low)
        assert np.array_equal(dm.close, close)
        assert isinstance(dm.overlays, dict)
        assert isinstance(dm.subplots, dict)
    
    def test_with_overlays_and_subplots(self):
        """Test initialization with overlays and subplots."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        overlays = {"SMA20": np.array([101, 102, 102, 103, 103])}
        subplots = {"Volume": np.array([1000, 1200, 1100, 1300, 1150])}

        dm = DataManager(index, open_data, high, low, close, overlays, subplots)

        assert len(dm.overlays) == 1
        assert "SMA20" in dm.overlays
        assert len(dm.subplots) == 1
        assert "Volume" in dm.subplots
        # Overlays are now styled objects with 'data' and 'style' keys
        assert np.array_equal(dm.overlays["SMA20"]["data"], [101, 102, 102, 103, 103])
    
    def test_invalid_data_raises_error(self):
        """Test that invalid OHLC data raises DataValidationError."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([99, 106, 105, 107, 106])  # First high < open
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        with pytest.raises(DataValidationError):
            DataManager(index, open_data, high, low, close)
    
    def test_repr(self):
        """Test string representation."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        repr_str = repr(dm)
        
        assert "DataManager" in repr_str
        assert "5 points" in repr_str
    
    def test_repr_with_overlays(self):
        """Test string representation with overlays."""
        index = np.arange(3)
        open_data = np.array([100, 102, 101])
        high = np.array([105, 106, 105])
        low = np.array([99, 100, 99])
        close = np.array([104, 103, 104])
        overlays = {"SMA20": np.array([101, 102, 102])}
        
        dm = DataManager(index, open_data, high, low, close, overlays=overlays)
        repr_str = repr(dm)
        
        assert "1 overlays" in repr_str
    
    def test_no_data_duplication(self):
        """Test that data is not duplicated unnecessarily."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        # Verify arrays are stored (conversion happened but data is referenced)
        assert dm.open.dtype == open_data.dtype
        assert len(dm.open) == len(open_data)
    
    def test_timestamp_conversion_to_milliseconds(self):
        """Test that DatetimeIndex is converted to Unix timestamps in milliseconds."""
        # Create a DatetimeIndex with known timestamps
        index = pd.date_range("2024-01-01", periods=5, freq="h")
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        # Get chunk should return timestamps in milliseconds
        chunk = dm.get_chunk(0, 5)
        
        # Verify that index is a list of integers (Unix timestamps in milliseconds)
        assert isinstance(chunk["index"], list)
        assert all(isinstance(x, int) for x in chunk["index"])
        
        # Verify timestamps are in the correct range (milliseconds since epoch)
        # For 2024-01-01, timestamps should be around 1704067200000 (ms)
        expected_first_ts = int(pd.Timestamp("2024-01-01").timestamp() * 1000)
        assert chunk["index"][0] == expected_first_ts
        
        # Verify timestamps are 1 hour apart (3600000 ms)
        assert chunk["index"][1] - chunk["index"][0] == 3600000
    
    def test_numeric_index_unchanged(self):
        """Test that numeric indices are not converted to timestamps."""
        # Use plain numeric index
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        # Get chunk should return plain numeric indices
        chunk = dm.get_chunk(0, 5)
        
        # Verify that index is unchanged
        assert chunk["index"] == [0, 1, 2, 3, 4]
    
    def test_unix_timestamp_index_unchanged(self):
        """Test that raw Unix timestamps (already in milliseconds) pass through unchanged."""
        # Use Unix timestamps in milliseconds (like JavaScript Date.now())
        base_ts = 1704067200000  # 2024-01-01 in milliseconds
        index = np.array([base_ts + i * 3600000 for i in range(5)])
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        # Get chunk should return timestamps unchanged
        chunk = dm.get_chunk(0, 5)
        
        # Verify timestamps are preserved
        assert chunk["index"] == index.tolist()
        assert all(isinstance(x, int) for x in chunk["index"])
    
    def test_timezone_aware_index(self):
        """Test that timezone-aware indices are correctly converted to milliseconds."""
        # Create a timezone-aware index (UTC)
        index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        
        dm = DataManager(index, open_data, high, low, close)
        
        # Get chunk should return valid integer timestamps, NOT Timestamps objects
        chunk = dm.get_chunk(0, 5)
        
        # Verify conversion
        assert isinstance(chunk["index"], list)
        assert all(isinstance(x, int) for x in chunk["index"])
        
        # Expected timestamp (1704067200000 for 2024-01-01 UTC)
        expected_ts = 1704067200000
        assert chunk["index"][0] == expected_ts


class TestStyledOverlays:
    """Tests for styled overlay support (new feature)."""

    def test_simple_overlay_becomes_styled(self):
        """Test that simple overlays are converted to styled format with line style."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        overlays = {"SMA20": np.array([101, 102, 102, 103, 103])}

        result = validate_input(index, open_data, high, low, close, overlays=overlays)

        # Simple overlay should be converted to styled format
        assert "SMA20" in result["overlays"]
        assert isinstance(result["overlays"]["SMA20"], dict)
        assert "data" in result["overlays"]["SMA20"]
        assert result["overlays"]["SMA20"]["style"] == "line"
        assert np.array_equal(result["overlays"]["SMA20"]["data"], [101, 102, 102, 103, 103])

    def test_styled_overlay_with_marker(self):
        """Test styled overlay with marker style."""
        index = np.arange(5)
        open_data = np.array([100, 102, 101, 103, 102])
        high = np.array([105, 106, 105, 107, 106])
        low = np.array([99, 100, 99, 101, 100])
        close = np.array([104, 103, 104, 105, 104])
        overlays = {
            "Highlights": {
                "data": np.array([np.nan, 106, np.nan, 107, np.nan]),
                "style": "marker",
                "color": "#00C853",
                "size": 10,
            }
        }

        result = validate_input(index, open_data, high, low, close, overlays=overlays)

        assert "Highlights" in result["overlays"]
        assert result["overlays"]["Highlights"]["style"] == "marker"
        assert result["overlays"]["Highlights"]["color"] == "#00C853"
        assert result["overlays"]["Highlights"]["size"] == 10

    def test_styled_overlay_with_dashed(self):
        """Test styled overlay with dashed line style."""
        index = np.arange(5)
        close = np.array([104, 103, 104, 105, 104])
        overlays = {
            "Threshold": {
                "data": np.array([107, 107, 107, 107, 107]),
                "style": "dashed",
                "color": "#FF0000",
            }
        }

        result = validate_input(index, close, close + 1, close - 1, close, overlays=overlays)

        assert result["overlays"]["Threshold"]["style"] == "dashed"
        assert result["overlays"]["Threshold"]["color"] == "#FF0000"
        assert result["overlays"]["Threshold"]["size"] is None  # Not provided

    def test_mixed_overlay_formats(self):
        """Test mixing simple and styled overlay formats."""
        index = np.arange(5)
        close = np.array([104, 103, 104, 105, 104])
        overlays = {
            "SMA": np.array([101, 102, 102, 103, 103]),  # Simple
            "Markers": {
                "data": np.array([np.nan, 106, np.nan, 107, np.nan]),
                "style": "marker",
            },  # Styled
        }

        result = validate_input(index, close, close + 1, close - 1, close, overlays=overlays)

        assert result["overlays"]["SMA"]["style"] == "line"
        assert result["overlays"]["Markers"]["style"] == "marker"


class TestGroupedSubplots:
    """Tests for grouped subplot support (new feature)."""

    def test_simple_subplot_unchanged(self):
        """Test that simple subplots still work as before."""
        index = np.arange(5)
        close = np.array([104, 103, 104, 105, 104])
        subplots = {"RSI": np.array([55, 58, 52, 60, 57])}

        result = validate_input(index, close, close + 1, close - 1, close, subplots=subplots)

        assert "RSI" in result["subplots"]
        # Simple subplot stays as array, not dict
        assert isinstance(result["subplots"]["RSI"], np.ndarray)
        assert np.array_equal(result["subplots"]["RSI"], [55, 58, 52, 60, 57])

    def test_grouped_subplot(self):
        """Test grouped subplot with multiple series (e.g., Stochastic %K/%D)."""
        index = np.arange(5)
        close = np.array([104, 103, 104, 105, 104])
        subplots = {
            "Stochastic": {
                "%K": np.array([70, 75, 72, 80, 78]),
                "%D": np.array([68, 72, 70, 76, 75]),
            }
        }

        result = validate_input(index, close, close + 1, close - 1, close, subplots=subplots)

        assert "Stochastic" in result["subplots"]
        assert isinstance(result["subplots"]["Stochastic"], dict)
        assert "%K" in result["subplots"]["Stochastic"]
        assert "%D" in result["subplots"]["Stochastic"]
        assert np.array_equal(result["subplots"]["Stochastic"]["%K"], [70, 75, 72, 80, 78])
        assert np.array_equal(result["subplots"]["Stochastic"]["%D"], [68, 72, 70, 76, 75])

    def test_histogram_subplot(self):
        """Test histogram subplot with _type metadata."""
        index = np.arange(5)
        close = np.array([104, 103, 104, 105, 104])
        subplots = {
            "Signals": {
                "_type": "histogram",
                "Positive": np.array([1, 0, 1, 1, 0]),
                "Negative": np.array([0, 1, 0, 0, 1]),
            }
        }

        result = validate_input(index, close, close + 1, close - 1, close, subplots=subplots)

        assert "Signals" in result["subplots"]
        assert result["subplots"]["Signals"]["_type"] == "histogram"
        assert np.array_equal(result["subplots"]["Signals"]["Positive"], [1, 0, 1, 1, 0])
        assert np.array_equal(result["subplots"]["Signals"]["Negative"], [0, 1, 0, 0, 1])

    def test_grouped_subplot_length_validation(self):
        """Test that grouped subplot series are validated for length."""
        index = np.arange(5)
        close = np.array([104, 103, 104, 105, 104])
        subplots = {
            "Stochastic": {
                "%K": np.array([70, 75, 72]),  # Wrong length!
                "%D": np.array([68, 72, 70, 76, 75]),
            }
        }

        with pytest.raises(DataValidationError, match="does not match"):
            validate_input(index, close, close + 1, close - 1, close, subplots=subplots)


class TestDataManagerStyledOverlays:
    """Tests for DataManager with styled overlays."""

    def test_get_chunk_styled_overlay(self):
        """Test get_chunk returns styled overlay metadata."""
        index = np.arange(10)
        close = np.random.randn(10) + 100
        overlays = {
            "Markers": {
                "data": np.array([np.nan] * 5 + [105, 106, 107, 108, 109]),
                "style": "marker",
                "color": "#00FF00",
                "size": 8,
            }
        }

        dm = DataManager(index, close, close + 1, close - 1, close, overlays=overlays)
        chunk = dm.get_chunk(5, 10)

        assert "Markers" in chunk["overlays"]
        assert chunk["overlays"]["Markers"]["style"] == "marker"
        assert chunk["overlays"]["Markers"]["color"] == "#00FF00"
        assert chunk["overlays"]["Markers"]["size"] == 8
        assert len(chunk["overlays"]["Markers"]["data"]) == 5

    def test_get_chunk_simple_overlay_has_style(self):
        """Test that simple overlays in get_chunk have style info."""
        index = np.arange(10)
        close = np.random.randn(10) + 100
        overlays = {"SMA": np.array(range(10))}

        dm = DataManager(index, close, close + 1, close - 1, close, overlays=overlays)
        chunk = dm.get_chunk(0, 5)

        assert chunk["overlays"]["SMA"]["style"] == "line"
        assert chunk["overlays"]["SMA"]["color"] is None
        assert len(chunk["overlays"]["SMA"]["data"]) == 5


class TestDataManagerGroupedSubplots:
    """Tests for DataManager with grouped subplots."""

    def test_get_chunk_grouped_subplot(self):
        """Test get_chunk correctly slices grouped subplots."""
        index = np.arange(10)
        close = np.random.randn(10) + 100
        subplots = {
            "Stochastic": {
                "%K": np.array(range(10)),
                "%D": np.array(range(10, 20)),
            }
        }

        dm = DataManager(index, close, close + 1, close - 1, close, subplots=subplots)
        chunk = dm.get_chunk(2, 5)

        assert "Stochastic" in chunk["subplots"]
        assert chunk["subplots"]["Stochastic"]["%K"] == [2, 3, 4]
        assert chunk["subplots"]["Stochastic"]["%D"] == [12, 13, 14]

    def test_get_chunk_histogram_subplot(self):
        """Test get_chunk preserves _type metadata in histogram subplots."""
        index = np.arange(10)
        close = np.random.randn(10) + 100
        subplots = {
            "Signals": {
                "_type": "histogram",
                "Positive": np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
                "Negative": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            }
        }

        dm = DataManager(index, close, close + 1, close - 1, close, subplots=subplots)
        chunk = dm.get_chunk(0, 4)

        assert chunk["subplots"]["Signals"]["_type"] == "histogram"
        assert chunk["subplots"]["Signals"]["Positive"] == [1, 0, 1, 0]
        assert chunk["subplots"]["Signals"]["Negative"] == [0, 1, 0, 1]

    def test_get_chunk_simple_subplot(self):
        """Test get_chunk with simple (non-grouped) subplot."""
        index = np.arange(10)
        close = np.random.randn(10) + 100
        subplots = {"Volume": np.array(range(100, 110))}

        dm = DataManager(index, close, close + 1, close - 1, close, subplots=subplots)
        chunk = dm.get_chunk(0, 3)

        # Simple subplot is just a list, not a dict
        assert chunk["subplots"]["Volume"] == [100, 101, 102]