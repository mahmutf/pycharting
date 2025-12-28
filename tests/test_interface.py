"""Tests for Python API interface."""

import pytest
import numpy as np
import socket
import time
from unittest.mock import patch, MagicMock

from pycharting.api.interface import plot, stop_server, get_server_status, _active_server, _get_local_ip
from pycharting.api.routes import _data_managers


@pytest.fixture(autouse=True)
def cleanup_globals():
    """Clean up global state between tests."""
    global _active_server
    
    # Stop any active server before test
    if _active_server and _active_server.is_running:
        _active_server.stop_server()
    
    # Clear data managers
    _data_managers.clear()
    
    yield
    
    # Clean up after test
    if _active_server and _active_server.is_running:
        _active_server.stop_server()
    _data_managers.clear()


class TestPlotFunction:
    """Tests for the main plot() function."""
    
    def test_plot_basic_usage(self):
        """Test basic plot creation with minimal arguments."""
        # Generate sample data
        n = 100
        index = np.arange(n)
        close = np.cumsum(np.random.randn(n)) + 100
        open_data = close + np.random.randn(n) * 0.5
        high = np.maximum(open_data, close) + np.abs(np.random.randn(n))
        low = np.minimum(open_data, close) - np.abs(np.random.randn(n))
        
        # Create chart without opening browser
        result = plot(
            index, open_data, high, low, close,
            open_browser=False
        )
        
        assert result['status'] == 'success'
        assert 'url' in result
        assert result['data_points'] == n
        assert result['server_running'] is True
        assert 'default' in result['session_id']
    
    def test_plot_with_custom_session(self):
        """Test plot with custom session ID."""
        n = 50
        index = np.arange(n)
        data = np.random.randn(n) + 100
        
        result = plot(
            index, data, data + 1, data - 1, data,
            session_id='custom_session',
            open_browser=False
        )
        
        assert result['status'] == 'success'
        assert result['session_id'] == 'custom_session'
        assert 'custom_session' in _data_managers
    
    def test_plot_with_overlays(self):
        """Test plot with overlay data."""
        n = 100
        index = np.arange(n)
        close = np.cumsum(np.random.randn(n)) + 100
        open_data = close + np.random.randn(n) * 0.5
        high = np.maximum(open_data, close) + np.abs(np.random.randn(n))
        low = np.minimum(open_data, close) - np.abs(np.random.randn(n))
        
        # Add moving average overlay
        ma = np.convolve(close, np.ones(10)/10, mode='same')
        
        result = plot(
            index, open_data, high, low, close,
            overlays={'MA10': ma},
            open_browser=False
        )
        
        assert result['status'] == 'success'
        # Check that data manager has overlay
        dm = _data_managers['default']
        assert 'MA10' in dm.overlays
    
    def test_plot_reuses_server(self):
        """Test that multiple plots reuse the same server."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)
        
        # First plot
        result1 = plot(
            index, data, data + 1, data - 1, data,
            session_id='session1',
            open_browser=False
        )
        
        first_server_url = result1['server_url']
        
        # Second plot should reuse server
        result2 = plot(
            index, data, data + 1, data - 1, data,
            session_id='session2',
            open_browser=False
        )
        
        assert result2['server_url'] == first_server_url
        assert result1['status'] == 'success'
        assert result2['status'] == 'success'
    
    def test_plot_with_invalid_data(self):
        """Test plot with invalid data returns error."""
        # Try with mismatched array lengths
        result = plot(
            np.arange(10),
            np.random.randn(10),
            np.random.randn(10),
            np.random.randn(10),
            np.random.randn(5),  # Wrong length!
            open_browser=False
        )
        
        assert result['status'] == 'error'
        assert 'error' in result
    
    @patch('webbrowser.open')
    def test_plot_opens_browser(self, mock_browser):
        """Test that plot opens browser when requested."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)
        
        result = plot(
            index, data, data + 1, data - 1, data,
            open_browser=True
        )
        
        assert result['status'] == 'success'
        # Check that browser.open was called
        mock_browser.assert_called_once()
        call_args = mock_browser.call_args[0][0]
        assert 'http://' in call_args
    
    @patch('webbrowser.open', side_effect=Exception("Browser error"))
    def test_plot_handles_browser_error(self, mock_browser):
        """Test that plot handles browser opening errors gracefully."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)
        
        # Should still succeed even if browser fails
        result = plot(
            index, data, data + 1, data - 1, data,
            open_browser=True
        )
        
        assert result['status'] == 'success'
        assert result['server_running'] is True


class TestStopServer:
    """Tests for stop_server() function."""
    
    def test_stop_server_when_running(self):
        """Test stopping an active server."""
        # Start a server first
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)
        
        plot(index, data, data + 1, data - 1, data, open_browser=False)
        
        # Now stop it
        stop_server()
        
        status = get_server_status()
        assert status['running'] is False
    
    def test_stop_server_when_not_running(self):
        """Test stopping when no server is active."""
        # Should not raise an error
        stop_server()  # Should print info message


class TestGetServerStatus:
    """Tests for get_server_status() function."""
    
    def test_status_when_no_server(self):
        """Test status when no server has been started."""
        status = get_server_status()
        
        assert status['running'] is False
        # server_info may exist from previous server, just check it's not running
        assert status['active_sessions'] == 0
    
    def test_status_when_server_running(self):
        """Test status when server is active."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)
        
        # Start server
        plot(index, data, data + 1, data - 1, data, open_browser=False)
        
        status = get_server_status()
        
        assert status['running'] is True
        assert status['server_info'] is not None
        assert status['active_sessions'] >= 1
        assert 'host' in status['server_info']
        assert 'port' in status['server_info']


class TestDataTypes:
    """Tests for different data type inputs."""
    
    def test_plot_with_numpy_arrays(self):
        """Test plot with NumPy arrays."""
        n = 50
        index = np.arange(n)
        close = np.random.randn(n) + 100
        
        result = plot(
            index, close, close + 1, close - 1, close,
            open_browser=False
        )
        
        assert result['status'] == 'success'
    
    def test_plot_with_lists(self):
        """Test plot with Python lists."""
        n = 50
        index = list(range(n))
        close = [100 + i * 0.1 for i in range(n)]
        
        result = plot(
            index, close, 
            [c + 1 for c in close],
            [c - 1 for c in close],
            close,
            open_browser=False
        )
        
        assert result['status'] == 'success'


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow(self):
        """Test complete workflow: plot -> check status -> stop."""
        # Generate data
        n = 100
        index = np.arange(n)
        close = np.cumsum(np.random.randn(n)) + 100
        open_data = close + np.random.randn(n) * 0.5
        high = np.maximum(open_data, close) + np.abs(np.random.randn(n))
        low = np.minimum(open_data, close) - np.abs(np.random.randn(n))
        
        # 1. Create chart
        result = plot(
            index, open_data, high, low, close,
            session_id='workflow_test',
            open_browser=False
        )
        assert result['status'] == 'success'
        
        # 2. Check status
        status = get_server_status()
        assert status['running'] is True
        assert status['active_sessions'] >= 1
        
        # 3. Stop server
        stop_server()
        
        # 4. Verify stopped
        status = get_server_status()
        assert status['running'] is False
    
    def test_multiple_sessions(self):
        """Test creating multiple chart sessions."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)
        
        # Create multiple sessions
        for i in range(3):
            result = plot(
                index, data, data + 1, data - 1, data,
                session_id=f'session_{i}',
                open_browser=False
            )
            assert result['status'] == 'success'
        
        # Check all sessions exist
        assert len(_data_managers) >= 3
        for i in range(3):
            assert f'session_{i}' in _data_managers


class TestGetLocalIP:
    """Tests for the _get_local_ip() helper function."""

    def test_get_local_ip_returns_string(self):
        """Test that _get_local_ip returns a valid IP string."""
        ip = _get_local_ip()
        assert isinstance(ip, str)
        # Should be a valid IP format (either localhost or actual IP)
        parts = ip.split('.')
        assert len(parts) == 4
        for part in parts:
            assert part.isdigit()
            assert 0 <= int(part) <= 255

    @patch('socket.socket')
    def test_get_local_ip_fallback_on_socket_error(self, mock_socket_class):
        """Test fallback when socket connection fails."""
        # Mock socket to raise an exception
        mock_socket = MagicMock()
        mock_socket.connect.side_effect = OSError("Network error")
        mock_socket_class.return_value = mock_socket

        # Should fall back to hostname resolution or localhost
        with patch('socket.gethostname', return_value='testhost'):
            with patch('socket.gethostbyname', return_value='192.168.1.100'):
                ip = _get_local_ip()
                assert ip == '192.168.1.100'

    @patch('socket.socket')
    def test_get_local_ip_final_fallback_to_localhost(self, mock_socket_class):
        """Test final fallback to 127.0.0.1 when all methods fail."""
        # Mock socket to fail
        mock_socket = MagicMock()
        mock_socket.connect.side_effect = OSError("Network error")
        mock_socket_class.return_value = mock_socket

        # Mock hostname resolution to also fail
        with patch('socket.gethostname', side_effect=OSError("Hostname error")):
            ip = _get_local_ip()
            assert ip == '127.0.0.1'

    @patch('socket.socket')
    def test_get_local_ip_ignores_localhost_from_hostname(self, mock_socket_class):
        """Test that localhost IPs from hostname resolution are skipped."""
        # Mock socket to fail
        mock_socket = MagicMock()
        mock_socket.connect.side_effect = OSError("Network error")
        mock_socket_class.return_value = mock_socket

        # Mock hostname to return localhost
        with patch('socket.gethostname', return_value='localhost'):
            with patch('socket.gethostbyname', return_value='127.0.0.1'):
                ip = _get_local_ip()
                # Should return localhost as final fallback (since hostname returned 127.x)
                assert ip == '127.0.0.1'


class TestPlotHostParameters:
    """Tests for host and external_host parameters in plot()."""

    def test_plot_default_host(self):
        """Test that default host is 127.0.0.1."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)

        result = plot(
            index, data, data + 1, data - 1, data,
            open_browser=False,
            block=False
        )

        assert result['status'] == 'success'
        # Default should bind to localhost
        assert '127.0.0.1' in result['server_url'] or 'localhost' in result['server_url']

    @patch('webbrowser.open')
    def test_plot_with_custom_host(self, mock_browser):
        """Test plot with custom host parameter."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)

        # Stop any existing server first to test fresh server with new host
        stop_server()

        result = plot(
            index, data, data + 1, data - 1, data,
            host="127.0.0.1",  # Explicitly set host
            open_browser=False,
            block=False
        )

        assert result['status'] == 'success'
        assert '127.0.0.1' in result['server_url']

    @patch('webbrowser.open')
    def test_plot_with_external_host_override(self, mock_browser):
        """Test plot with external_host parameter overriding display URL."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)

        # Stop any existing server first
        stop_server()

        result = plot(
            index, data, data + 1, data - 1, data,
            host="127.0.0.1",
            external_host="myhost.local",
            open_browser=True,
            block=False
        )

        assert result['status'] == 'success'
        # Display URL should use external_host
        assert 'myhost.local' in result['url']
        assert 'myhost.local' in result['server_url']

        # Browser should be called with external_host URL
        mock_browser.assert_called_once()
        call_url = mock_browser.call_args[0][0]
        assert 'myhost.local' in call_url

    def test_plot_url_contains_session_id(self):
        """Test that the chart URL includes session ID parameter."""
        n = 50
        data = np.random.randn(n) + 100
        index = np.arange(n)

        result = plot(
            index, data, data + 1, data - 1, data,
            session_id='test_session_123',
            open_browser=False,
            block=False
        )

        assert result['status'] == 'success'
        assert 'session=test_session_123' in result['url']
