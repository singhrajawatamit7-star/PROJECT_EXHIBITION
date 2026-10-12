import unittest
import json
import io
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.health import handler


class DummySocket:
    def __init__(self):
        self.output = io.BytesIO()

    def makefile(self, *args, **kwargs):
        return self.output


class TestApiHealth(unittest.TestCase):
    def test_health_handler_response(self):
        # Verify handler structure
        self.assertTrue(hasattr(handler, "do_GET"))
        self.assertTrue(hasattr(handler, "do_OPTIONS"))

    def test_mock_get(self):
        # Create a mock handler instance without spinning up full network socket
        class MockHandler(handler):
            def __init__(self):
                self.rfile = io.BytesIO(b"GET /api/health HTTP/1.1\r\n\r\n")
                self.wfile = io.BytesIO()
                self.headers_sent = {}
                self.status_code = None

            def send_response(self, code, message=None):
                self.status_code = code

            def send_header(self, keyword, value):
                self.headers_sent[keyword] = value

            def end_headers(self):
                pass

        mock = MockHandler()
        mock.do_GET()

        self.assertEqual(mock.status_code, 200)
        self.assertEqual(mock.headers_sent.get("Content-Type"), "application/json")
        response_body = mock.wfile.getvalue().decode("utf-8")
        data = json.loads(response_body)
        self.assertEqual(data.get("status"), "online")
        self.assertEqual(data.get("service"), "driver-drowsiness-ai")


if __name__ == "__main__":
    unittest.main()
