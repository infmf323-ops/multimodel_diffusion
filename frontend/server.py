from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os


PORT = int(os.getenv("PORT", "3000"))
DIRECTORY = os.path.dirname(__file__)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


with TCPServer(("", PORT), Handler) as httpd:
    print(f"Frontend is serving on port {PORT}")
    httpd.serve_forever()
