import os
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn, TCPServer


PORT = int(os.getenv("PORT", "3000"))
DIRECTORY = os.path.dirname(__file__)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)


class ThreadingHTTPServer(ThreadingMixIn, TCPServer):
    daemon_threads = True


with ThreadingHTTPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"Frontend is serving on 0.0.0.0:{PORT} (cwd={DIRECTORY})")
    httpd.serve_forever()
