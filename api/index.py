"""
Cloud Serverless API Entrypoint (Vercel, AWS Lambda, Render)
Routes incoming requests to the Edge-AI telemetry WSGI dashboard application.
"""

import sys
from pathlib import Path

# Add project root and app directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
app_dir = root_dir / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from app.main import app

# Standard WSGI callable exported for cloud serverless platforms
handler = app
application = app
