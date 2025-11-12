"""
Vercel serverless function wrapper for Flask app.
"""
from web_app import app

# Vercel expects the handler to be named 'handler'
handler = app

