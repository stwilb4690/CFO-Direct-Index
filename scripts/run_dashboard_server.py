"""
Dashboard Web Server for CFO Direct Index Forward Test.

A simple Flask server that:
1. Serves the dashboard at http://localhost:5000
2. Provides API endpoints for portfolio data
3. Can be run as a background service

Usage:
    python scripts/run_dashboard_server.py
    
Then open: http://localhost:5000
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, send_from_directory, jsonify

# Import data generator
from scripts.generate_dashboard_data import generate_dashboard_data

app = Flask(__name__, 
            static_folder=str(Path(__file__).parent.parent / "dashboard"),
            static_url_path='')


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/data/portfolio.json')
def portfolio_data():
    """Serve fresh portfolio data as JSON."""
    try:
        data = generate_dashboard_data("forward_10mm")
        if data:
            return jsonify(data)
        else:
            return jsonify({"error": "Portfolio not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/<path:path>')
def static_files(path):
    """Serve static files (CSS, JS, etc)."""
    return send_from_directory(app.static_folder, path)


def main():
    """Run the dashboard server."""
    print("=" * 60)
    print("CFO Direct Index - Dashboard Server")
    print("=" * 60)
    print()
    print("Starting server at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()
