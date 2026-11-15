"""
Root entry point for Streamlit Cloud.
Adds the repo root to sys.path so that `from src.X import Y` works,
then runs the actual UI module.
"""
import sys
import os

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now execute the real app
exec(open(os.path.join(os.path.dirname(__file__), "src", "ui", "app.py")).read())
