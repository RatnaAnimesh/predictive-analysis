import spacy
import sys

print(f"Python version: {sys.version}")
print(f"spaCy version: {spacy.__version__}")

print("\n--- Checking for Apple Silicon (MPS) support ---")
try:
    import thinc_apple_ops
    print("Successfully imported 'thinc_apple_ops'.")
except ImportError:
    print("ERROR: Could not import 'thinc_apple_ops'. The Apple Silicon specific library is not correctly installed.")
    sys.exit(1)

# Check what spaCy's internal require_gpu() function says
print("\n--- Running spaCy's GPU requirement check ---")
try:
    spacy.require_gpu()
    print("spacy.require_gpu() check PASSED.")
except Exception as e:
    print(f"spacy.require_gpu() FAILED with an error: {e}")

# Check the active Thinc backend
from thinc.api import get_current_ops

ops = get_current_ops()
print(f"\n--- Active Thinc backend ---")
print(f"Current Thinc Ops: {ops.__class__.__name__}")
if "mps" in ops.__class__.__name__.lower():
    print("SUCCESS: The Metal Performance Shaders (MPS) backend is active.")
else:
    print("INFO: The active backend is not the Apple MPS backend. GPU is not being used.")
