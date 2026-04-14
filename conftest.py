import sys

# Block tensorflow from being imported — it conflicts with protobuf versions
# and is not needed for sentence-transformers / this project.
sys.modules["tensorflow"] = None
