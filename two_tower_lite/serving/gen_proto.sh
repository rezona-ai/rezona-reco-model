#!/usr/bin/env bash
# Generate Python gRPC stubs from twotower_ranker.proto.
# Run once after install: pip install grpcio-tools
set -euo pipefail
cd "$(dirname "$0")"
python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  twotower_ranker.proto
echo "generated twotower_ranker_pb2.py and twotower_ranker_pb2_grpc.py"
