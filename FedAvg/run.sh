#!/bin/bash
# Usage: bash run_example.sh <n_clients>
N=${1:-3}
# Start server in background
python server.py --data_csv data/features.csv --rounds 5 --num_clients $N &
SERVER_PID=$!
sleep 2

# Start N clients
for i in $(seq 0 $((N-1))); do
  echo "Starting client $i"
  python client.py --cid $i --n_clients $N --data_csv data/features.csv --local_epochs 2 --iid &
done

wait $SERVER_PID
