#!/bin/bash

# Script to gracefully restart the proxy server

echo "Finding proxy server process..."
PID=$(ps aux | grep "python3 proxy_server.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "No proxy server process found."
else
    echo "Found proxy server with PID: $PID"
    echo "Sending SIGTERM to gracefully shutdown..."
    kill -TERM $PID

    # Wait for process to exit
    echo "Waiting for process to exit..."
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "Process exited gracefully"
            break
        fi
        sleep 1
    done

    # Force kill if still running
    if ps -p $PID > /dev/null 2>&1; then
        echo "Process did not exit, forcing shutdown..."
        kill -9 $PID
        sleep 2
    fi
fi

echo "Starting proxy server..."
cd "$(dirname "$0")"
nohup python3 proxy_server.py --config config.json > logs/proxy.log 2>&1 &
NEW_PID=$!

echo "Proxy server started with PID: $NEW_PID"
echo "Waiting for server to initialize..."
sleep 3

# Test health endpoint
if curl -s http://127.0.0.1:3001/health > /dev/null 2>&1; then
    echo "✓ Health check passed - server is running"
else
    echo "✗ Health check failed - server may not be ready yet"
    echo "Check logs/proxy.log for details"
fi

echo ""
echo "Server status:"
ps aux | grep "python3 proxy_server.py" | grep -v grep
