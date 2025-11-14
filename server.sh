#!/bin/bash

# Server management script
# Usage: ./server.sh [start|stop|restart|status]

PORT=8000
OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key}"
CONFIG_NAME="${CONFIG_NAME:-prod}"

# Function to check if server is running
check_status() {
    PIDS=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "Server is running on port $PORT (PID: $PIDS)"
        return 0
    else
        echo "Server is not running"
        return 1
    fi
}

# Function to start server
start_server() {
    if check_status >/dev/null 2>&1; then
        echo "Server is already running on port $PORT"
        return 1
    fi
    
    echo "Starting server on port $PORT..."
    OPENAI_API_KEY="$OPENAI_API_KEY" CONFIG_NAME="$CONFIG_NAME" nohup uvicorn src.main:app --port $PORT --host 0.0.0.0 --workers 10 > server_log.log 2>&1 &
    
    # Wait longer for server to fully initialize
    echo "Waiting for server to initialize..."
    for i in {1..15}; do
        sleep 2
        if check_status >/dev/null 2>&1; then
            echo "Server started successfully"
            return 0
        fi
        echo -n "."
    done
    
    echo ""
    echo "Server startup timeout - but it may still be initializing"
    echo "Check server_log.log for details or run './server.sh status' to verify"
    echo "If server is running, this is just a detection issue"
    return 0
}

# Function to stop server
stop_server() {
    PIDS=$(lsof -ti:$PORT 2>/dev/null)
    
    if [ -z "$PIDS" ]; then
        echo "No process found listening on port $PORT"
        return 1
    fi
    
    echo "Stopping server processes on port $PORT..."
    echo "Process IDs: $PIDS"
    kill -TERM $PIDS
    
    # Wait for graceful shutdown
    sleep 3
    
    # Check if processes are still running
    REMAINING_PIDS=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$REMAINING_PIDS" ]; then
        echo "Forcing termination of remaining processes..."
        kill -9 $REMAINING_PIDS
    fi
    
    echo "Server stopped"
}

# Function to restart server
restart_server() {
    echo "Restarting server..."
    stop_server
    sleep 1
    start_server
}

# Main script logic
case "${1:-}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the server"
        echo "  stop    - Stop the server"
        echo "  restart - Restart the server"
        echo "  status  - Check server status"
        echo ""
        echo "Environment variables:"
        echo "  OPENAI_API_KEY - OpenAI API key (default: your-api-key)"
        echo "  CONFIG_NAME    - Configuration name (default: prod)"
        exit 1
        ;;
esac