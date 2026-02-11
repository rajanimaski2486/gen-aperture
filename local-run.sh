#!/bin/bash
# Gen-Aperture: Start backend + frontend locally

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill anything already on the ports
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :5173 | xargs kill -9 2>/dev/null || true
echo "✅ Ports 8000 & 5173 cleared"

# Start backend
echo "🚀 Starting backend..."
cd "$ROOT_DIR/backend"
source venv/bin/activate
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "🚀 Starting frontend..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=================================="
echo "  Gen-Aperture is running!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "=================================="
echo "Press Ctrl+C to stop both servers"
echo ""

# Trap Ctrl+C to kill both
cleanup() {
  echo ""
  echo "🛑 Shutting down..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  lsof -ti :8000 | xargs kill -9 2>/dev/null || true
  lsof -ti :5173 | xargs kill -9 2>/dev/null || true
  echo "✅ Stopped"
  exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for both
wait
