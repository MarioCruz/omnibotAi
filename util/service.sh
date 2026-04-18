#!/bin/bash
# Thin wrapper around systemctl/journalctl for the omniai service.
# Usage: ~/omniai/util/service.sh {start|stop|restart|status|logs|tail}

set -euo pipefail

UNIT=omniai

usage() {
    cat <<EOF
Usage: $0 {start|stop|restart|status|logs|tail}

  start    - sudo systemctl start $UNIT
  stop     - sudo systemctl stop $UNIT
  restart  - sudo systemctl restart $UNIT
  status   - sudo systemctl status $UNIT
  logs     - journalctl -u $UNIT --since '10 min ago'
  tail     - journalctl -u $UNIT -f
EOF
    exit 1
}

[ $# -eq 1 ] || usage

case "$1" in
    start)   sudo systemctl start "$UNIT" ;;
    stop)    sudo systemctl stop "$UNIT" ;;
    restart) sudo systemctl restart "$UNIT" ;;
    status)  sudo systemctl status "$UNIT" ;;
    logs)    journalctl -u "$UNIT" --since '10 min ago' ;;
    tail)    journalctl -u "$UNIT" -f ;;
    *)       usage ;;
esac
