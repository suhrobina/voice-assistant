#!/usr/bin/env bash
# Start phone mode (./run.sh --web) automatically at boot, as a system service.
#   sudo ./install-service.sh              install, enable at boot and start now
#   sudo ./install-service.sh --uninstall  stop and remove
set -e
cd "$(dirname "$0")"
DIR=$(pwd)
NAME=voice-assistant
UNIT=/etc/systemd/system/$NAME.service

if [ "$(id -u)" -ne 0 ]; then
  echo "Needs root: sudo $0 $*"
  exit 1
fi

if [ "$1" = "--uninstall" ]; then
  systemctl disable --now "$NAME" 2>/dev/null || true
  rm -f "$UNIT"
  systemctl daemon-reload
  echo "Removed. Start phone mode by hand again with ./run.sh --web"
  exit 0
fi

# Runs as the user who called sudo (their models, memory, profiles and local.env)
RUN_AS=${SUDO_USER:-$(stat -c %U "$DIR")}

# Only one copy fits on the GPU
if pgrep -u "$RUN_AS" -f "python (assistant|web)\.py" >/dev/null; then
  echo "The assistant is running already (./run.sh). Stop it (Ctrl+C), then run this again."
  exit 1
fi

sed -e "s|@USER@|$RUN_AS|g" -e "s|@DIR@|$DIR|g" voice-assistant.service > "$UNIT"
systemctl daemon-reload
systemctl enable --now "$NAME"

PORT=$(grep -E '^PORT=' local.env 2>/dev/null | cut -d= -f2)
echo
echo "Installed: phone mode now starts at boot (as $RUN_AS)."
echo "  Open on your phone:  https://$(hostname -I | awk '{print $1}'):${PORT:-8443}  (ready in ~30 s)"
echo "  Status:   systemctl status $NAME"
echo "  Live log: journalctl -u $NAME -f"
echo "  Stop / start / restart:  sudo systemctl stop|start|restart $NAME"
echo "  Remove:   sudo $0 --uninstall"
