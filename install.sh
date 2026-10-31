#!/usr/bin/env bash
# Installs and starts vllm-thermald as a systemd user service running from this directory.
set -euo pipefail

install_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit="$unit_dir/vllm-thermald.service"

case "$install_dir" in
    *[!a-zA-Z0-9_./-]*)
        echo "install path contains characters unsafe in a unit file: $install_dir" >&2
        exit 1
        ;;
esac

python3 "$install_dir/vllm_thermald.py" check-config >/dev/null

systemctl --user disable vllm-thermald.service 2>/dev/null || true
mkdir -p "$unit_dir"
rm -f "$unit"  # an older install may have left a symlink here
sed "s|@INSTALL_DIR@|$install_dir|g" "$install_dir/systemd/vllm-thermald.service.in" > "$unit"
systemctl --user daemon-reload
systemctl --user enable vllm-thermald.service
systemctl --user restart vllm-thermald.service

if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    echo "note: run 'loginctl enable-linger $USER' so the service starts at boot without a login" >&2
fi
echo "installed: $unit"
