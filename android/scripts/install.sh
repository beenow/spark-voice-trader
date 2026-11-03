#!/bin/zsh
# Install the debug APK on the connected phone (USB debugging, or `adb pair` + `adb connect` for wireless).
# The model is NOT pushed: the app downloads it from the step server on first launch (sha256-verified).
set -e
ADB=/opt/homebrew/share/android-commandlinetools/platform-tools/adb
APK=$(cd "$(dirname "$0")/.." && pwd)/app/build/outputs/apk/debug/app-debug.apk
PKG=com.leaps.voiceagent

$ADB devices -l
[ "$($ADB devices | awk 'NR>1 && $2=="device"' | wc -l | tr -d ' ')" = 1 ] || { echo "need exactly one device"; exit 1; }
[ -f "$APK" ] || { echo "no APK; run android/build_apk.sh"; exit 1; }
$ADB install -r -g "$APK"          # -g grants RECORD_AUDIO
[ "$1" = "--usb-reverse" ] && $ADB reverse tcp:8310 tcp:8310 && echo "phone 127.0.0.1:8310 -> Mac :8310"
$ADB shell am start -n $PKG/.MainActivity
cat <<MSG
Installed. On the Mac, start the step server bound to an address the phone can reach (your choice: LAN or
Tailscale IP; 127.0.0.1 with --usb-reverse):
  python -m voice.step_server --host <ip>      (from the repo root, in its venv)
In the app: server http://<ip>:8310, passcode, tap "Get model" (downloads the 808 MB model + 115 MB Kokoro voice once, verifies sha256, loads).
MSG
