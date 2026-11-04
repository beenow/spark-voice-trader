#!/bin/zsh
# Compile-only build of the debug APK (no emulator). Output: app/build/outputs/apk/debug/app-debug.apk
set -e
cd "$(dirname "$0")"
[ -f app/src/main/jniLibs/arm64-v8a/libllama.so ] || ./ops/build-llama-android.sh
# optional private on-device voice (see app/build.gradle.kts): its native part, when that dir exists
PV=${LEAPS_PRIVATE_VOICE:-$HOME/leaps-voice-agent-private/voice}
if [ -x "$PV/build-phonemizer-android.sh" ]; then
  [ -f app/src/main/jniLibs/arm64-v8a/libphonemizer.so -a -f app/src/main/assets/phonemizer/lexicon.bin ] || "$PV/build-phonemizer-android.sh" "$PWD"
else
  rm -f app/src/main/jniLibs/arm64-v8a/libphonemizer.so; rm -rf app/src/main/assets/phonemizer    # public build: Android TTS
fi
[ -f local.properties ] || printf 'sdk.dir=/opt/homebrew/share/android-commandlinetools\n' > local.properties
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
# default server shown in the app (editable there): $LEAPS_SERVER, else this Mac's Tailscale IPv4 if Tailscale is installed
if [ -z "$LEAPS_SERVER" ]; then
  TS=$( (/Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4 || tailscale ip -4) 2>/dev/null | head -1)
  [ -n "$TS" ] && export LEAPS_SERVER="http://$TS:8310"
fi
./gradlew --no-daemon assembleDebug "$@"
ls -la app/build/outputs/apk/debug/
