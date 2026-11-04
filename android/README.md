# Leaps on-device (Android)

A hands-free voice trading agent whose language model, a fine-tuned **Llama 3.2 1B**, runs **on the
phone** (llama.cpp, CPU, arm64), inside a small trading app, **Leaps Trader**, whose screens the agent opens as it
answers. Built for a hackathon demo on a Galaxy Z Fold.

## What runs where (honest version)

| Step | Where | How |
|---|---|---|
| Speech to text | **phone** | Android `SpeechRecognizer`; the on-device recognizer when the phone has one (offline), else the system recognizer asked to prefer offline |
| The model: every token of every tool call and answer | **phone** | `llama-1b-loop-Q4_K_M.gguf` (Llama 3.2 1B + our LoRA, fused, 808 MB) through llama.cpp over JNI |
| Text to speech | **phone** | Android `TextToSpeech` in this repo. The demo video used a proprietary on-device Kokoro-82M voice (prior work, not included), loaded through `OnDeviceVoice.kt` when a build has it; see "Voice" below |
| Prompt rendering, output parsing, validators/repairs, tool execution (mock market), slot fill, conversation memory | **Mac** (step server, `voice/step_server.py`, port 8310) | the same Python code the model was evaluated with, imported unchanged |

The split is deliberate: the phone does all the *generation*, the Mac keeps the code checks identical to the
evaluated path (no Kotlin re-implementation of the validators to drift). The app's "where it ran" chips light
green for phone steps and blue for Mac steps, and every turn shows on-phone model ms, tokens/s, Mac+network ms
and time to first audio.

Protocol: the phone posts `{session, question}`; the Mac answers `{prompt, tokens}` (the next prompt, already
tokenized); the phone generates and posts `{session, model_output}`; repeat until the Mac answers
`{say, display, events}`. The fixed system+tools prefix (577 tokens) is prefilled once per app start
("warm") and its KV cache is reused, so a turn only processes its new tokens.

## Measured on a Galaxy Z Fold (Snapdragon 8 Elite Gen 5, 6 CPU threads, phone cool)

- Model load 1.6 s. The 577-token system+tools prefix is prefilled once (2.7 s), saved to disk, and restored in 48 ms on later launches.
- Turn "What is Nvidia trading at?": 0.97 s. Step 1 (tool call) prefills 15 new tokens in 165 ms and decodes 18 in 293 ms. Step 2 (answer) prefills 55 in 216 ms and decodes 14 in 212 ms. HTTP to the Mac, including its checks and tools, is ~75 ms.
- Kokoro: first audio 0.7-1.06 s after the answer arrives (the first clause is synthesized first; RTF ~0.35).
- Two-ticker comparison: 2.1 s turn.
- See CHANGES-NIGHT.md for what was tried (an i8mm build was slower; 8 threads collapses under load).

## Glasses (Meta Ray-Ban or any Bluetooth headset)

Pair the glasses with the phone as a normal Bluetooth headset. When the app sees a Bluetooth SCO or LE Audio
headset, a "Glasses · <name>" badge appears and:

- **Mic:** the app records from the glasses' microphone itself (AudioRecord, VOICE_COMMUNICATION, preferred device =
  the glasses) and streams the PCM into the on-device recognizer (`EXTRA_AUDIO_SOURCE`). The system recognizer
  does not reliably follow the Bluetooth route on its own, so the app does not rely on it. The log shows the
  device the capture was really routed to.
- **Hands-free:** one tap (the on-screen button, or the glasses' touchpad, which arrives as play/pause or
  headset-hook on a MediaSession) starts a conversation. A question ends after about 1.5 s of quiet (the app
  records the mic itself and hands the audio to each recognizer session, so a pause never loses words), and the mic
  reopens by itself once the answer has been spoken. It is half duplex: to talk over an answer, tap (the answer
  stops and the app listens). Tap while listening sends what was heard at once, or ends the conversation if nothing
  was heard yet.
- **Speaker:** with Glasses mode OFF (the default), the Bluetooth call link (SCO/LE) is brought up only while
  listening and dropped as soon as the words are in. The answer then plays as media (A2DP) on the glasses in full
  quality. With Glasses mode ON (Setup), the link stays up for the whole session: no per-turn switch-on cost
  (often 0.5-1 s), but answers play on the narrow-band call path.
- **Logs:** `adb logcat -s LeapsGlasses LeapsStt LeapsTiming` shows the device added/removed, the link time
  ("acquire ... link up in N ms"), the capture route, the end of speech, the playback route, each tap, and
  release->first audio with the glasses details.

## Agreement with the trained model

The GGUF was checked against the MLX model (base + adapter) on 30 dev-split prompts, greedy, with the prompt
rendered by `agent/format.py`: tokenization identical 30/30, tool calls identical **20/20** for both Q8_0 and
Q4_K_M; free-text answers identical 7/10 (Q4_K_M) and 8/10 (Q8_0). Q4_K_M ships (smaller, faster).

## Build it yourself

Needs: Android SDK + NDK 29 (`ANDROID_HOME`), JDK 17, CMake.

```sh
./scripts/fetch_llama_cpp.sh              # pinned llama.cpp into third_party/
./ops/build-llama-android.sh              # llama.cpp -> app/src/main/jniLibs/arm64-v8a
LEAPS_SERVER=http://<mac-ip>:8310 ./build_apk.sh   # compile only; -> app/build/outputs/apk/debug/app-debug.apk
./scripts/install.sh                      # adb install -r -g + launch
```

## Mac side

Put the files in `<home>` (`$VOICE_HOME`, default `~/.leaps-voice-agent`):

- `gguf/llama-1b-loop-Q4_K_M.gguf`: the fused model (convert with llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`).
- `kokoro-onnx/{model.onnx,voice.bin,tokenizer.json}`: Kokoro-82M v1.0 ONNX (`onnx/model_uint8f16.onnx`
  from `onnx-community/Kokoro-82M-v1.0-ONNX`, Apache-2.0), the `af_heart` voice as raw float32 [510,256],
  and the model's tokenizer.json.
- `.passcode`: one line; the app asks for it once.

```sh
python -m voice.step_server --host <lan-or-tailscale-ip>     # never a public tunnel
```

`STEP_MODEL_FILES` (comma-separated GGUF names in `<home>/gguf`) chooses what is offered; the first is the
default, and the app switches to it on its next launch (`adb shell am start ... --es model <name part>` pins one).
`scripts/start_step_server.sh <ip> [repo] [python]` starts it detached.

All `/step`, `/prefix` and `/model/*` requests need the `X-Passcode` header. `GET /model/manifest.json` lists
each file with size and sha256; `GET /model/<name>` supports `Range`, so an interrupted download resumes. On
first launch the app downloads the GGUF (808 MB) and the Kokoro bundle (115 MB), verifies the sha256 of each,
then loads them. Over USB instead of Wi-Fi: `./scripts/install.sh --usb-reverse` and server `http://127.0.0.1:8310`.

## Code

- `app/src/main/cpp/agent_llama.cpp`, `LlamaBridge.kt`: llama.cpp JNI shim (KV-prefix reuse, greedy decode with
  the same stop rules as the Mac backend).
- `StepClient.kt`, `ModelStore.kt`: step protocol; manifest download with resume + sha256.
- `KokoroSpeaker.kt`, `OnDeviceVoice.kt`: speech out, sentence by sentence, with first-audio timing; Android
  `TextToSpeech` unless the build includes an on-device voice.

### Voice

The neural voice heard in the demo video (Kokoro-82M on the phone, with a phonemizer and pronunciation lexicon from
the Leapsify app) is prior, proprietary work and is not in this repository. `OnDeviceVoice.kt` is the seam: a build
that adds a `com.leaps.voiceagent.privatevoice.Provider` source dir (see `app/build.gradle.kts`) uses it; without one,
every answer is spoken by Android `TextToSpeech`. Timings in this README that mention Kokoro were measured with it.
- `MainActivity.kt`: the single-activity UI.
- `src/debug/.../DebugAskReceiver.kt` (debug builds only): `adb shell am broadcast -n com.leaps.voiceagent/.DebugAskReceiver
  --es q "..." [--es session id] [--ez mute true]` runs a typed question through the same pipeline (STT skipped) and
  appends one JSON line of timings and results to `files/soak.jsonl`.
