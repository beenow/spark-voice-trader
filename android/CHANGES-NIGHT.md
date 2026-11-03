# App-side changes, night of 2026-11-02

Everything was measured on the Galaxy Z Fold (SM-F971U1, Snapdragon 8 Elite Gen 5) with nothing else
running and the CPU at about 48 °C. Turn timings exclude STT. The question was "What is Nvidia trading at?"
unless a row says otherwise.

| # | Change | Before | After |
|---|---|---|---|
| 1 | Kokoro-82M on-device TTS (ONNX Runtime 1.30 XNNPACK + Rust misaki phonemizer), af_heart at 1.05x; Android TTS is now only the fallback. The bundle (115 MB) downloads from :8310 with manifest, sha256, Range resume and the passcode | Android TTS (robotic) | Kokoro on the phone; time to first audio 0.7–1.06 s after the answer arrives |
| 2 | Fillers ("Pulling up Nvidia.") are no longer spoken. Synthesizing them used the CPU at the same time as the model | first audio 2.6 s (filler synth was blocking the answer) | 1.06 s |
| 3 | The first sentence is cut at its first comma (when it is longer than 40 characters), so the first audio unit is shorter | 70-character threshold | 40-character threshold; the first unit renders in about 0.7–1.0 s |
| 4 | llama threads 4 -> 6 (bench: prefill 164 -> 178 tok/s; 8 threads collapses under load) | turn 1.87 s (4 threads, with a bench running alongside) | turn 0.97 s (6 threads, clean) |
| 5 | The system+tools prefix KV (577 tokens) is saved to disk after the first prefill and restored at launch | 2.7 s prefill on every app start | restored in 48 ms; "warm" is skipped |
| 6 | Tried an i8mm build of llama.cpp (armv8.6-a+dotprod+i8mm+fp16) | turn 0.97 s (base build) | turn 1.5 s, two-ticker turn 2.8 s: slower, REVERTED to armv8.2-a+dotprod+fp16 |
| 7 | Per-turn timing breakdown in the feed and in logcat (`LeapsTiming`): prefill/decode per step, HTTP round trips, STT finalisation | - | - |
| 8 | Debug-only soak hook `DebugAskReceiver` -> files/soak.jsonl, processed first in, first out (the first version let broadcasts overtake each other: "start over" ran before the turn sent ahead of it) | out of order | in order |
| 9 | FLAG_KEEP_SCREEN_ON while in the foreground: the phone dozed mid-download and wireless debugging dropped | screen timeout | WindowManager holds a SCREEN_BRIGHT wake lock for the app |
| 10 | The model is chosen by config: the app follows the server's first GGUF (STEP_MODEL_FILES), downloads and verifies it, and only then deletes the old one; `--es model <substring>` pins one | the first GGUF was hard-wired; a failed download deleted the working model | a switch is one env var + an app relaunch |

## Breakdown of one turn after these changes (6 threads, prefix KV reused)

- step 1, tool call: prefill 15 tokens in 165 ms (577 reused), decode 18 tokens in 293 ms
- step 2, answer: prefill 55 tokens in 216 ms (609 reused), decode 14 tokens in 212 ms
- HTTP to :8310 (including the Mac's checks and tools): 32 + 23 + 20 ms
- Kokoro first unit "Nvidia is at about one eighty-five,": 0.99 s synth -> first audio 1.06 s
- Total from end of the question to the first audio: about 2.0 s. The target of 1.0 s is not met: model ~0.9 s + Kokoro ~0.7–1.0 s.

## Note on later timings

After about 20:15 the phone was at 24% battery on a wireless charger. The same question then took 1.7 s per turn
and 1.4–1.6 s to first audio, against 0.97 s and 0.7–1.0 s at 20:09. Compare soak timings with that in mind.

## Open

- Kokoro's first unit takes 0.7–1.0 s for about 2.5 s of audio (RTF ~0.35). Thread and spin settings have not been swept yet.
- The OpenCL Adreno backend was skipped: it needs an OpenCL ICD loader and headers for the NDK build, and its fast kernels target Q4_0, which would mean requantizing (and re-checking agreement on) the model.
