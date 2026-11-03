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

## Second pass, about 20:28–21:00: the spoken path (end of speech -> first audio)

How it was measured: `DebugSpeakReceiver` (debug builds) streams a 16 kHz WAV of the question into the on-device
recognizer (SODA) through `EXTRA_AUDIO_SOURCE` in real time. The end of the file counts as the button release.
logcat `LeapsTiming` prints `release->first audio` with the STT, turn and TTS parts and the battery temperature.
Questions: "What is Nvidia trading at?" (q1), "Compare Apple and Microsoft today." (q2), "What's driving Oracle today?" (q3).
Battery 26–27%, 36–38.6 °C, wireless charging.

| # | Change | Before | After |
|---|---|---|---|
| 11 | Measured the spoken path | q1: release->audio 2951 ms = STT 31 + turn 1502 + Kokoro first 1403 | - |
| 12 | STT: release calls stopListening() at once; the recognizer end-pointing extras (complete / possibly complete silence) are set to 300 ms; the on-device recognizer is preferred (it was already); if the final result is more than 600 ms late, the last partial is used | STT final 25–54 ms after release (on-device SODA) | unchanged: STT is not the bottleneck. The injected-audio finals came back empty, so the partial is now the fallback (it can lose the last word: "What is Nvidia trading") |
| 13 | Kokoro sweep (3 muted turns each; first audio in ms): XNNPACK 4 threads spin 1010–1322 · XNNPACK 4 no spin 1490–1582 · intra-op 1 + XNNPACK 6 1464–1636 · XNNPACK 6 768–894 · CPU 4 931–1037 · CPU 6 787–911 · **CPU 8 605–734** | XNNPACK, 4 threads | CPU EP, 8 threads (default now) |
| 14 | Kokoro model variants at CPU 8 / XNNPACK 6: fp32 644–772 / 712–781 · fp16 760–764 / 722–837 · int8 dynamic 1196–1978 / 2594–2773 · q8f16 27 s (unusable) | uint8f16 605–734 | kept uint8f16 (as fast as fp32, a third of the size) |
| 15 | llama.cpp i8mm build, re-tested with this code (two alternating rounds) | base: turn 897–1312 | i8mm: 892–1407. No gain, stays reverted |
| 16 | Bug: after a reply was cut off (a new press calls stop(), which flushes the track), the audio thread still waited for the old reply's frame count, so the next answer's first audio came up to 13 s late | Oracle turn: first audio 10–13 s | 0.6 s (the reset frame count and the drain both stop on a new generation) |
| 17 | Warm-up at launch: one throwaway token after the restored prefix pages the mmapped weights in (617 ms, off the turn), and Kokoro renders a full-length sentence after "Ready." | first turn after launch: turn 1.6–1.8 s, TTS first 1.8 s | first turn is still slow (1.6 s / 1.8 s): OPEN. Later turns are the steady state |
| 18 | Step-2 prompt: checked what the step server adds. It is only the model's call + `<|eot_id|>` + the ipython header + the tool result + the assistant header (55 new tokens for a quote) | - | nothing redundant to cut without retraining the format. The tool-call decode stops at end-of-generation (18 tokens for one quote, 35 for two) |

Steady state after the changes, release -> first audio:

| Question | Before (20:33) | After (20:59) |
|---|---|---|
| q1 Nvidia | 2951 ms | 1930 ms (STT 46 + turn 1202 + TTS 677) |
| q3 Oracle | - | 1837 / 2252 ms (TTS 605 / 638) |
| q2 two tickers | - | 2920 ms (turn 2225: two calls + a longer answer to decode) |

The ≤2.0 s target is met for single-ticker questions on a warm phone. The ≤1.5 s stretch is not met. What is left:
the model's ~1.2 s (prefill is ~20 ms/token at small batch sizes, the same as decode) and Kokoro's ~0.6 s for the
first clause. Later turns in a session also prefill the conversation memory (90–149 new tokens in step 1, +200–300 ms).

Not done:
- Kokoro on NNAPI/QNN: no quick path. The uint8f16 graph is mostly CPU-EP ops, and QNN needs a separate onnxruntime-qnn AAR plus a QDQ model.
- Pinning Kokoro and llama to different cores: they no longer run at the same time (fillers are off and the turn finishes before synthesis starts).
- Synthesizing the first clause while step 2 decodes: the answer is a template that the Mac fills and validates only once it is complete, and running Kokoro alongside the decode would compete for the same cores.
- A phrase cache: the common replies are filled templates with numbers, so little would hit it.

## Third pass, about 21:00–21:10

| # | Change | Before | After |
|---|---|---|---|
| 19 | REGRESSION FIX (from #12): the 300 ms silence end-pointing ended the owner's utterance at a pause while the button was still held ("Anything that stopped" was answered; he meant "Anything that stops bleeding"). The silence extras are now 10 s, and only the release finalises. If the recognizer still ends while the button is held (a final, NO_MATCH or SPEECH_TIMEOUT), it is restarted and the text so far is kept. After a pause, a new segment's partials no longer contain the earlier words; that segment is detected (shorter, different first word) and the earlier words are kept | "What is Nvidia ⟨1.6 s pause⟩ trading at?" -> heard "Trading" | heard "What is Nvidia Trading", answered correctly; release->audio 1.9–2.4 s |
| 20 | STT quality (logged only): the owner's first spoken turn was heard as "Can you help me with life". No change made | - | - |
| 21 | Glasses (Bluetooth headset) support, see README "Glasses". Tested on the phone mic only: the AudioRecord -> PCM pipe -> on-device recognizer path, and the media-button tap (KEYCODE_MEDIA_PLAY_PAUSE / HEADSETHOOK) that starts and stops listening. NOT tested with the glasses: the SCO/LE link time, routing and touchpad events need the owner | - | phone-mic capture path: release->first audio 1.62 s |

## Note on later timings

After about 20:15 the phone was at 24% battery on a wireless charger. The same question then took 1.7 s per turn
and 1.4–1.6 s to first audio, against 0.97 s and 0.7–1.0 s at 20:09. Compare soak timings with that in mind.

## Open

- Kokoro's first unit takes 0.7–1.0 s for about 2.5 s of audio (RTF ~0.35). Thread and spin settings have not been swept yet.
- The OpenCL Adreno backend was skipped: it needs an OpenCL ICD loader and headers for the NDK build, and its fast kernels target Q4_0, which would mean requantizing (and re-checking agreement on) the model.
