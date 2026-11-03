package com.leaps.voiceagent

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import org.json.JSONObject
import java.io.File

/**
 * DEBUG BUILDS ONLY. Runs a typed question through the same pipeline as a spoken one (STT skipped):
 * on-device LLM -> step server :8310 -> Kokoro. One JSON line per turn is appended to files/soak.jsonl.
 *
 *   adb shell am broadcast -n com.leaps.voiceagent/.DebugAskReceiver --es q "What is Nvidia at?" \
 *       [--es session soak-1] [--ez mute true]
 *
 * The app must be open (it owns the loaded model). Same session id = multi-turn; "start over" resets
 * memory through the agent's own memory commands. A turn waits for the previous turn and its speech.
 */
class DebugAskReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val q = intent.getStringExtra("q")?.trim().orEmpty()
        val sid = intent.getStringExtra("session")
        val mute = intent.getBooleanExtra("mute", false)
        val act = MainActivity.instance
        if (q.isEmpty() || act == null) {
            val rec = JSONObject().put("ts", System.currentTimeMillis()).put("question", q).put("session", sid)
                .put("error", if (q.isEmpty()) "no question (--es q)" else "app not open")
            runCatching { File(context.filesDir, "soak.jsonl").appendText(rec.toString() + "\n") }
            return
        }
        act.debugAsk(q, sid, mute)
    }
}

/**
 * DEBUG BUILDS ONLY. A spoken turn without a person: streams files/<wav> (16 kHz, 16-bit mono PCM) into the
 * on-device recognizer in real time (EXTRA_AUDIO_SOURCE), and "releases" the button at the end of the file.
 *   adb shell am broadcast -n com.leaps.voiceagent/.DebugSpeakReceiver --es wav q1.wav
 */
class DebugSpeakReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val name = intent.getStringExtra("wav") ?: return
        MainActivity.instance?.debugSpeak(File(context.filesDir, name))
    }
}
