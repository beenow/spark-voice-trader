package com.leaps.voiceagent

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

/**
 * Bluetooth glasses (Meta Ray-Ban or any BT headset) as the agent's microphone and speaker.
 *
 * - **Detection:** a Bluetooth SCO or LE Audio headset in AudioManager's communication devices. [onChange] fires on
 *   connect/disconnect.
 * - **Mic:** the system SpeechRecognizer captures audio itself and does not reliably follow the communication
 *   device, so for the glasses the app captures with AudioRecord (VOICE_COMMUNICATION, preferred device = the
 *   glasses' input) and streams the PCM into the recognizer through EXTRA_AUDIO_SOURCE ([MicStream]). The routed
 *   device is logged, so the log shows which mic was really used.
 * - **Route:** [acquire] selects the glasses as the communication device (MODE_IN_COMMUNICATION, SCO/LE link up)
 *   and reports how long the link took; [release] clears it so playback goes back to A2DP media quality.
 *   With [keepUp] ("Glasses mode"), the link stays up for the whole session: no per-turn switch-on cost, but
 *   answers play over the narrow-band call path.
 *
 * Every decision is logged under the tag `LeapsGlasses`.
 */
class GlassesAudio(context: Context, private val onChange: (AudioDeviceInfo?) -> Unit) {
    private val am = context.getSystemService(AudioManager::class.java)
    private val main = Handler(Looper.getMainLooper())
    @Volatile var keepUp = false
    @Volatile private var held = false

    val device: AudioDeviceInfo?
        get() = am.availableCommunicationDevices.firstOrNull { it.type in BT_TYPES }

    /** The glasses' microphone (the input side of the same headset). */
    val inputDevice: AudioDeviceInfo?
        get() = am.getDevices(AudioManager.GET_DEVICES_INPUTS).firstOrNull { it.type in BT_TYPES }

    val linkUp: Boolean get() = am.communicationDevice?.type in BT_TYPES

    private val callback = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(added: Array<out AudioDeviceInfo>) = changed("added", added)
        override fun onAudioDevicesRemoved(removed: Array<out AudioDeviceInfo>) = changed("removed", removed)
    }

    private val commListener = AudioManager.OnCommunicationDeviceChangedListener { d ->
        log("communication device -> ${d?.let { name(it) } ?: "none"} (mode ${am.mode})")
    }

    init {
        am.registerAudioDeviceCallback(callback, main)
        am.addOnCommunicationDeviceChangedListener(Executors.newSingleThreadExecutor(), commListener)
    }

    private fun changed(what: String, devs: Array<out AudioDeviceInfo>) {
        val bt = devs.filter { it.type in BT_TYPES }
        if (bt.isEmpty()) return
        log("device $what: ${bt.joinToString { name(it) }}")
        if (what == "removed" && held && device == null) { held = false; runCatching { am.mode = AudioManager.MODE_NORMAL } }
        onChange(device)
    }

    /**
     * Bring the glasses link up as the communication device. Blocking (call off the main thread). Returns the
     * ms until AudioManager reports the glasses as the communication device, or -1 when there are no glasses
     * or the switch did not happen within 3 s.
     */
    fun acquire(reason: String): Long {
        val d = device ?: return -1
        if (linkUp) { held = true; return 0 }
        val t0 = SystemClock.elapsedRealtime()
        val done = CountDownLatch(1)
        val l = AudioManager.OnCommunicationDeviceChangedListener { cd -> if (cd?.type in BT_TYPES) done.countDown() }
        val ex = Executors.newSingleThreadExecutor()
        am.addOnCommunicationDeviceChangedListener(ex, l)
        am.mode = AudioManager.MODE_IN_COMMUNICATION
        val ok = am.setCommunicationDevice(d)
        val up = ok && (linkUp || done.await(3, TimeUnit.SECONDS))
        am.removeOnCommunicationDeviceChangedListener(l)
        ex.shutdown()
        val ms = SystemClock.elapsedRealtime() - t0
        held = up
        log("acquire ($reason): ${name(d)} setCommunicationDevice=$ok link ${if (up) "up" else "NOT up"} in $ms ms")
        if (!up) { am.clearCommunicationDevice(); am.mode = AudioManager.MODE_NORMAL }
        return if (up) ms else -1
    }

    /** Drop the link (A2DP media playback comes back), unless Glasses mode keeps it up. */
    fun release(reason: String, force: Boolean = false) {
        if (!held || (keepUp && !force)) return
        held = false
        am.clearCommunicationDevice()
        am.mode = AudioManager.MODE_NORMAL
        log("release ($reason): link down, media route back to ${mediaRoute()}")
    }

    /** Where media (A2DP) would play now, for the log. */
    fun mediaRoute(): String =
        am.getDevices(AudioManager.GET_DEVICES_OUTPUTS).firstOrNull {
            it.type == AudioDeviceInfo.TYPE_BLUETOOTH_A2DP || it.type == AudioDeviceInfo.TYPE_BLE_HEADSET
        }?.let { name(it) } ?: "phone speaker"

    fun close() {
        release("close", force = true)
        am.unregisterAudioDeviceCallback(callback)
        am.removeOnCommunicationDeviceChangedListener(commListener)
    }

    /**
     * Microphone capture from [input] (the glasses, or the phone mic when null) streamed into [sink] as 16 kHz
     * 16-bit mono PCM. Ends on [stop], or by itself after [silenceMs] of quiet that follows speech (hands-free).
     *
     * The capture keeps running across recognizer sessions: when the recognizer ends a segment early, [swap] hands
     * the stream to the next session's pipe and replays the last [REPLAY_MS] first, so words said while the next
     * session starts are not lost (with the recognizer's own mic, "What is Oracle today" came out "What is Call today").
     */
    class MicStream(private val input: AudioDeviceInfo?, sink: ParcelFileDescriptor,
                    private val silenceMs: Long = 900, private val onAutoEnd: () -> Unit) {
        @Volatile private var running = true
        private var thread: Thread? = null
        @Volatile var routed: String = "?"; private set
        private val lock = Any()
        private var out: java.io.OutputStream? = ParcelFileDescriptor.AutoCloseOutputStream(sink)
        private val recent = ArrayDeque<ByteArray>()      // the last REPLAY_MS, for the next session

        /** The next recognizer session's pipe: it gets the last REPLAY_MS, then the live audio. */
        fun swap(sink: ParcelFileDescriptor) = synchronized(lock) {
            out?.let { runCatching { it.close() } }
            val o = ParcelFileDescriptor.AutoCloseOutputStream(sink)
            out = runCatching { recent.forEach { o.write(it) }; o }.getOrElse { runCatching { o.close() }; null }
            log("capture: next recognizer session (replayed ${recent.size * 20} ms)")
        }

        private fun push(b: ByteArray, n: Int) = synchronized(lock) {
            recent.addLast(b.copyOf(n)); while (recent.size > REPLAY_MS / 20) recent.removeFirst()
            val o = out ?: return@synchronized
            // The session ended and closed its end (EPIPE): keep capturing; swap() starts the next one.
            runCatching { o.write(b, 0, n) }.onFailure { runCatching { o.close() }; out = null }
        }

        @SuppressLint("MissingPermission")    // RECORD_AUDIO is checked before a capture starts
        fun start() {
            val min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
            val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_COMMUNICATION, RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(min, RATE / 5 * 2))
            if (input != null) rec.setPreferredDevice(input)
            rec.startRecording()
            routed = rec.routedDevice?.let { name(it) } ?: "default mic"
            log("capture: preferred ${input?.let { name(it) } ?: "phone mic"}, routed to $routed")
            thread = Thread({
                val buf = ShortArray(RATE / 50)                 // 20 ms
                val bytes = ByteArray(buf.size * 2)
                var heard = false
                var quietSince = 0L
                var peak = 0.0
                try {
                    while (running) {
                        val n = rec.read(buf, 0, buf.size)
                        if (n <= 0) continue
                        var sum = 0.0
                        for (i in 0 until n) {
                            val v = buf[i].toInt(); sum += v * v
                            bytes[2 * i] = (v and 0xff).toByte(); bytes[2 * i + 1] = (v shr 8 and 0xff).toByte()
                        }
                        push(bytes, n * 2)
                        val rms = sqrt(sum / n)
                        peak = maxOf(peak, rms)
                        val now = SystemClock.elapsedRealtime()
                        if (rms > SPEECH_RMS) { heard = true; quietSince = 0 }
                        else if (heard) {
                            if (quietSince == 0L) quietSince = now
                            else if (now - quietSince > silenceMs) { log("capture: end of speech (silence)"); running = false; onAutoEnd() }
                        }
                    }
                } catch (e: Exception) {
                    log("capture stopped: ${e.message}")
                } finally {
                    runCatching { rec.stop() }; rec.release(); synchronized(lock) { out?.let { runCatching { it.close() } }; out = null }
                    log("capture closed (peak rms ${peak.toInt()}, speech ${if (heard) "heard" else "NOT heard"})")
                }
            }, "glasses-mic").apply { start() }
        }

        fun stop() { running = false }

        companion object {
            const val RATE = 16_000
            const val SPEECH_RMS = 700.0
            const val REPLAY_MS = 700
        }
    }

    companion object {
        val BT_TYPES = setOf(AudioDeviceInfo.TYPE_BLUETOOTH_SCO, AudioDeviceInfo.TYPE_BLE_HEADSET)
        fun name(d: AudioDeviceInfo): String = "${d.productName} (${typeName(d.type)})"
        private fun typeName(t: Int) = when (t) {
            AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> "BT SCO"
            AudioDeviceInfo.TYPE_BLE_HEADSET -> "BLE headset"
            AudioDeviceInfo.TYPE_BLUETOOTH_A2DP -> "A2DP"
            AudioDeviceInfo.TYPE_BUILTIN_MIC -> "phone mic"
            AudioDeviceInfo.TYPE_BUILTIN_SPEAKER -> "phone speaker"
            else -> "type $t"
        }
        fun log(s: String) { Log.i("LeapsGlasses", s) }
    }
}
