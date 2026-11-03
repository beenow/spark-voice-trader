package com.leaps.voiceagent

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Process
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.leapsify.kokoro.KokoroSynth
import com.leapsify.phonemizer.Phonemizer
import com.leapsify.phonemizer.PhonemizerAssets
import java.io.File
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * On-device speech: Kokoro-82M (ONNX Runtime) fed by the Rust misaki phonemizer, both on the phone.
 *
 * text -> sentences -> per sentence: [TtsNormalizer] -> [Phonemizer] -> [KokoroSynth] -> one streaming
 * [AudioTrack]. Sentences are synthesized one after another on [synthThread] while earlier ones play, so
 * playback starts as soon as the FIRST sentence is rendered ([onFirstAudio] reports that time).
 *
 * Android [TextToSpeech] is only the fallback: for the whole reply when Kokoro is not loaded, and for one
 * sentence the lexicon cannot say (unknown word), or whose render fails / is NaN / is silent.
 */
class KokoroSpeaker(private val context: Context, private val systemTts: () -> TextToSpeech?) {
    private val synthThread = Executors.newSingleThreadExecutor { r -> Thread(r, "kokoro-synth") }
    private val generation = AtomicInteger()
    private val queue = LinkedBlockingQueue<Item>()
    @Volatile private var phonemizer: Phonemizer? = null
    @Volatile private var synth: KokoroSynth? = null
    @Volatile private var track: AudioTrack? = null
    @Volatile var speed = 1.05f

    /**
     * Where answers play. null = the media route (A2DP to glasses/headphones when connected, else the speaker).
     * A device = the call path to it (VOICE_COMMUNICATION), for Glasses mode where the SCO/LE link stays up.
     */
    @Volatile var commDevice: android.media.AudioDeviceInfo? = null
        set(v) { if (field?.id != v?.id) { field = v; reopen = true } }
    @Volatile private var reopen = false
    @Volatile var loadError: String? = null; private set

    val ready: Boolean get() = synth != null && phonemizer != null
    val provider: String get() = synth?.executionProvider?.name ?: "-"
    var config = ""; private set

    private sealed class Item(val gen: Int) {
        class Pcm(gen: Int, val samples: FloatArray, val onFirst: (() -> Unit)?) : Item(gen)
        class System(gen: Int, val text: String, val onFirst: (() -> Unit)?) : Item(gen)
        class End(gen: Int, val onDone: (() -> Unit)?) : Item(gen)
    }

    init {
        Thread({
            runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO) }
            writerLoop()
        }, "kokoro-audio").apply { isDaemon = true }.start()
    }

    /** Loads the phonemizer and the ONNX session (seconds). Blocking; returns the load time in ms or -1. */
    fun load(dir: File, threads: Int = 4, xnnpack: Boolean = true, spin: Boolean = true, xnnThreads: Int = threads): Long {
        val t0 = SystemClock.elapsedRealtime()
        return try {
            val p = Phonemizer(PhonemizerAssets.install(context))
            val s = try {
                KokoroSynth(File(dir, "model.onnx"), File(dir, "voice.bin"), File(dir, "tokenizer.json"),
                    preferXnnpack = xnnpack, threads = threads, allowSpinning = spin, xnnpackThreads = xnnThreads)
            } catch (e: Throwable) { p.close(); throw e }
            phonemizer = p; synth = s; loadError = null
            config = "${s.executionProvider} t$threads${if (xnnpack) "/x$xnnThreads" else ""}${if (spin) "" else " nospin"}"
            // Warm-up render: the first session.run pays one-off allocation costs.
            runCatching { s.synthesize(p.phonemize("Ready.").phonemes, speed) }
            runCatching { s.synthesize(p.phonemize("Apple is at about two forty-nine, down one point four percent today.").phonemes, speed) }
            // Synth benchmark line for tuning: one fixed clause, timed after the warm-up.
            runCatching {
                val b0 = SystemClock.elapsedRealtime()
                val pcm = s.synthesize(p.phonemize(TtsNormalizer.normalize("Nvidia is at about one eighty-five,")).phonemes, speed)
                Log.i(TAG, "bench $config: ${SystemClock.elapsedRealtime() - b0} ms for ${pcm.size * 1000L / KokoroSynth.SAMPLE_RATE} ms audio")
            }
            SystemClock.elapsedRealtime() - t0
        } catch (e: Throwable) {
            Log.e(TAG, "kokoro load failed", e)
            loadError = e.message ?: e.javaClass.simpleName
            -1
        }
    }

    /** Stop whatever is playing or queued. */
    fun stop() {
        generation.incrementAndGet()
        val dropped = mutableListOf<Item>()
        queue.drainTo(dropped)
        dropped.filterIsInstance<Item.End>().forEach { it.onDone?.invoke() }
        track?.let { runCatching { it.pause(); it.flush() } }
        frames = 0L
        systemTts()?.stop()
    }

    /**
     * Speak [text]. [flush] drops what is queued first. [onFirstAudio] runs once with the ms from this call
     * to the first sample reaching the speaker and which voice produced it ("kokoro" / "system").
     */
    fun speak(text: String, flush: Boolean, mute: Boolean = false, onFirstAudio: ((Long, String) -> Unit)? = null,
              onSentence: ((String) -> Unit)? = null, onDone: (() -> Unit)? = null, onSynthDone: (() -> Unit)? = null) {
        if (text.isBlank()) return
        if (flush) stop()
        val gen = generation.get()
        val t0 = SystemClock.elapsedRealtime()
        var reported = false
        fun first(voice: String): () -> Unit = {
            if (!reported) { reported = true; onFirstAudio?.invoke(SystemClock.elapsedRealtime() - t0, voice) }
        }
        synthThread.execute {
            for (unit in units(text)) {
                if (generation.get() != gen) break
                val pcm = if (ready) render(unit) else null
                val voice = if (pcm != null) "kokoro" else "system"
                onSentence?.invoke(voice)
                when {
                    mute -> first(voice)()          // synthesis timed, nothing played
                    pcm != null -> queue.put(Item.Pcm(gen, pcm, first(voice)))
                    else -> queue.put(Item.System(gen, unit, first(voice)))
                }
            }
            onSynthDone?.invoke()                // all units rendered: the CPU is free while the audio plays
            if (mute) onDone?.invoke() else queue.put(Item.End(gen, onDone))
        }
    }

    /** One sentence through the chain; null means "say it with the system voice". */
    private fun render(unit: String): FloatArray? {
        val p = phonemizer ?: return null
        val s = synth ?: return null
        return try {
            var spoken = TtsNormalizer.normalize(unit)
            var ph = p.phonemize(spoken)
            if (ph.unknownWords.isNotEmpty()) {            // spell what the lexicon does not know, then retry
                for (w in ph.unknownWords) spoken = spoken.replace(w, TtsNormalizer.spellOut(w) ?: "")
                ph = p.phonemize(spoken)
            }
            if (ph.unknownWords.isNotEmpty()) { Log.i(TAG, "unknown words ${ph.unknownWords}: system voice"); return null }
            if (s.tokenize(ph.phonemes).size !in 1..KokoroSynth.MAX_TOKENS) return null
            val t0 = SystemClock.elapsedRealtime()
            val pcm = s.synthesize(ph.phonemes, speed)
            Log.i(TAG, "synth ${SystemClock.elapsedRealtime() - t0} ms for ${pcm.size * 1000L / KokoroSynth.SAMPLE_RATE} ms audio: $unit")
            var peak = 0f
            for (x in pcm) { if (x.isNaN()) return null; peak = maxOf(peak, kotlin.math.abs(x)) }
            if (peak < 1e-3f) null else pcm
        } catch (e: Throwable) {
            Log.e(TAG, "render failed", e); null
        }
    }

    /** Frames written since the track was opened or last flushed (a flush resets the play head to 0). */
    @Volatile private var frames = 0L

    private fun writerLoop() {
        while (true) {
            val item = queue.take()
            if (item.gen != generation.get()) { if (item is Item.End) item.onDone?.invoke(); continue }
            when (item) {
                is Item.Pcm -> {
                    if (reopen) { reopen = false; track?.let { drain(it, frames, item.gen); it.release() }; track = null }
                    val t = track ?: open().also { track = it; frames = 0L }
                    var off = 0
                    while (off < item.samples.size && item.gen == generation.get()) {
                        val n = runCatching {
                            t.write(item.samples, off, minOf(CHUNK, item.samples.size - off), AudioTrack.WRITE_BLOCKING)
                        }.getOrDefault(-1)
                        if (n <= 0) break
                        if (off == 0) {
                            if (t.playState != AudioTrack.PLAYSTATE_PLAYING) runCatching { t.play() }
                            item.onFirst?.invoke()
                        }
                        off += n; frames += n
                    }
                }
                is Item.System -> {
                    track?.let { drain(it, frames, item.gen) }
                    item.onFirst?.invoke()
                    sayWithSystem(item.text)
                }
                is Item.End -> {
                    track?.let { drain(it, frames, item.gen) }
                    item.onDone?.invoke()
                }
            }
        }
    }

    private fun drain(t: AudioTrack, frames: Long, gen: Int) {
        if (t.playState != AudioTrack.PLAYSTATE_PLAYING) runCatching { t.play() }
        val deadline = SystemClock.elapsedRealtime() + frames * 1000 / KokoroSynth.SAMPLE_RATE + 2000
        while (SystemClock.elapsedRealtime() < deadline) {
            val head = (runCatching { t.playbackHeadPosition }.getOrDefault(0)).toLong() and 0xffffffffL
            if (head >= frames || gen != generation.get()) break      // stop() flushed: nothing left to wait for
            Thread.sleep(15)
        }
    }

    private fun sayWithSystem(text: String) {
        val tts = systemTts() ?: return
        val done = CountDownLatch(1)
        val id = UUID.randomUUID().toString()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(u: String?) {}
            override fun onDone(u: String?) { if (u == id) done.countDown() }
            @Deprecated("Deprecated in Java") override fun onError(u: String?) { if (u == id) done.countDown() }
            override fun onStop(u: String?, interrupted: Boolean) { if (u == id) done.countDown() }
        })
        tts.setAudioAttributes(attributes())
        tts.speak(text, TextToSpeech.QUEUE_ADD, null, id)
        done.await(30, TimeUnit.SECONDS)
    }

    private fun open(): AudioTrack {
        val min = AudioTrack.getMinBufferSize(KokoroSynth.SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val comm = commDevice
        Log.i("LeapsGlasses", "playback route: ${comm?.let { "call path to " + GlassesAudio.name(it) } ?: "media"}")
        return AudioTrack.Builder()
            .setAudioAttributes(attributes())
            .setAudioFormat(AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                .setSampleRate(KokoroSynth.SAMPLE_RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(min, KokoroSynth.SAMPLE_RATE / 2 * 4))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
            .also { t -> comm?.let { t.setPreferredDevice(it) } }
    }

    /** The answer's audio attributes for the current route (also used for the Android TTS fallback). */
    fun attributes(): AudioAttributes = AudioAttributes.Builder()
        .setUsage(if (commDevice != null) AudioAttributes.USAGE_VOICE_COMMUNICATION else AudioAttributes.USAGE_MEDIA)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()

    fun close() {
        stop()
        synthThread.execute {
            synth?.close(); phonemizer?.close(); synth = null; phonemizer = null
            track?.release(); track = null
        }
    }

    companion object {
        private const val TAG = "KokoroSpeaker"
        private const val CHUNK = KokoroSynth.SAMPLE_RATE / 20      // 50 ms per blocking write

        /**
         * Sentences, and a long first sentence cut at its first comma so the first audio comes sooner
         * (Kokoro renders a whole unit before any of it can play).
         */
        fun units(text: String): List<String> {
            val sentences = Regex("""(?<=[.!?])\s+""").split(text.trim()).map { it.trim() }.filter { it.isNotEmpty() }
            if (sentences.isEmpty()) return emptyList()
            val first = sentences[0]
            val comma = first.indexOf(", ", 12)
            return if (first.length > 40 && comma in 12 until first.length - 12)
                listOf(first.substring(0, comma + 1), first.substring(comma + 2)) + sentences.drop(1)
            else sentences
        }
    }
}
