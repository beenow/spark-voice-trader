package com.leaps.voiceagent

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Executors

/**
 * Leaps on-device voice agent (demo app).
 *
 * ON THE PHONE: speech-to-text (Android SpeechRecognizer, on-device recognizer when available), the Llama 3.2 1B
 * model (llama.cpp, CPU, every token generated here) and text-to-speech (Kokoro-82M on ONNX Runtime + the Rust
 * misaki phonemizer; Android TextToSpeech only as the fallback).
 * ON THE MAC (step server, voice/step_server.py): prompt rendering, output parsing, validators/repairs, tool
 * execution on the mock market, slot fill and conversation memory. See android/README.md.
 */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val prefs by lazy { getSharedPreferences("leaps", MODE_PRIVATE) }
    private val llama = LlamaBridge()
    private val store by lazy { ModelStore(this) }

    // palette
    private val bg = Color.rgb(11, 13, 17)
    private val card = Color.rgb(22, 26, 33)
    private val line = Color.rgb(40, 46, 56)
    private val fg = Color.rgb(236, 239, 244)
    private val muted = Color.rgb(140, 150, 165)
    private val phone = Color.rgb(64, 214, 138)     // ran on the phone
    private val mac = Color.rgb(110, 160, 255)      // ran on the Mac
    private val warn = Color.rgb(255, 184, 77)

    private lateinit var status: TextView
    private lateinit var setupPanel: LinearLayout
    private lateinit var urlField: EditText
    private lateinit var passField: EditText
    private lateinit var threadsBtn: Button
    private lateinit var askField: EditText
    private lateinit var talk: TextView
    private lateinit var feed: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var tokS: TextView
    private lateinit var devMs: TextView
    private lateinit var macMs: TextView
    private lateinit var ttfa: TextView
    private lateinit var voiceInfo: TextView
    private val speaker by lazy { KokoroSpeaker(this) { tts } }
    @Volatile private var voiceDownloading = false
    private val chips = mutableMapOf<String, TextView>()

    private var tts: TextToSpeech? = null
    private var recognizer: SpeechRecognizer? = null
    private var useOnDeviceStt = true
    private var sttOnDevice: Boolean? = null
    private var listening = false
    @Volatile private var modelLoaded = false
    @Volatile private var busy = false
    @Volatile private var cancelDownload = false
    @Volatile private var session = UUID.randomUUID().toString()
    private var threads = 4
    @Volatile private var restoredPrefix = IntArray(0)
    @Volatile private var prefixFile: File? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Tuning hooks for adb (`am start --ei threads 6 --ei tts_threads 4 --ez tts_xnn true --ez tts_spin false`).
        intent?.extras?.let { x ->
            val e = prefs.edit()
            if (x.containsKey("threads")) e.putInt("threads", x.getInt("threads"))
            if (x.containsKey("tts_threads")) e.putInt("tts_threads", x.getInt("tts_threads"))
            if (x.containsKey("tts_xnn")) e.putBoolean("tts_xnn", x.getBoolean("tts_xnn"))
            if (x.containsKey("tts_spin")) e.putBoolean("tts_spin", x.getBoolean("tts_spin"))
            if (x.containsKey("tts_xnn_threads")) e.putInt("tts_xnn_threads", x.getInt("tts_xnn_threads"))
            if (x.containsKey("model")) e.putString("model", x.getString("model"))   // part of a GGUF name
            if (x.containsKey("stt_partial_ms")) e.putInt("stt_partial_ms", x.getInt("stt_partial_ms"))
            e.apply()
        }
        threads = prefs.getInt("threads", 6)   // Fold bench: 6 > 4 for prefill; 8 collapses under load
        instance = this
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)   // demo/soak: never doze mid-turn
        window.statusBarColor = bg
        window.navigationBarColor = bg
        buildUi()
        tts = TextToSpeech(this) { st -> if (st == TextToSpeech.SUCCESS) tts?.language = Locale.US }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
        speaker.speed = prefs.getFloat("tts_speed", 1.05f)
        if (store.kokoroInstalled()) loadVoice() else setVoiceInfo(null)
        if (store.installed(modelPref()) != null) loadModel()
        else if (passField.text.isNotEmpty()) setupModel()          // passcode already entered once: fetch it
        else { showSetup(true); setStatus("Setup: server + passcode, then Get model") }
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        cancelDownload = true
        recognizer?.destroy()
        tts?.shutdown()
        speaker.close()
        worker.execute { llama.close() }
        super.onDestroy()
    }

    // ================================================================ UI
    private fun dp(v: Int) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()

    private fun rounded(color: Int, radius: Int = 14, stroke: Int? = null) = GradientDrawable().apply {
        setColor(color); cornerRadius = dp(radius).toFloat()
        if (stroke != null) setStroke(dp(1), stroke)
    }

    private fun label(s: String, size: Float, color: Int = fg, bold: Boolean = false) = TextView(this).apply {
        this.text = s; textSize = size; setTextColor(color)
        if (bold) typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private fun button(s: String, onClick: () -> Unit) = Button(this).apply {
        text = s; isAllCaps = false; textSize = 14f; setTextColor(fg)
        background = rounded(card, 10, line); setOnClickListener { onClick() }
    }

    private fun field(hint: String, value: String, password: Boolean = false) = EditText(this).apply {
        this.hint = hint; setText(value); textSize = 15f; setTextColor(fg); setHintTextColor(muted)
        isSingleLine = true; background = rounded(card, 10, line); setPadding(dp(12), dp(10), dp(12), dp(10))
        if (password) inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
    }

    private fun hrow(vararg views: View, gap: Int = 8) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        views.forEachIndexed { i, v ->
            addView(v, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply { if (i > 0) leftMargin = dp(gap) })
        }
    }

    private fun metric(caption: String): Pair<LinearLayout, TextView> {
        val value = label("–", 26f, fg, true)
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; gravity = Gravity.CENTER
            background = rounded(card, 14, line); setPadding(dp(8), dp(10), dp(8), dp(10))
            addView(value.apply { gravity = Gravity.CENTER })
            addView(label(caption, 12f, muted).apply { gravity = Gravity.CENTER })
        }
        return box to value
    }

    private fun chip(key: String, title: String): TextView = label(title, 12.5f, muted, true).apply {
        gravity = Gravity.CENTER; setPadding(dp(6), dp(8), dp(6), dp(8)); background = rounded(card, 10, line)
        chips[key] = this
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(36), dp(16), dp(16))
            setBackgroundColor(bg)
        }
        // header: name + the badge
        val badge = label("Llama 3.2 1B · on device", 13.5f, Color.rgb(8, 30, 18), true).apply {
            background = rounded(phone, 20); setPadding(dp(12), dp(5), dp(12), dp(5))
        }
        val title = label("Leaps", 26f, fg, true)
        val gear = label("Setup", 14f, muted).apply { setPadding(dp(12), dp(6), dp(4), dp(6)); setOnClickListener { showSetup(setupPanel.visibility != View.VISIBLE) } }
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            addView(title)
            addView(View(context), LinearLayout.LayoutParams(dp(12), 1))
            addView(badge)
            addView(View(context), LinearLayout.LayoutParams(0, 1, 1f))
            addView(gear)
        }
        status = label("starting…", 13f, muted).apply { setPadding(0, dp(6), 0, 0) }
        voiceInfo = label("voice: …", 13f, muted).apply { setPadding(0, dp(2), 0, dp(10)) }

        // setup (hidden once the model is loaded)
        urlField = field("step server, e.g. http://192.168.1.2:8310", prefs.getString("url", null) ?: BuildConfig.DEFAULT_SERVER)
        passField = field("passcode", prefs.getString("pass", "") ?: "", password = true)
        threadsBtn = button("threads $threads") { cycleThreads() }
        setupPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL; setPadding(0, 0, 0, dp(10))
            addView(urlField); addView(View(context), LinearLayout.LayoutParams(1, dp(8)))
            addView(passField); addView(View(context), LinearLayout.LayoutParams(1, dp(8)))
            addView(hrow(button("Get model") { setupModel() }, threadsBtn, button("Warm") { warm() },
                button("New chat") { newSession() }))
        }

        // where each step runs (lit per turn from local measurements; no network needed to draw it)
        val where = hrow(chip("stt", "Speech→text\n·"), chip("model", "Llama 1B\n·"),
            chip("tools", "Tools + checks\n·"), chip("tts", "Voice (Kokoro)\n·"), gap = 6)
        val legend = label("green = ran on this phone   blue = ran on the Mac", 12f, muted).apply { setPadding(0, dp(6), 0, dp(10)) }

        val (m1, v1) = metric("tokens / s (phone)")
        val (m2, v2) = metric("model ms on phone")
        val (m3, v3) = metric("Mac + network ms")
        val (m4, v4) = metric("first audio ms")
        tokS = v1; devMs = v2; macMs = v3; ttfa = v4
        val metrics = hrow(m1, m2, m3, m4, gap = 6)

        feed = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(4), 0, dp(8)) }
        scroll = ScrollView(this).apply { addView(feed); isVerticalScrollBarEnabled = false }

        askField = field("type a question", "")
        val ask = button("Ask") {
            val q = askField.text.toString().trim()
            if (q.isNotEmpty()) { askField.setText(""); runTurn(q) }
        }
        talk = label("Hold to talk", 22f, Color.rgb(8, 30, 18), true).apply {
            gravity = Gravity.CENTER; background = rounded(phone, 40)
            setOnTouchListener { _, ev ->
                when (ev.action) {
                    MotionEvent.ACTION_DOWN -> startListening()
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> stopListening()
                }
                true
            }
        }
        val askRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(8), 0, dp(10))
            addView(askField, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            addView(ask, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { leftMargin = dp(8) })
        }

        root.addView(header)
        root.addView(status)
        root.addView(voiceInfo)
        root.addView(setupPanel)
        root.addView(where)
        root.addView(legend)
        root.addView(metrics)
        root.addView(scroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f).apply { topMargin = dp(10) })
        root.addView(askRow)
        root.addView(talk, LinearLayout.LayoutParams(MATCH_PARENT, dp(84)))
        setContentView(root)
        showSetup(false)
    }

    private fun showSetup(show: Boolean) { setupPanel.visibility = if (show) View.VISIBLE else View.GONE }

    private fun ui(f: () -> Unit) { main.post(f) }
    private fun setStatus(s: String) = ui { status.text = s }

    /** Light a pipeline chip: where it ran (phone/mac) and a short detail (ms, offline…). */
    private fun light(key: String, onPhone: Boolean?, detail: String) = ui {
        val c = chips[key] ?: return@ui
        val name = c.text.toString().substringBefore('\n')
        val color = when (onPhone) { true -> phone; false -> mac; null -> warn }
        c.text = "$name\n$detail"
        c.setTextColor(color)
        c.background = rounded(card, 10, color)
    }

    private fun resetChips() = ui {
        chips.values.forEach { c ->
            c.text = c.text.toString().substringBefore('\n') + "\n·"; c.setTextColor(muted); c.background = rounded(card, 10, line)
        }
    }

    private fun bubble(s: String, size: Float, color: Int, fill: Int?, mono: Boolean = false) = ui {
        val v = label(s, size, color).apply {
            if (mono) typeface = Typeface.MONOSPACE
            setTextIsSelectable(true)
            if (fill != null) { background = rounded(fill, 14); setPadding(dp(14), dp(10), dp(14), dp(10)) }
            else setPadding(dp(4), dp(2), dp(4), dp(2))
        }
        feed.addView(v, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(6) })
        while (feed.childCount > 120) feed.removeViewAt(0)
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun note(s: String) = bubble(s, 13f, muted, null, mono = true)

    private fun saveSettings() {
        prefs.edit().putString("url", urlField.text.toString().trim()).putString("pass", passField.text.toString()).apply()
    }

    private fun client() = StepClient(urlField.text.toString().trim(), passField.text.toString())

    private fun cycleThreads() {
        val opts = listOf(2, 4, 6, 8)
        threads = opts[(opts.indexOf(threads) + 1) % opts.size]
        prefs.edit().putInt("threads", threads).apply()
        threadsBtn.text = "threads $threads"
        if (modelLoaded) worker.execute { llama.setThreads(threads) }
    }

    // ================================================================ model
    /** The GGUF comes from the step server (passcode), resumable, verified by sha256, then loaded. */
    private fun setupModel() {
        saveSettings()
        if (store.installed(modelPref()) != null) { downloadVoice(); loadModel(); return }
        if (passField.text.isEmpty()) { setStatus("Enter the passcode, then Get model"); return }
        cancelDownload = false
        worker.execute {
            val t0 = SystemClock.elapsedRealtime()
            try {
                var last = 0L
                val file = store.download(urlField.text.toString().trim(), passField.text.toString(), { done, total ->
                    val now = SystemClock.elapsedRealtime()
                    if (now - last > 250 || done == total) {
                        last = now
                        setStatus(if (done == total) "Verifying sha256…"
                        else "Downloading model ${done / 1_000_000} / ${total / 1_000_000} MB (${done * 100 / maxOf(total, 1)}%)")
                    }
                }, { cancelDownload }, prefer = modelPref())
                note("model ${file.name} downloaded + sha256 verified in ${(SystemClock.elapsedRealtime() - t0) / 1000} s")
                loadNow()
            } catch (e: Exception) {
                setStatus("Model download failed: ${e.message}")
            }
        }
    }

    private fun loadModel() = worker.execute { loadNow() }

    /**
     * The server's default GGUF is the first "llm" entry of its manifest (STEP_MODEL_FILES on the Mac). When
     * it names a different file than the one installed (and no model was pinned with --es model), fetch it
     * (resumable, sha256-verified) and swap it in. Runs on the worker thread.
     */
    private fun checkServerModel() {
        val m = runCatching { store.manifest(urlField.text.toString().trim(), passField.text.toString()) }.getOrNull() ?: return
        val files = m.getJSONArray("files")
        val want = (0 until files.length()).map { files.getJSONObject(it) }
            .firstOrNull { it.optString("kind", "llm") == "llm" && it.getString("name").endsWith(".gguf") }?.getString("name") ?: return
        val have = store.installed()?.name
        if (have == want) return
        note("server default model is $want (installed: $have): downloading it")
        try {
            var last = 0L
            val f = store.download(urlField.text.toString().trim(), passField.text.toString(), { done, total ->
                val now = SystemClock.elapsedRealtime()
                if (now - last > 500 || done == total) { last = now; setStatus("Downloading $want ${done / 1_000_000} / ${total / 1_000_000} MB") }
            }, { cancelDownload }, manifest = m, prefer = want)
            llama.close(); modelLoaded = false; restoredPrefix = IntArray(0)
            note("switched to ${f.name}")
            loadNow()
        } catch (e: Exception) {
            note("model switch failed: ${e.message}")
        }
    }

    /** Which GGUF to use when the server offers several (part of its name); "" = the server's first. */
    private fun modelPref(): String = prefs.getString("model", "") ?: ""

    // ================================================================ voice (Kokoro on the phone)
    private fun setVoiceInfo(kokoro: String?) = ui {
        if (kokoro != null) { voiceInfo.text = "voice: Kokoro-82M af_heart · on device · $kokoro"; voiceInfo.setTextColor(phone) }
        else { voiceInfo.text = "voice: Android TTS (fallback)" + (speaker.loadError?.let { " · Kokoro failed: $it" } ?: ""); voiceInfo.setTextColor(warn) }
    }

    private fun loadVoice() = Thread {
        setVoiceInfoLoading()
        val ms = speaker.load(store.kokoroDir(), prefs.getInt("tts_threads", 8), prefs.getBoolean("tts_xnn", false),
            prefs.getBoolean("tts_spin", true), prefs.getInt("tts_xnn_threads", prefs.getInt("tts_threads", 8)))
        if (ms >= 0) { setVoiceInfo("${speaker.config} · ${speaker.speed}x · loaded in $ms ms"); note("Kokoro voice loaded on the phone in $ms ms (${speaker.provider})") }
        else setVoiceInfo(null)
    }.start()

    private fun setVoiceInfoLoading() = ui { voiceInfo.text = "voice: loading Kokoro…"; voiceInfo.setTextColor(muted) }

    /** The Kokoro bundle from the step server (same manifest, passcode, resume and sha256 as the GGUF). */
    private fun downloadVoice() {
        if (store.kokoroInstalled() || voiceDownloading || passField.text.isEmpty()) return
        voiceDownloading = true
        val url = urlField.text.toString().trim(); val pass = passField.text.toString()
        Thread {
            try {
                var last = 0L
                val cfg = store.downloadKokoro(url, pass, { done, total ->
                    val now = SystemClock.elapsedRealtime()
                    if (now - last > 300 || done == total) { last = now; ui { voiceInfo.text = "voice: downloading Kokoro ${done / 1_000_000} / ${total / 1_000_000} MB" } }
                }, { cancelDownload })
                if (cfg == null) { note("server has no Kokoro bundle: Android TTS"); setVoiceInfo(null); return@Thread }
                speaker.speed = cfg.optDouble("speed", 1.05).toFloat()
                prefs.edit().putFloat("tts_speed", speaker.speed).apply()
                note("Kokoro voice downloaded + sha256 verified")
                loadVoice()
            } catch (e: Exception) {
                note("voice download failed: ${e.message}"); setVoiceInfo(null)
            } finally { voiceDownloading = false }
        }.start()
    }

    private fun loadNow() {
        if (llama.isLoaded) return
        if (!LlamaBridge.available) { setStatus("Native library missing from this build"); return }
        val f = store.installed(modelPref()) ?: run { setStatus("No model yet: Setup → Get model"); ui { showSetup(true) }; return }
        setStatus("Loading ${f.name}…")
        val t0 = SystemClock.elapsedRealtime()
        if (!llama.load(f.absolutePath, null, 4096, threads)) { setStatus("Model load failed"); return }
        prefixFile = File(f.parentFile, "${f.name}.prefix.kv")
        val r0 = SystemClock.elapsedRealtime()
        restoredPrefix = if (prefixFile!!.isFile) llama.loadState(prefixFile!!.absolutePath) else IntArray(0)
        if (restoredPrefix.isNotEmpty())
            note("prefix KV (${restoredPrefix.size} tokens) restored from disk in ${SystemClock.elapsedRealtime() - r0} ms")
        modelLoaded = true
        setStatus("${f.name} · CPU · $threads threads · loaded in ${SystemClock.elapsedRealtime() - t0} ms")
        ui { showSetup(false) }
        if (passField.text.isNotEmpty()) { ui { downloadVoice() }; warmNow() }
    }

    private fun warm() = worker.execute { warmNow() }

    /** Prefill the fixed system+tools prefix once; turns then only process their new tokens. */
    private fun warmNow() {
        if (!modelLoaded) return
        try {
            if (modelPref().isEmpty()) checkServerModel()
            val toks = client().prefix().getJSONArray("tokens").toIntArray()
            if (toks.contentEquals(restoredPrefix)) {
                // Touch every weight page once (mmap is lazy): one throwaway token after the prefix.
                val w0 = SystemClock.elapsedRealtime()
                llama.generate(toks + intArrayOf(toks.last()), 1, 1)
                note("warm: prefix KV already loaded; weights paged in in ${SystemClock.elapsedRealtime() - w0} ms")
                return
            }
            val ms = llama.prefill(toks)
            restoredPrefix = toks
            val saved = prefixFile?.let { llama.saveState(it.absolutePath) } ?: false
            note("warm: ${toks.size}-token system+tools prefix prefilled on the phone in ${"%.0f".format(ms)} ms" +
                if (saved) " (KV saved for next launch)" else "")
        } catch (e: Exception) {
            note("warm skipped: ${e.message}")
        }
    }

    private fun newSession() = worker.execute {
        try { client().step(JSONObject().put("session", session).put("reset", true)) } catch (_: Exception) {}
        session = UUID.randomUUID().toString()
        ui { feed.removeAllViews() }
        resetChips()
    }

    // ================================================================ speech in
    private fun recognizerIntent() = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        // Push-to-talk: release ends the utterance; keep the recognizer's own end-pointing short too.
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 300L)
        putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 300L)
    }

    private fun ensureRecognizer(): SpeechRecognizer? {
        recognizer?.let { return it }
        val r = if (useOnDeviceStt && SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) {
            sttOnDevice = true
            SpeechRecognizer.createOnDeviceSpeechRecognizer(this)
        } else if (SpeechRecognizer.isRecognitionAvailable(this)) {
            sttOnDevice = false          // system recognizer asked to prefer offline; may use the network
            SpeechRecognizer.createSpeechRecognizer(this)
        } else {
            note("no speech recognizer on this device: type instead"); return null
        }
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(p: Bundle?) { talk.text = "Listening…" }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onEvent(t: Int, p: Bundle?) {}
            override fun onPartialResults(p: Bundle?) {
                p?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.takeIf { it.isNotBlank() }?.let {
                    talk.text = it; lastPartial = it
                }
            }
            override fun onResults(p: Bundle?) {
                val all = p?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION).orEmpty()
                android.util.Log.i("LeapsStt", "final $all (last partial: $lastPartial)")
                // Some recognizers return an empty top hypothesis for the final; fall back to the best non-empty.
                val q = all.firstOrNull { it.isNotBlank() }?.trim() ?: lastPartial
                finishListening(q, "final")
            }
            override fun onError(code: Int) {
                if (listenDone) return            // a partial already started the turn
                listenDone = true
                listening = false
                talk.text = "Hold to talk"
                if (useOnDeviceStt && (code == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED ||
                            code == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE ||
                            code == SpeechRecognizer.ERROR_SERVER_DISCONNECTED)) {
                    note("on-device recognizer unavailable (error $code); using the system recognizer")
                    useOnDeviceStt = false
                    recognizer?.destroy(); recognizer = null
                } else note("speech error $code")
            }
        })
        recognizer = r
        return r
    }

    private var listenStart = 0L

    private fun startListening() {
        if (busy) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1); return
        }
        speaker.stop()
        val r = ensureRecognizer() ?: return
        listening = true
        listenDone = false
        lastPartial = null
        pressId++
        listenStart = SystemClock.elapsedRealtime()
        r.startListening(recognizerIntent())
    }

    private var releasedAt = 0L
    @Volatile private var sttFinalMs = -1L
    @Volatile private var sttFinalLast = -1L

    @Volatile private var lastPartial: String? = null
    @Volatile private var pressId = 0
    @Volatile private var listenDone = true
    @Volatile private var turnReleaseAt = 0L

    private fun stopListening() {
        if (!listening) return
        releasedAt = SystemClock.elapsedRealtime()
        recognizer?.stopListening()
        // If the final result is slow, go with the last partial (0 = always wait for the final).
        val wait = prefs.getInt("stt_partial_ms", 600).toLong()
        val press = pressId                 // a later press must not be finished by this press's timer
        if (wait > 0) main.postDelayed({
            if (press == pressId && !listenDone && !lastPartial.isNullOrBlank()) finishListening(lastPartial, "partial")
        }, wait)
    }

    /** One result per press: the final result, or the last partial when the final is late (see stopListening). */
    private fun finishListening(text: String?, kind: String) {
        if (listenDone) return
        listenDone = true
        sttFinalMs = if (releasedAt > 0) SystemClock.elapsedRealtime() - releasedAt else -1
        sttKind = kind
        turnReleaseAt = releasedAt
        releasedAt = 0
        listening = false
        talk.text = "Hold to talk"
        val q = text?.trim()
        if (!q.isNullOrEmpty()) runTurn(q, fromVoice = true) else note("heard nothing")
    }
    @Volatile private var sttKind = "final"

    // ================================================================ speech out
    /** Kokoro on the phone, sentence by sentence; Android TTS only where Kokoro cannot (see KokoroSpeaker). */
    private fun speak(s: String, flush: Boolean, answer: Boolean = false, mute: Boolean = false,
                      onFinished: ((ttfaMs: Long, voices: Set<String>) -> Unit)? = null) {
        if (s.isBlank()) { onFinished?.invoke(-1, emptySet()); return }
        speakingCount.incrementAndGet()
        val used = mutableSetOf<String>()
        var first = -1L
        speaker.speak(s, flush, mute = mute,
            onFirstAudio = { ms, voice -> first = ms
                if (answer) ui { ttfa.text = "$ms" }
                if (answer && turnReleaseAt > 0) {
                    val e2e = SystemClock.elapsedRealtime() - turnReleaseAt
                    turnReleaseAt = 0
                    android.util.Log.i("LeapsTiming", "release->first audio ${e2e}ms (stt ${sttFinalLast}ms ${sttKind}, tts first ${ms}ms, ${tempC()})")
                    note("release → first audio ${e2e} ms")
                }
                if (answer) note("first audio ${ms} ms after the answer arrived (${if (voice == "kokoro") "Kokoro on phone" else "Android TTS"})" +
                    if (mute) " [muted: synthesis only]" else "") },
            onSentence = { v -> synchronized(used) { used += v } },
            onDone = {
                val v = synchronized(used) { used.toSet() }
                if (answer) {
                    val k = "kokoro" in v; val sys = "system" in v
                    light("tts", true, when { k && sys -> "Kokoro + Android" ; k -> "Kokoro · phone"; else -> "Android TTS" })
                }
                speakingCount.decrementAndGet()
                onFinished?.invoke(first, v)
            })
    }

    // ================================================================ debug soak hook (DebugAskReceiver, debug builds)
    /** One typed question through the spoken pipeline minus STT; one JSON line per turn in files/soak.jsonl. */
    fun debugAsk(question: String, sid: String?, mute: Boolean) {
        val rec = JSONObject().put("ts", System.currentTimeMillis()).put("question", question)
            .put("session", sid ?: session).put("mute", mute).put("threads", threads)
        if (!modelLoaded) { rec.put("error", "model not loaded"); writeSoak(rec); return }
        debugQueue.execute {
            // FIFO, one at a time: wait for the previous turn AND its speech, so turns never overlap.
            val deadline = SystemClock.elapsedRealtime() + 90_000
            while ((busy || speaking > 0) && SystemClock.elapsedRealtime() < deadline) Thread.sleep(50)
            if (busy || speaking > 0) { rec.put("error", "busy"); writeSoak(rec); return@execute }
            val started = java.util.concurrent.CountDownLatch(1)
            ui { runTurn(question, false, sid, mute, rec); started.countDown() }
            started.await(5, java.util.concurrent.TimeUnit.SECONDS)      // busy is set before the next one looks
        }
    }

    /** Battery + hottest CPU temperature, logged with every measurement (heat inflates everything). */
    private fun tempC(): String {
        val batt = registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val b = (batt?.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10.0
        val lvl = batt?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val cpu = runCatching {
            File("/sys/class/thermal").listFiles()?.filter { z -> runCatching { File(z, "type").readText().startsWith("cpu-") &&
                !File(z, "type").readText().contains("trip") }.getOrDefault(false) }
                ?.maxOf { File(it, "temp").readText().trim().toInt() / 1000.0 }
        }.getOrNull()
        return "batt ${lvl}% ${b}C" + (cpu?.let { " cpu ${"%.0f".format(it)}C" } ?: "")
    }

    /**
     * DEBUG: a spoken turn without a person. The WAV's PCM (16-bit mono) is streamed into the recognizer in real
     * time through EXTRA_AUDIO_SOURCE (no speaker/mic loop); the end of the file is the button release.
     */
    fun debugSpeak(wav: File) = ui {
        if (busy) return@ui
        val r = ensureRecognizer() ?: return@ui
        val bytes = wav.readBytes()
        val rate = java.nio.ByteBuffer.wrap(bytes, 24, 4).order(java.nio.ByteOrder.LITTLE_ENDIAN).int
        val pipe = android.os.ParcelFileDescriptor.createPipe()
        speaker.stop()
        listening = true; listenDone = false; lastPartial = null; pressId++
        listenStart = SystemClock.elapsedRealtime()
        r.startListening(recognizerIntent().apply {
            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipe[0])
            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING, android.media.AudioFormat.ENCODING_PCM_16BIT)
            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, rate)
        })
        Thread {
            android.os.ParcelFileDescriptor.AutoCloseOutputStream(pipe[1]).use { out ->
                val chunk = rate / 50 * 2                      // 20 ms
                var off = 44
                while (off < bytes.size) {
                    val n = minOf(chunk, bytes.size - off)
                    runCatching { out.write(bytes, off, n) }.onFailure { return@Thread }
                    off += n
                    Thread.sleep(20)
                }
                ui { stopListening() }                         // end of speech = release
            }
        }.start()
    }

    private fun writeSoak(rec: JSONObject) {
        synchronized(soakLock) { runCatching { File(filesDir, "soak.jsonl").appendText(rec.toString() + "\n") } }
        android.util.Log.i("LeapsSoak", rec.toString())
    }
    private val soakLock = Any()
    private val debugQueue = Executors.newSingleThreadExecutor()
    private val speakingCount = java.util.concurrent.atomic.AtomicInteger()
    private val speaking: Int get() = speakingCount.get()

    // ================================================================ the turn
    private fun runTurn(question: String, fromVoice: Boolean = false, sid: String? = null, mute: Boolean = false,
                        rec: JSONObject? = null) {
        if (!modelLoaded) { note("model not loaded yet"); return }
        if (busy) { rec?.let { it.put("error", "busy"); writeSoak(it) }; return }
        busy = true
        ui { talk.alpha = 0.5f }
        resetChips()
        if (fromVoice) light("stt", sttOnDevice, if (sttOnDevice == true) "phone · offline" else "system recognizer")
        else light("stt", true, "typed")
        worker.execute {
            try { turn(question, sid ?: session, mute, rec) } catch (e: Exception) {
                note("error: ${e.message}")
                light("tools", null, "Mac unreachable")
                speak("Sorry, I couldn't reach the Mac.", true, mute = mute)
                rec?.let { it.put("error", "${e.javaClass.simpleName}: ${e.message}"); writeSoak(it) }
            } finally {
                busy = false
                ui { talk.alpha = 1f }
            }
        }
    }

    private fun turn(question: String, sid: String, mute: Boolean = false, rec: JSONObject? = null) {
        val steps = JSONArray()
        bubble(question, 20f, fg, Color.rgb(30, 36, 46))
        val c = client()
        val t0 = SystemClock.elapsedRealtime()
        var modelMs = 0.0
        var genTok = 0
        var genMs = 0.0
        var step = 0
        val http = mutableListOf<Long>()
        val parts = mutableListOf<String>()
        fun timed(body: JSONObject): JSONObject {
            val h0 = SystemClock.elapsedRealtime(); val res = c.step(body); http += SystemClock.elapsedRealtime() - h0; return res
        }
        var r = timed(JSONObject().put("session", sid).put("question", question))
        while (r.has("prompt")) {
            showEvents(r.optJSONArray("events"))
            step++
            light("model", true, "step $step…")
            val g = llama.generate(r.getJSONArray("tokens").toIntArray(), r.optInt("max_tokens", 160),
                r.optInt("answer_max_tokens", 70))
            if (g.has("error")) {
                note("model error: ${g.getString("error")}")
                c.step(JSONObject().put("session", sid).put("reset", true))
                if (sid == session) session = UUID.randomUUID().toString()
                speak("Sorry, something went wrong on the phone.", true, mute = mute)
                rec?.let { it.put("error", "model: ${g.getString("error")}").put("steps", steps); writeSoak(it) }
                return
            }
            modelMs += g.getDouble("total_ms"); genTok += g.getInt("gen_tokens"); genMs += g.getDouble("gen_ms")
            val tps = g.getDouble("gen_tok_s")
            ui {
                tokS.text = "%.1f".format(tps); devMs.text = "%.0f".format(modelMs)
            }
            light("model", true, "phone · ${"%.0f".format(modelMs)} ms")
            note("step $step · phone: ${g.getInt("prefilled_tokens")} new prompt tok in ${"%.0f".format(g.getDouble("prefill_ms"))} ms " +
                "(${g.getInt("reused_tokens")} reused from KV), ${g.getInt("gen_tokens")} tok at ${"%.1f".format(tps)} tok/s")
            note("model> ${g.getString("text").trim()}")
            val stats = JSONObject(g.toString()).apply { remove("text") }
            parts += "s$step ${if (g.getString("text").trimStart().startsWith("{")) "call" else "answer"}: prefill " +
                "${g.getInt("prefilled_tokens")}t/${"%.0f".format(g.getDouble("prefill_ms"))}ms (reused ${g.getInt("reused_tokens")}) " +
                "decode ${g.getInt("gen_tokens")}t/${"%.0f".format(g.getDouble("gen_ms"))}ms"
            steps.put(JSONObject().put("kind", if (g.getString("text").trimStart().startsWith("{")) "call" else "answer")
                .put("prompt_tokens", g.getInt("prompt_tokens")).put("reused_tokens", g.getInt("reused_tokens"))
                .put("prefill_tokens", g.getInt("prefilled_tokens")).put("prefill_ms", g.getDouble("prefill_ms"))
                .put("decode_tokens", g.getInt("gen_tokens")).put("decode_ms", g.getDouble("gen_ms"))
                .put("capped", g.optBoolean("capped")).put("output", g.getString("text")))
            r = timed(JSONObject().put("session", sid).put("model_output", g.getString("text"))
                .put("capped", g.optBoolean("capped")).put("stats", stats))
        }
        showEvents(r.optJSONArray("events"))
        val wall = SystemClock.elapsedRealtime() - t0
        val breakdown = "${tempC()} · threads $threads · " + parts.joinToString(" · ") + " · http(:8310 incl. Mac) " +
            http.joinToString("+") + "ms" + (if (sttFinalMs >= 0) " · stt final ${sttFinalMs}ms" else "")
        sttFinalLast = sttFinalMs
        sttFinalMs = -1
        android.util.Log.i("LeapsTiming", "turn ${wall}ms · $breakdown")
        note(breakdown)
        val answer = r.optString("say")
        val macPart = wall - modelMs
        ui {
            if (genMs > 0) tokS.text = "%.1f".format(genTok * 1000.0 / genMs)
            devMs.text = "%.0f".format(modelMs); macMs.text = "%.0f".format(macPart)
        }
        light("tools", false, "Mac · ${"%.0f".format(macPart)} ms")
        light("tts", true, if (speaker.ready) "Kokoro…" else "Android TTS…")
        if (step == 0) light("model", null, "not needed")
        bubble(r.optString("display").ifEmpty { answer }, 21f, fg, Color.rgb(16, 48, 34))
        r.optString("error").takeIf { it.isNotEmpty() }?.let { note("server error: $it") }
        note("turn ${wall} ms · model on phone ${"%.0f".format(modelMs)} ms over $step step(s)" +
            if (r.optBoolean("fallback")) " · code fallback" else "")
        val sttMs = sttFinalLast
        rec?.put("steps", steps)?.put("http_ms", JSONArray(http))?.put("turn_ms", wall)?.put("model_ms", modelMs)
            ?.put("tool_calls", r.optJSONArray("tool_calls") ?: JSONArray())?.put("say", answer)
            ?.put("display", r.optString("display"))?.put("fallback", r.optBoolean("fallback"))
            ?.put("server_error", r.optString("error").ifEmpty { null })?.put("temp", tempC())?.put("stt_final_ms", if (sttMs >= 0) sttMs else null)
        speak(answer, false, answer = true, mute = mute) { ttfaMs, voices ->
            rec?.let {
                it.put("ttfa_ms", ttfaMs).put("end_to_audio_ms", if (ttfaMs >= 0) wall + ttfaMs else null)
                    .put("voice", when { voices.isEmpty() -> "none"; voices.size > 1 -> "kokoro+system"; else -> voices.first() })
                writeSoak(it)
            }
        }
    }

    private fun showEvents(events: JSONArray?) {
        if (events == null) return
        for (i in 0 until events.length()) {
            val e = events.getJSONObject(i)
            when (e.optString("type")) {
                "tool_call" -> bubble("→ ${e.optString("tool")}(${argsText(e.optJSONObject("args"))})" +
                    (e.optString("corrected_from").takeIf { it.isNotEmpty() }?.let { "  [fixed from $it]" } ?: ""),
                    15f, mac, null, mono = true)
                "tool_result" -> if (!e.optBoolean("ok")) note("← ${e.optString("tool")} error: ${e.optString("error")}")
                "filler" -> {}   // not spoken: synthesizing it would compete with the model for the CPU
                "error" -> note("check (${e.optString("stage")}): ${e.optString("message")}")
            }
        }
    }

    private fun argsText(a: JSONObject?): String =
        a?.keys()?.asSequence()?.joinToString(", ") { "$it=${a.get(it)}" } ?: ""

    private fun JSONArray.toIntArray() = IntArray(length()) { getInt(it) }

    companion object {
        /** The live activity, for the debug soak hook (DebugAskReceiver). */
        @Volatile var instance: MainActivity? = null
    }
}
