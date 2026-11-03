package com.leaps.voiceagent

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * The model files in app storage, downloaded once from the step server (GET /model/manifest.json, then
 * each file) with a passcode header, Range resume of a partial `.part` file, progress, and a sha256 check
 * against the manifest before a file is renamed into place. A verified file has a `<name>.ok` marker.
 *
 *  - kind "llm": the GGUF, in filesDir/model/.
 *  - kind "tts": the Kokoro-82M ONNX voice bundle (roles model / voice / tokenizer), in filesDir/kokoro/
 *    as model.onnx, voice.bin, tokenizer.json.
 */
class ModelStore(private val context: Context) {
    private fun dir(): File = File(context.filesDir, "model").also { it.mkdirs() }
    fun kokoroDir(): File = File(context.filesDir, "kokoro").also { it.mkdirs() }

    /** The installed GGUF (the one whose name contains [prefer], when given), or null. */
    fun installed(prefer: String = ""): File? = dir().listFiles()?.firstOrNull {
        it.name.endsWith(".gguf") && File(dir(), it.name + ".ok").isFile && it.name.contains(prefer)
    }

    /** True when all three Kokoro files are present and verified. */
    fun kokoroInstalled(): Boolean = KOKORO_ROLES.values.all {
        File(kokoroDir(), it).isFile && File(kokoroDir(), "$it.ok").isFile
    }

    fun remove() { dir().listFiles()?.forEach { it.delete() } }

    fun manifest(baseUrl: String, passcode: String): JSONObject =
        JSONObject(get("${baseUrl.trimEnd('/')}/model/manifest.json", passcode).inputStream.bufferedReader().readText())

    private fun entries(m: JSONObject, kind: String): List<JSONObject> {
        val files = m.getJSONArray("files")
        return (0 until files.length()).map { files.getJSONObject(it) }.filter {
            it.optString("kind", if (it.getString("name").endsWith(".gguf")) "llm" else "") == kind
        }
    }

    /**
     * Download (or resume) the GGUF. [onProgress] gets (done, total) bytes; [cancelled] is polled between
     * chunks. Returns the verified file; throws with a readable message.
     */
    fun download(baseUrl: String, passcode: String, onProgress: (Long, Long) -> Unit,
                 cancelled: () -> Boolean = { false }, manifest: JSONObject? = null, prefer: String = ""): File {
        val m = manifest ?: manifest(baseUrl, passcode)
        val llms = entries(m, "llm")
        val entry = llms.firstOrNull { it.getString("name").contains(prefer) } ?: llms.firstOrNull() ?: error("server lists no model")
        val dest = File(dir(), entry.getString("name"))
        if (dest.isFile && File(dir(), "${dest.name}.ok").isFile) return dest
        fetch(baseUrl, passcode, entry, dest, { d, _ -> onProgress(d, entry.getLong("size")) }, cancelled)
        // One model on the phone at a time (storage): only once the new one is verified, drop the others
        // (GGUF, marker, prefix KV, partial downloads). A loaded model keeps working: its mapping survives.
        dir().listFiles()?.filter { !it.name.startsWith(dest.name) }?.forEach { it.delete() }
        return dest
    }

    /**
     * Download (or resume) the Kokoro bundle. Returns the server's tts settings ({voice, speed}) or null
     * when the server has no voice bundle.
     */
    fun downloadKokoro(baseUrl: String, passcode: String, onProgress: (Long, Long) -> Unit,
                       cancelled: () -> Boolean = { false }, manifest: JSONObject? = null): JSONObject? {
        val m = manifest ?: manifest(baseUrl, passcode)
        val list = entries(m, "tts").filter { KOKORO_ROLES.containsKey(it.optString("role")) }
        if (list.map { it.getString("role") }.toSet() != KOKORO_ROLES.keys) return null
        val total = list.sumOf { it.getLong("size") }
        var base = 0L
        for (e in list) {
            val dest = File(kokoroDir(), KOKORO_ROLES.getValue(e.getString("role")))
            val ok = File(kokoroDir(), "${dest.name}.ok")
            if (!(dest.isFile && ok.isFile && ok.readText().trim() == e.getString("sha256").lowercase())) {
                ok.delete()
                fetch(baseUrl, passcode, e, dest, { d, _ -> onProgress(base + d, total) }, cancelled)
            }
            base += e.getLong("size")
        }
        return m.optJSONObject("tts")
    }

    /** One manifest entry into [dest]: resume `.part`, verify size + sha256, rename, write the `.ok` marker. */
    private fun fetch(baseUrl: String, passcode: String, entry: JSONObject, dest: File,
                      onProgress: (Long, Long) -> Unit, cancelled: () -> Boolean) {
        val root = baseUrl.trimEnd('/')
        val want = entry.getString("sha256").lowercase()
        val total = entry.getLong("size")
        val part = File(dest.parentFile, "${dest.name}.part")
        var done = if (part.isFile) part.length() else 0L
        if (done > total) { part.delete(); done = 0L }
        if (done < total) {
            val conn = get(root + entry.getString("url"), passcode, if (done > 0) "bytes=$done-" else null)
            if (done > 0 && conn.responseCode != 206) done = 0L          // server ignored Range: start over
            FileOutputStream(part, done > 0).use { out ->
                conn.inputStream.use { input ->
                    val buf = ByteArray(1 shl 20)
                    while (true) {
                        if (cancelled()) error("cancelled")
                        val n = input.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        done += n
                        onProgress(done, total)
                    }
                }
            }
        }
        if (part.length() != total) error("download incomplete (${part.length()} of $total bytes); try again to resume")
        onProgress(total, total)
        if (sha256(part) != want) { part.delete(); error("sha256 mismatch for ${dest.name}; deleted, download again") }
        dest.delete()
        if (!part.renameTo(dest)) error("could not move ${dest.name} into place")
        File(dest.parentFile, "${dest.name}.ok").writeText(want)
    }

    private fun get(url: String, passcode: String, range: String? = null): HttpURLConnection {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 8000
        c.readTimeout = 60000
        c.setRequestProperty("X-Passcode", passcode)
        if (range != null) c.setRequestProperty("Range", range)
        val code = c.responseCode
        if (code == 401) error("wrong passcode")
        if (code !in 200..299) error("server returned HTTP $code for $url")
        return c
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        RandomAccessFile(f, "r").use { raf ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = raf.read(buf)
                if (n <= 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        /** Manifest role -> file name KokoroSynth reads. */
        val KOKORO_ROLES = mapOf("model" to "model.onnx", "voice" to "voice.bin", "tokenizer" to "tokenizer.json")
    }
}
