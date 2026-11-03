package com.leaps.voiceagent

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/** The Mac step server (voice/step_server.py): renders prompts, runs tools/validators/slots/memory. */
class StepClient(private val baseUrl: String, private val passcode: String) {
    fun step(body: JSONObject): JSONObject = request("POST", "/step", body)
    fun prefix(): JSONObject = request("GET", "/prefix", null)

    private fun request(method: String, path: String, body: JSONObject?): JSONObject {
        val c = URL(baseUrl.trimEnd('/') + path).openConnection() as HttpURLConnection
        c.requestMethod = method
        c.connectTimeout = 5000
        c.readTimeout = 180000
        c.setRequestProperty("X-Passcode", passcode)
        if (body != null) {
            c.doOutput = true
            c.setRequestProperty("Content-Type", "application/json")
            c.outputStream.use { it.write(body.toString().toByteArray()) }
        }
        val code = c.responseCode
        val text = (if (code in 200..299) c.inputStream else c.errorStream)?.bufferedReader()?.readText() ?: ""
        if (code !in 200..299) throw RuntimeException("step server HTTP $code: ${text.take(200)}")
        return JSONObject(text)
    }
}
