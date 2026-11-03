package com.leaps.voiceagent

import org.json.JSONObject

/**
 * The JNI boundary (src/main/cpp/agent_llama.cpp), and nothing else: a model handle, KV-prefix reuse,
 * greedy generation. Prompts, parsing, validation and tools live in the step server.
 * Not thread-safe: call from one worker thread.
 */
class LlamaBridge {
    private external fun nativeLoad(modelPath: String, loraPath: String?, nCtx: Int): Long
    private external fun nativeSetThreads(handle: Long, n: Int)
    private external fun nativeSystemInfo(): String
    private external fun nativeTokenCount(handle: Long, prompt: String): Int
    private external fun nativePrefill(handle: Long, tokens: IntArray): Double
    private external fun nativeBegin(handle: Long, prompt: String): Boolean
    private external fun nativeSampleMasked(handle: Long, mask: BooleanArray): Int
    private external fun nativeGenerate(handle: Long, tokens: IntArray, maxTokens: Int, answerMaxTokens: Int): ByteArray
    private external fun nativeReset(handle: Long)
    private external fun nativeSaveState(handle: Long, path: String): Boolean
    private external fun nativeLoadState(handle: Long, path: String): IntArray
    private external fun nativeFree(handle: Long)

    private var handle = 0L
    val isLoaded: Boolean get() = handle != 0L

    /** Base GGUF (the fused model) and an optional LoRA GGUF. False when either fails to load. */
    fun load(modelPath: String, loraPath: String? = null, contextTokens: Int = 4096, threads: Int = 4): Boolean {
        if (handle != 0L) return true
        handle = try { nativeLoad(modelPath, loraPath, contextTokens) } catch (e: UnsatisfiedLinkError) { 0L }
        if (handle != 0L) nativeSetThreads(handle, threads)
        return handle != 0L
    }

    fun setThreads(n: Int) { if (handle != 0L) nativeSetThreads(handle, n) }
    fun systemInfo(): String = runCatching { nativeSystemInfo() }.getOrDefault("")
    fun tokenCount(prompt: String): Int? = if (handle == 0L) null else nativeTokenCount(handle, prompt).takeIf { it >= 0 }

    /** Prefill tokens (the fixed system+tools prefix) so later prompts reuse their KV. Returns ms or -1. */
    fun prefill(tokens: IntArray): Double = if (handle == 0L) -1.0 else nativePrefill(handle, tokens)

    /** Greedy decode inside a token mask, driven from Kotlin (begin, then one sample per step). */
    fun begin(prompt: String): Boolean = handle != 0L && nativeBegin(handle, prompt)
    fun sampleMasked(mask: BooleanArray): Int = if (handle == 0L) -1 else nativeSampleMasked(handle, mask)

    /** {text, capped, ended, prompt_tokens, reused_tokens, prefilled_tokens, gen_tokens, prefill_ms, gen_ms, ...} */
    fun generate(tokens: IntArray, maxTokens: Int, answerMaxTokens: Int): JSONObject =
        if (handle == 0L) JSONObject().put("error", "model not loaded")
        else JSONObject(String(nativeGenerate(handle, tokens, maxTokens, answerMaxTokens), Charsets.UTF_8))

    /** Persist the prefix KV (what the cache holds now) so the next app start restores it instead of prefilling. */
    fun saveState(path: String): Boolean = handle != 0L && nativeSaveState(handle, path)

    /** Restore a saved prefix KV; returns its tokens (empty when there is none or it does not fit). */
    fun loadState(path: String): IntArray = if (handle == 0L) IntArray(0) else nativeLoadState(handle, path)

    fun reset() { if (handle != 0L) nativeReset(handle) }

    fun close() {
        if (handle != 0L) { nativeFree(handle); handle = 0L }
    }

    companion object {
        val available: Boolean by lazy { runCatching { System.loadLibrary("agent_llama") }.isSuccess }
    }
}
