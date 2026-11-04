package com.leaps.voiceagent

import android.content.Context
import java.io.File

/**
 * An optional on-device neural voice (24 kHz mono float PCM). This repository ships none: answers are spoken by
 * Android [android.speech.tts.TextToSpeech]. A build that has a voice provider on its source path (see
 * android/README.md, "Voice") gets it through [load]; the demo video used a proprietary Kokoro-82M voice this way.
 */
interface OnDeviceVoice : AutoCloseable {
    /** Short label for logs and the screen (execution provider, threads). */
    val provider: String

    /** PCM for one sentence, or null to say that sentence with the system voice. */
    fun render(text: String, speed: Float): FloatArray?

    companion object {
        const val SAMPLE_RATE = 24_000
        private const val PROVIDER_CLASS = "com.leaps.voiceagent.privatevoice.Provider"

        /** The provider's voice, or null when this build has none. Throws when a provider exists but fails. */
        fun load(context: Context, dir: File, threads: Int, xnnpack: Boolean, spin: Boolean, xnnThreads: Int): OnDeviceVoice? {
            val c = try { Class.forName(PROVIDER_CLASS) } catch (e: ClassNotFoundException) { return null }
            val m = c.getMethod("load", Context::class.java, File::class.java, Int::class.javaPrimitiveType,
                Boolean::class.javaPrimitiveType, Boolean::class.javaPrimitiveType, Int::class.javaPrimitiveType)
            return try {
                m.invoke(null, context, dir, threads, xnnpack, spin, xnnThreads) as OnDeviceVoice
            } catch (e: java.lang.reflect.InvocationTargetException) { throw e.targetException }
        }
    }
}
