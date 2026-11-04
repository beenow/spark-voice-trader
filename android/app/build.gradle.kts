plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// llama.cpp is NOT built here: ops/build-llama-android.sh cross-compiles it (pinned tag, see
// scripts/fetch_llama_cpp.sh) into src/main/jniLibs. Gradle builds only the small JNI shim.
android {
    namespace = "com.leaps.voiceagent"
    compileSdk = 35
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "com.leaps.voiceagent"
        minSdk = 33
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-hackathon"
        ndk { abiFilters += listOf("arm64-v8a") }
        // Prefilled (editable) step-server address: -PleapsServer=... or $LEAPS_SERVER at build time.
        val server = (project.findProperty("leapsServer") as String?) ?: System.getenv("LEAPS_SERVER")
            ?: "http://192.168.1.2:8310"
        buildConfigField("String", "DEFAULT_SERVER", "\"$server\"")
    }
    buildFeatures { buildConfig = true }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { jniLibs { useLegacyPackaging = true } }
}

// Optional on-device voice (OnDeviceVoice.kt): not in this repository. A private source dir is added when it exists
// ($LEAPS_PRIVATE_VOICE, default ~/leaps-voice-agent-private/voice); without it the app speaks with Android TTS.
val privateVoice = file(System.getenv("LEAPS_PRIVATE_VOICE") ?: "${System.getProperty("user.home")}/leaps-voice-agent-private/voice")
if (privateVoice.resolve("java").isDirectory) {
    android.sourceSets["main"].java.srcDir(privateVoice.resolve("java"))
    // the private voice runs an ONNX model; abiFilters above keeps arm64-v8a only
    dependencies { implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0") }
}
