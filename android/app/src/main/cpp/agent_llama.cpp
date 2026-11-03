// The JNI shim between Kotlin and llama.cpp: a model handle (base GGUF + optional LoRA GGUF), KV-prefix
// reuse across calls, and greedy generation. Everything else (prompt rendering, parsing, validation,
// tools) lives off-device in the step server; this file owns only what cannot cross JNI.
#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>
#include "llama.h"

#define LOG_TAG "AgentLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Session {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    llama_adapter_lora *lora = nullptr;
    const llama_vocab *vocab = nullptr;
    int n_batch = 512;
    // Tokens whose KV occupies positions [0, cached.size()) of sequence 0 (prompt + fed generation).
    std::vector<llama_token> cached;
};

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Evaluate `tokens` as the whole prompt, REUSING the KV of the longest prefix shared with what the
// cache holds: the fixed system+tools prefix is prefilled once per session, and within a turn only the
// new tool results are processed. The last token is always re-evaluated so its logits are fresh. Any
// failure falls back to a cleared cache. Returns the reused count, or -1 on a decode error.
int evalPrompt(Session *s, const std::vector<llama_token> &tokens) {
    if (tokens.empty()) return -1;
    llama_memory_t mem = llama_get_memory(s->ctx);
    size_t keep = 0;
    while (keep < s->cached.size() && keep < tokens.size() && s->cached[keep] == tokens[keep]) keep++;
    if (keep == tokens.size()) keep--;
    if (keep == 0 || !llama_memory_seq_rm(mem, 0, (llama_pos) keep, -1)) {
        llama_memory_clear(mem, true);
        keep = 0;
    }
    s->cached.assign(tokens.begin(), tokens.begin() + (long) keep);
    std::vector<llama_token> rest(tokens.begin() + (long) keep, tokens.end());
    for (size_t i = 0; i < rest.size(); i += s->n_batch) {
        int n = (int) std::min((size_t) s->n_batch, rest.size() - i);
        llama_batch batch = llama_batch_get_one(rest.data() + i, n);
        if (llama_decode(s->ctx, batch) != 0) {
            llama_memory_clear(mem, true);
            s->cached.clear();
            return -1;
        }
    }
    s->cached = tokens;
    return (int) keep;
}

std::vector<llama_token> tokenize(Session *s, const std::string &text) {
    // add_special=false: the rendered prompt already starts with <|begin_of_text|> (parse_special=true).
    int n = -llama_tokenize(s->vocab, text.c_str(), (int) text.size(), nullptr, 0, false, true);
    if (n <= 0) return {};
    std::vector<llama_token> t(n);
    if (llama_tokenize(s->vocab, text.c_str(), (int) text.size(), t.data(), n, false, true) < 0) return {};
    return t;
}

std::vector<llama_token> fromJava(JNIEnv *env, jintArray arr) {
    jsize n = env->GetArrayLength(arr);
    std::vector<llama_token> t(n);
    env->GetIntArrayRegion(arr, 0, n, reinterpret_cast<jint *>(t.data()));
    return t;
}

jbyteArray bytes(JNIEnv *env, const std::string &s) {
    jbyteArray a = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(a, 0, (jsize) s.size(), reinterpret_cast<const jbyte *>(s.data()));
    return a;
}

std::string jsonEscape(const std::string &s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char) c;
        }
    }
    return o;
}

int sentenceEnds(const std::string &s) {
    int c = 0;
    for (size_t i = 0; i < s.size(); i++)
        if ((s[i] == '.' || s[i] == '!' || s[i] == '?') && (i + 1 == s.size() || isspace((unsigned char) s[i + 1]))) c++;
    return c;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeLoad(JNIEnv *env, jobject, jstring model_path, jstring lora_path,
                                                 jint n_ctx) {
    static bool initialised = false;
    if (!initialised) { llama_backend_init(); initialised = true; }

    const char *mp = env->GetStringUTFChars(model_path, nullptr);
    auto *s = new Session();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU only
    s->model = llama_model_load_from_file(mp, mparams);
    env->ReleaseStringUTFChars(model_path, mp);
    if (!s->model) { LOGE("model failed to load"); delete s; return 0; }
    s->vocab = llama_model_get_vocab(s->model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? (uint32_t) n_ctx : 2048;
    cparams.n_batch = s->n_batch;
    cparams.n_ubatch = s->n_batch;
    s->ctx = llama_init_from_model(s->model, cparams);
    if (!s->ctx) { LOGE("context failed"); llama_model_free(s->model); delete s; return 0; }

    if (lora_path != nullptr) {
        const char *lp = env->GetStringUTFChars(lora_path, nullptr);
        s->lora = llama_adapter_lora_init(s->model, lp);
        env->ReleaseStringUTFChars(lora_path, lp);
        if (!s->lora) {
            LOGE("lora failed to load");
            llama_free(s->ctx); llama_model_free(s->model); delete s;
            return 0;
        }
        llama_adapter_lora *adapters[1] = {s->lora};
        float scales[1] = {1.0f};
        llama_set_adapters_lora(s->ctx, adapters, 1, scales);
    }
    LOGI("loaded");
    return reinterpret_cast<jlong>(s);
}

JNIEXPORT void JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeSetThreads(JNIEnv *, jobject, jlong handle, jint n) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (s && s->ctx && n > 0) llama_set_n_threads(s->ctx, n, n);
}

JNIEXPORT jstring JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeSystemInfo(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

// Prompt length in tokens, tokenised as nativeBegin tokenises it. -1 when unusable.
JNIEXPORT jint JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeTokenCount(JNIEnv *env, jobject, jlong handle, jstring prompt) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx || !prompt) return -1;
    const char *p = env->GetStringUTFChars(prompt, nullptr);
    std::string text(p);
    env->ReleaseStringUTFChars(prompt, p);
    auto t = tokenize(s, text);
    return t.empty() ? -1 : (jint) t.size();
}

// Prefill only (e.g. the fixed system+tools prefix at startup). Returns ms, or -1.
JNIEXPORT jdouble JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativePrefill(JNIEnv *env, jobject, jlong handle, jintArray jtoks) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx) return -1;
    double t0 = now_ms();
    return evalPrompt(s, fromJava(env, jtoks)) < 0 ? -1 : now_ms() - t0;
}

// Tokenize + evaluate a text prompt; next-token logits are then ready for nativeSampleMasked.
JNIEXPORT jboolean JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeBegin(JNIEnv *env, jobject, jlong handle, jstring prompt) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx) return JNI_FALSE;
    const char *p = env->GetStringUTFChars(prompt, nullptr);
    std::string text(p);
    env->ReleaseStringUTFChars(prompt, p);
    auto t = tokenize(s, text);
    return !t.empty() && evalPrompt(s, t) >= 0 ? JNI_TRUE : JNI_FALSE;
}

// Greedy over the tokens `mask` allows; evaluated at once unless it ends generation. -1 on failure.
JNIEXPORT jint JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeSampleMasked(JNIEnv *env, jobject, jlong handle, jbooleanArray mask) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx || !mask) return -1;
    const int n = std::min<int>(llama_vocab_n_tokens(s->vocab), env->GetArrayLength(mask));
    jboolean *allowed = env->GetBooleanArrayElements(mask, nullptr);
    if (!allowed) return -1;
    const float *logits = llama_get_logits_ith(s->ctx, -1);
    llama_token best = -1;
    float best_logit = 0.0f;
    for (int t = 0; logits && t < n; t++) {
        if (!allowed[t]) continue;
        if (best < 0 || logits[t] > best_logit) { best = t; best_logit = logits[t]; }
    }
    env->ReleaseBooleanArrayElements(mask, allowed, JNI_ABORT);
    if (best < 0) return -1;
    if (!llama_vocab_is_eog(s->vocab, best)) {
        llama_batch next = llama_batch_get_one(&best, 1);
        if (llama_decode(s->ctx, next) != 0) return -1;
        s->cached.push_back(best);
    }
    return best;
}

// Greedy generation from a token prompt. Stops at end-of-generation, at max_tokens, or -- for an answer
// (output not starting with '{') -- at answer_max_tokens or after two complete sentences: the same rules
// as the Mac backend (agent/backends.py LlamaBackend.decide). Returns UTF-8 JSON bytes: text + timings.
JNIEXPORT jbyteArray JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeGenerate(JNIEnv *env, jobject, jlong handle, jintArray jtoks,
                                                     jint max_tokens, jint answer_max_tokens) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx) return bytes(env, "{\"error\":\"model not loaded\"}");
    auto tokens = fromJava(env, jtoks);
    if (tokens.empty()) return bytes(env, "{\"error\":\"empty prompt\"}");
    if ((int) tokens.size() + max_tokens >= (int) llama_n_ctx(s->ctx))
        return bytes(env, "{\"error\":\"prompt too long for context\"}");

    double t0 = now_ms();
    int reused = evalPrompt(s, tokens);
    if (reused < 0) return bytes(env, "{\"error\":\"decode failed\"}");
    double t_prefill = now_ms();

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    std::string out;
    int n_gen = 0;
    bool capped = false, ended = false;
    double t_first = 0;
    while (n_gen < max_tokens) {
        llama_token tok = llama_sampler_sample(sampler, s->ctx, -1);
        if (n_gen == 0) t_first = now_ms();
        n_gen++;
        if (llama_vocab_is_eog(s->vocab, tok)) { ended = true; break; }
        char buf[256];
        int n = llama_token_to_piece(s->vocab, tok, buf, sizeof(buf), 0, false);
        if (n > 0) out.append(buf, n);
        size_t b = out.find_first_not_of(" \t\n\r");
        if (b != std::string::npos && out[b] != '{') {
            std::string body = out.substr(b);
            size_t e = body.find_last_not_of(" \t\n\r");
            char last = e == std::string::npos ? 0 : body[e];
            if (n_gen >= answer_max_tokens ||
                (sentenceEnds(body) >= 2 && (last == '.' || last == '!' || last == '?'))) {
                capped = true;
                break;
            }
        }
        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(s->ctx, next) != 0) break;
        s->cached.push_back(tok);
    }
    llama_sampler_free(sampler);
    double t_end = now_ms();
    double pre = t_prefill - t0, gen = t_end - t_prefill;
    int prefilled = (int) tokens.size() - reused;
    char stats[512];
    snprintf(stats, sizeof stats,
             "\"prompt_tokens\":%d,\"reused_tokens\":%d,\"prefilled_tokens\":%d,\"gen_tokens\":%d,"
             "\"prefill_ms\":%.1f,\"first_token_ms\":%.1f,\"gen_ms\":%.1f,\"total_ms\":%.1f,"
             "\"gen_tok_s\":%.2f,\"prefill_tok_s\":%.2f,\"ended\":%s,\"capped\":%s",
             (int) tokens.size(), reused, prefilled, n_gen, pre, t_first - t0, gen, t_end - t0,
             gen > 0 ? n_gen * 1000.0 / gen : 0.0, pre > 0 ? prefilled * 1000.0 / pre : 0.0,
             ended ? "true" : "false", capped ? "true" : "false");
    return bytes(env, "{\"text\":\"" + jsonEscape(out) + "\"," + stats + "}");
}

// Save the KV of what sequence 0 holds (normally the system+tools prefix after a prefill) with its tokens.
JNIEXPORT jboolean JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeSaveState(JNIEnv *env, jobject, jlong handle, jstring path) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx || s->cached.empty()) return JNI_FALSE;
    const char *p = env->GetStringUTFChars(path, nullptr);
    size_t n = llama_state_seq_save_file(s->ctx, p, 0, s->cached.data(), s->cached.size());
    env->ReleaseStringUTFChars(path, p);
    return n > 0 ? JNI_TRUE : JNI_FALSE;
}

// Restore a saved prefix KV into sequence 0: returns its tokens (empty on failure; the cache is then clear).
JNIEXPORT jintArray JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeLoadState(JNIEnv *env, jobject, jlong handle, jstring path) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx) return env->NewIntArray(0);
    llama_memory_t mem = llama_get_memory(s->ctx);
    llama_memory_clear(mem, true);
    s->cached.clear();
    std::vector<llama_token> toks(llama_n_ctx(s->ctx));
    size_t n = 0;
    const char *p = env->GetStringUTFChars(path, nullptr);
    size_t r = llama_state_seq_load_file(s->ctx, p, 0, toks.data(), toks.size(), &n);
    env->ReleaseStringUTFChars(path, p);
    if (r == 0 || n == 0) { llama_memory_clear(mem, true); return env->NewIntArray(0); }
    toks.resize(n);
    s->cached = toks;
    jintArray out = env->NewIntArray((jsize) n);
    env->SetIntArrayRegion(out, 0, (jsize) n, reinterpret_cast<const jint *>(toks.data()));
    return out;
}

JNIEXPORT void JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeReset(JNIEnv *, jobject, jlong handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s || !s->ctx) return;
    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->cached.clear();
}

JNIEXPORT void JNICALL
Java_com_leaps_voiceagent_LlamaBridge_nativeFree(JNIEnv *, jobject, jlong handle) {
    auto *s = reinterpret_cast<Session *>(handle);
    if (!s) return;
    if (s->lora) llama_adapter_lora_free(s->lora);
    if (s->ctx) llama_free(s->ctx);
    if (s->model) llama_model_free(s->model);
    delete s;
}

}  // extern "C"
