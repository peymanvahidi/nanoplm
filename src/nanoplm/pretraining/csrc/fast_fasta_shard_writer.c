#define _GNU_SOURCE

#include <errno.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif

enum {
    TOKEN_PAD = 0,
    TOKEN_EOS = 1,
    TOKEN_UNK = 2,
    TOKEN_A = 4,
    TOKEN_L = 5,
    TOKEN_G = 6,
    TOKEN_V = 7,
    TOKEN_S = 8,
    TOKEN_R = 9,
    TOKEN_E = 10,
    TOKEN_D = 11,
    TOKEN_T = 12,
    TOKEN_I = 13,
    TOKEN_P = 14,
    TOKEN_K = 15,
    TOKEN_F = 16,
    TOKEN_Q = 17,
    TOKEN_N = 18,
    TOKEN_Y = 19,
    TOKEN_M = 20,
    TOKEN_H = 21,
    TOKEN_W = 22,
    TOKEN_C = 23,
    TOKEN_X = 24,
    TOKEN_BOS = 29
};

typedef struct {
    char *seq;
    size_t len;
} Sequence;

typedef struct {
    Sequence *items;
    size_t count;
    size_t cap;
} SequenceBatch;

typedef void (*progress_cb_t)(int phase, double progress, long long completed, long long total);

static void report_progress(
    progress_cb_t progress_cb,
    int phase,
    long long completed,
    long long total,
    long long *last_reported_percent) {
    if (!progress_cb || total <= 0) {
        return;
    }
    long long percent = (completed * 100LL) / total;
    if (percent > 100) {
        percent = 100;
    }
    if (percent <= *last_reported_percent) {
        return;
    }
    *last_reported_percent = percent;
    progress_cb(phase, (double)percent / 100.0, completed, total);
}

static void set_error(char *error_msg, size_t error_cap, const char *fmt, ...) {
    if (error_msg == NULL || error_cap == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(error_msg, error_cap, fmt, args);
    va_end(args);
}

static int batch_init(SequenceBatch *batch, size_t initial_cap) {
    batch->count = 0;
    batch->cap = (initial_cap > 0) ? initial_cap : 1024;
    batch->items = (Sequence *)calloc(batch->cap, sizeof(Sequence));
    return batch->items ? 0 : -1;
}

static void batch_clear(SequenceBatch *batch) {
    if (!batch || !batch->items) {
        return;
    }
    for (size_t i = 0; i < batch->count; i++) {
        free(batch->items[i].seq);
        batch->items[i].seq = NULL;
        batch->items[i].len = 0;
    }
    batch->count = 0;
}

static void batch_destroy(SequenceBatch *batch) {
    if (!batch) {
        return;
    }
    batch_clear(batch);
    free(batch->items);
    batch->items = NULL;
    batch->cap = 0;
}

static int batch_push(SequenceBatch *batch, char *seq, size_t len) {
    if (batch->count == batch->cap) {
        size_t new_cap = batch->cap * 2;
        Sequence *new_items =
            (Sequence *)realloc(batch->items, new_cap * sizeof(Sequence));
        if (!new_items) {
            return -1;
        }
        memset(new_items + batch->cap, 0, (new_cap - batch->cap) * sizeof(Sequence));
        batch->items = new_items;
        batch->cap = new_cap;
    }
    // Ensure seq is never NULL so tokenization loops can safely read src pointer.
    if (!seq) {
        seq = (char *)calloc(1, 1);
        if (!seq) {
            return -1;
        }
        len = 0;
    }
    batch->items[batch->count].seq = seq;
    batch->items[batch->count].len = len;
    batch->count += 1;
    return 0;
}

static int append_sequence_line(char **buf, size_t *len, size_t *cap, const char *line, size_t n) {
    if (n == 0) {
        return 0;
    }

    size_t needed = *len + n;
    if (needed > *cap) {
        size_t new_cap = *cap > 0 ? *cap : 256;
        while (new_cap < needed) {
            new_cap *= 2;
        }
        char *new_buf = (char *)realloc(*buf, new_cap);
        if (!new_buf) {
            return -1;
        }
        *buf = new_buf;
        *cap = new_cap;
    }

    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)line[i];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t' || c == '\v' || c == '\f') {
            continue;
        }
        (*buf)[*len] = (char)c;
        *len += 1;
    }
    return 0;
}

static void init_token_lut(uint8_t lut[256]) {
    for (int i = 0; i < 256; i++) {
        lut[i] = TOKEN_UNK;
    }

    lut[(unsigned char)'A'] = TOKEN_A;
    lut[(unsigned char)'L'] = TOKEN_L;
    lut[(unsigned char)'G'] = TOKEN_G;
    lut[(unsigned char)'V'] = TOKEN_V;
    lut[(unsigned char)'S'] = TOKEN_S;
    lut[(unsigned char)'R'] = TOKEN_R;
    lut[(unsigned char)'E'] = TOKEN_E;
    lut[(unsigned char)'D'] = TOKEN_D;
    lut[(unsigned char)'T'] = TOKEN_T;
    lut[(unsigned char)'I'] = TOKEN_I;
    lut[(unsigned char)'P'] = TOKEN_P;
    lut[(unsigned char)'K'] = TOKEN_K;
    lut[(unsigned char)'F'] = TOKEN_F;
    lut[(unsigned char)'Q'] = TOKEN_Q;
    lut[(unsigned char)'N'] = TOKEN_N;
    lut[(unsigned char)'Y'] = TOKEN_Y;
    lut[(unsigned char)'M'] = TOKEN_M;
    lut[(unsigned char)'H'] = TOKEN_H;
    lut[(unsigned char)'W'] = TOKEN_W;
    lut[(unsigned char)'C'] = TOKEN_C;
    lut[(unsigned char)'X'] = TOKEN_X;

    // Match tokenizer normalization: [UZOB] -> X
    lut[(unsigned char)'U'] = TOKEN_X;
    lut[(unsigned char)'Z'] = TOKEN_X;
    lut[(unsigned char)'O'] = TOKEN_X;
    lut[(unsigned char)'B'] = TOKEN_X;

    for (int c = 'A'; c <= 'Z'; c++) {
        lut[(unsigned char)(c + 32)] = lut[(unsigned char)c];
    }
}

static inline int encoded_length(size_t seq_len, int max_length, int use_bos_token) {
    int special = use_bos_token ? 2 : 1;
    if (max_length <= 0) {
        return 0;
    }
    if (max_length <= special) {
        return max_length;
    }
    int allowed_seq = max_length - special;
    int used_seq = seq_len < allowed_seq ? seq_len : allowed_seq;
    return used_seq + special;
}

static int write_npy_int32(const char *path, const int32_t *data, size_t n, char *error_msg,
                           size_t error_cap) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        set_error(error_msg, error_cap, "Failed to open %s: %s", path, strerror(errno));
        return -1;
    }

    char dict[128];
    int dict_len = snprintf(dict, sizeof(dict),
                            "{'descr': '<i4', 'fortran_order': False, 'shape': (%zu,), }", n);
    if (dict_len <= 0 || (size_t)dict_len >= sizeof(dict)) {
        fclose(fp);
        set_error(error_msg, error_cap, "Failed to create NPY header for %s", path);
        return -1;
    }

    size_t preamble_len = 10;  // magic (6) + version (2) + header_len (2)
    size_t unpadded = preamble_len + (size_t)dict_len + 1;
    size_t pad = (16 - (unpadded % 16)) % 16;
    size_t header_len = (size_t)dict_len + 1 + pad;

    if (header_len > 65535) {
        fclose(fp);
        set_error(error_msg, error_cap, "NPY header too large for %s", path);
        return -1;
    }

    uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    if (fwrite(magic, 1, sizeof(magic), fp) != sizeof(magic)) {
        fclose(fp);
        set_error(error_msg, error_cap, "Failed to write NPY magic for %s", path);
        return -1;
    }

    uint16_t h16 = (uint16_t)header_len;
    uint8_t hbytes[2] = {(uint8_t)(h16 & 0xFF), (uint8_t)((h16 >> 8) & 0xFF)};
    if (fwrite(hbytes, 1, 2, fp) != 2) {
        fclose(fp);
        set_error(error_msg, error_cap, "Failed to write NPY header length for %s", path);
        return -1;
    }

    char *header = (char *)malloc(header_len);
    if (!header) {
        fclose(fp);
        set_error(error_msg, error_cap, "Out of memory for NPY header");
        return -1;
    }
    memset(header, ' ', header_len);
    memcpy(header, dict, (size_t)dict_len);
    header[header_len - 1] = '\n';

    if (fwrite(header, 1, header_len, fp) != header_len) {
        free(header);
        fclose(fp);
        set_error(error_msg, error_cap, "Failed to write NPY header for %s", path);
        return -1;
    }
    free(header);

    if (n > 0 && fwrite(data, sizeof(int32_t), n, fp) != n) {
        fclose(fp);
        set_error(error_msg, error_cap, "Failed to write NPY data for %s", path);
        return -1;
    }

    if (fclose(fp) != 0) {
        set_error(error_msg, error_cap, "Failed to close %s: %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

static int tokenize_and_write_shard(const SequenceBatch *batch, const char *output_dir, int shard_idx,
                                    int max_length, int use_bos_token, int num_threads,
                                    const uint8_t lut[256], char *error_msg, size_t error_cap) {
    char bin_path[4096];
    char idx_path[4096];

    if (snprintf(bin_path, sizeof(bin_path), "%s/shard_%04d.bin", output_dir, shard_idx) >=
        (int)sizeof(bin_path)) {
        set_error(error_msg, error_cap, "Output path too long for .bin shard");
        return -1;
    }
    if (snprintf(idx_path, sizeof(idx_path), "%s/shard_%04d.idx.npy", output_dir, shard_idx) >=
        (int)sizeof(idx_path)) {
        set_error(error_msg, error_cap, "Output path too long for .idx shard");
        return -1;
    }

    FILE *bin_fp = fopen(bin_path, "wb");
    if (!bin_fp) {
        set_error(error_msg, error_cap, "Failed to open %s: %s", bin_path, strerror(errno));
        return -1;
    }

    size_t n = batch->count;
    int32_t *lengths = (int32_t *)malloc(n * sizeof(int32_t));
    size_t *offsets = (size_t *)malloc(n * sizeof(size_t));
    if (!lengths || !offsets) {
        fclose(bin_fp);
        free(lengths);
        free(offsets);
        set_error(error_msg, error_cap, "Out of memory while preparing shard");
        return -1;
    }

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        lengths[i] = (int32_t)encoded_length(batch->items[i].len, max_length, use_bos_token);
    }

    size_t total_tokens = 0;
    for (size_t i = 0; i < n; i++) {
        offsets[i] = total_tokens;
        total_tokens += (size_t)lengths[i];
    }

    uint8_t *tokens = (uint8_t *)malloc(total_tokens > 0 ? total_tokens : 1);
    if (!tokens) {
        fclose(bin_fp);
        free(lengths);
        free(offsets);
        set_error(error_msg, error_cap, "Out of memory while tokenizing shard");
        return -1;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        size_t off = offsets[i];
        int out_len = (int)lengths[i];
        int pos = 0;

        if (out_len <= 0) {
            continue;
        }

        if (use_bos_token) {
            tokens[off + (size_t)pos] = TOKEN_BOS;
            pos += 1;
            if (pos >= out_len) {
                continue;
            }
        }

        int slots_for_seq = out_len - pos - 1;  // Keep last slot for EOS.
        if (slots_for_seq < 0) {
            slots_for_seq = 0;
        }
        size_t copy_len = batch->items[i].len < (size_t)slots_for_seq ? batch->items[i].len : (size_t)slots_for_seq;
        const unsigned char *src = (const unsigned char *)batch->items[i].seq;

#if defined(_OPENMP)
#pragma omp simd
#endif
        for (size_t j = 0; j < copy_len; j++) {
            tokens[off + (size_t)(pos + j)] = lut[src[j]];
        }
        pos += copy_len;

        if (pos < out_len) {
            tokens[off + (size_t)pos] = TOKEN_EOS;
        }
    }

    if (total_tokens > 0 && fwrite(tokens, 1, total_tokens, bin_fp) != total_tokens) {
        free(tokens);
        free(lengths);
        free(offsets);
        fclose(bin_fp);
        set_error(error_msg, error_cap, "Failed to write %s", bin_path);
        return -1;
    }

    free(tokens);

    if (fclose(bin_fp) != 0) {
        free(lengths);
        free(offsets);
        set_error(error_msg, error_cap, "Failed to close %s: %s", bin_path, strerror(errno));
        return -1;
    }

    int npy_status = write_npy_int32(idx_path, lengths, n, error_msg, error_cap);
    free(lengths);
    free(offsets);
    return npy_status;
}

API_EXPORT int nanoplm_create_fasta_shards(
    const char *fasta_path,
    const char *output_dir,
    int max_length,
    int samples_per_shard,
    int num_threads,
    int use_bos_token,
    progress_cb_t progress_cb,
    int *out_num_shards,
    long long *out_num_sequences,
    char *error_msg,
    size_t error_cap) {
    if (!fasta_path || !output_dir || !out_num_shards || !out_num_sequences) {
        set_error(error_msg, error_cap, "Invalid null argument");
        return -1;
    }
    if (max_length < 1) {
        set_error(error_msg, error_cap, "max_length must be at least 1");
        return -1;
    }
    if (samples_per_shard < 1) {
        set_error(error_msg, error_cap, "samples_per_shard must be at least 1");
        return -1;
    }
    if (num_threads < 1) {
        num_threads = 1;
    }

    FILE *fp = fopen(fasta_path, "rb");
    if (!fp) {
        set_error(error_msg, error_cap, "Failed to open FASTA %s: %s", fasta_path, strerror(errno));
        return -1;
    }
    setvbuf(fp, NULL, _IOFBF, 4 * 1024 * 1024);

    struct stat st;
    if (fstat(fileno(fp), &st) != 0) {
        fclose(fp);
        set_error(error_msg, error_cap, "Failed to stat FASTA %s: %s", fasta_path, strerror(errno));
        return -1;
    }
    long long input_size = (long long)st.st_size;

    uint8_t lut[256];
    init_token_lut(lut);

    SequenceBatch batch;
    if (batch_init(&batch, (size_t)samples_per_shard) != 0) {
        fclose(fp);
        set_error(error_msg, error_cap, "Out of memory while initializing sequence batch");
        return -1;
    }

    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len = 0;
    int seen_header = 0;

    char *current_seq = NULL;
    size_t current_len = 0;
    size_t current_cap = 0;

    int shard_idx = 0;
    long long total_sequences = 0;
    int status = 0;
    long long bytes_read = 0;
    long long last_reported_percent = -1;

    while ((line_len = getline(&line, &line_cap, fp)) != -1) {
        bytes_read += (long long)line_len;
        if (line_len > 0 && line[0] == '>') {
            if (seen_header) {
                if (batch_push(&batch, current_seq, current_len) != 0) {
                    set_error(error_msg, error_cap, "Out of memory while storing sequences");
                    status = -1;
                    goto cleanup;
                }
                current_seq = NULL;
                current_len = 0;
                current_cap = 0;
            } else {
                seen_header = 1;
            }

            if (batch.count == (size_t)samples_per_shard) {
                if (tokenize_and_write_shard(&batch, output_dir, shard_idx, max_length, use_bos_token,
                                             num_threads, lut, error_msg, error_cap) != 0) {
                    status = -1;
                    goto cleanup;
                }
                total_sequences += (long long)batch.count;
                shard_idx += 1;
                batch_clear(&batch);
            }
        } else if (seen_header) {
            if (append_sequence_line(&current_seq, &current_len, &current_cap, line, (size_t)line_len) != 0) {
                set_error(error_msg, error_cap, "Out of memory while reading FASTA sequence");
                status = -1;
                goto cleanup;
            }
        }

        long long report_bytes = bytes_read;
        if (report_bytes >= input_size && input_size > 1) {
            report_bytes = input_size - 1;
        }
        report_progress(progress_cb, 1, report_bytes, input_size, &last_reported_percent);
    }

    if (ferror(fp)) {
        set_error(error_msg, error_cap, "Error while reading FASTA %s", fasta_path);
        status = -1;
        goto cleanup;
    }

    if (seen_header) {
        if (batch_push(&batch, current_seq, current_len) != 0) {
            set_error(error_msg, error_cap, "Out of memory while finalizing sequences");
            status = -1;
            goto cleanup;
        }
        current_seq = NULL;
        current_len = 0;
        current_cap = 0;
    }

    if (batch.count > 0) {
        if (tokenize_and_write_shard(&batch, output_dir, shard_idx, max_length, use_bos_token, num_threads,
                                     lut, error_msg, error_cap) != 0) {
            status = -1;
            goto cleanup;
        }
        total_sequences += (long long)batch.count;
        shard_idx += 1;
        batch_clear(&batch);
    }

    if (total_sequences == 0) {
        set_error(error_msg, error_cap, "No sequences found in FASTA file");
        status = -1;
        goto cleanup;
    }

    *out_num_shards = shard_idx;
    *out_num_sequences = total_sequences;
    report_progress(progress_cb, 1, input_size, input_size, &last_reported_percent);

cleanup:
    free(line);
    free(current_seq);
    batch_destroy(&batch);
    fclose(fp);
    return status;
}
