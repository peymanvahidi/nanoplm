#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_WIN32)
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} DynBuf;

typedef struct {
    uint64_t src;
    uint64_t dst;
    uint64_t len;
} CopyTask;

typedef void (*progress_cb_t)(int phase, double progress, long long completed, long long total);
typedef void (*shuffle_progress_cb_t)(int phase, double progress, long long completed, long long total,
                                      long long aux);

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int compare_copy_task_by_src(const void *a, const void *b) {
    const CopyTask *lhs = (const CopyTask *)a;
    const CopyTask *rhs = (const CopyTask *)b;
    if (lhs->src < rhs->src) {
        return -1;
    }
    if (lhs->src > rhs->src) {
        return 1;
    }
    return 0;
}

static void report_shuffle_progress(
    shuffle_progress_cb_t progress_cb,
    int phase,
    long long completed,
    long long total,
    long long aux,
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
    progress_cb(phase, (double)percent / 100.0, completed, total, aux);
}

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

static int count_fasta_records(const unsigned char *data, size_t size, uint64_t *out_count,
                               shuffle_progress_cb_t progress_cb) {
    uint64_t count = 0;
    if (size == 0) {
        *out_count = 0;
        return 0;
    }

    const unsigned char *cur = data;
    const unsigned char *end = data + size;
    long long last_reported_percent = -1;

    if (data[0] == '>') {
        count++;
    }

    while (cur < end) {
        const void *found = memchr(cur, '>', (size_t)(end - cur));
        if (!found) {
            break;
        }
        const unsigned char *p = (const unsigned char *)found;
        if (p > data && p[-1] == '\n') {
            count++;
        }
        cur = p + 1;
        report_shuffle_progress(
            progress_cb,
            1,
            (long long)(cur - data),
            (long long)size,
            (long long)count,
            &last_reported_percent);
    }

    report_shuffle_progress(
        progress_cb,
        1,
        (long long)size,
        (long long)size,
        (long long)count,
        &last_reported_percent);
    *out_count = count;
    return 0;
}

static int fill_fasta_record_starts(const unsigned char *data, size_t size, uint64_t *starts,
                                    uint64_t count, shuffle_progress_cb_t progress_cb) {
    uint64_t idx = 0;
    if (size == 0) {
        return count == 0 ? 0 : -1;
    }

    const unsigned char *cur = data;
    const unsigned char *end = data + size;
    long long last_reported_percent = -1;

    if (data[0] == '>') {
        starts[idx++] = 0;
    }

    while (cur < end) {
        const void *found = memchr(cur, '>', (size_t)(end - cur));
        if (!found) {
            break;
        }
        const unsigned char *p = (const unsigned char *)found;
        if (p > data && p[-1] == '\n') {
            if (idx >= count) {
                return -1;
            }
            starts[idx++] = (uint64_t)(p - data);
        }
        cur = p + 1;
        report_shuffle_progress(
            progress_cb,
            2,
            (long long)(cur - data),
            (long long)size,
            (long long)idx,
            &last_reported_percent);
    }

    if (idx != count) {
        return -1;
    }

    starts[count] = (uint64_t)size;
    report_shuffle_progress(
        progress_cb,
        2,
        (long long)size,
        (long long)size,
        (long long)idx,
        &last_reported_percent);
    return 0;
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

static int dynbuf_init(DynBuf *buf, size_t initial_cap) {
    buf->len = 0;
    buf->cap = initial_cap > 0 ? initial_cap : 256;
    buf->data = (char *)malloc(buf->cap);
    return buf->data ? 0 : -1;
}

static void dynbuf_free(DynBuf *buf) {
    if (!buf) {
        return;
    }
    free(buf->data);
    buf->data = NULL;
    buf->len = 0;
    buf->cap = 0;
}

static int dynbuf_reserve(DynBuf *buf, size_t needed) {
    if (needed <= buf->cap) {
        return 0;
    }
    size_t new_cap = buf->cap > 0 ? buf->cap : 256;
    while (new_cap < needed) {
        new_cap *= 2;
    }
    char *new_data = (char *)realloc(buf->data, new_cap);
    if (!new_data) {
        return -1;
    }
    buf->data = new_data;
    buf->cap = new_cap;
    return 0;
}

static inline int is_whitespace(unsigned char c) {
    return c == '\n' || c == '\r' || c == ' ' || c == '\t' || c == '\v' || c == '\f';
}

static int dynbuf_set_header_from_line(DynBuf *buf, const char *line, size_t n) {
    // Strip all leading and trailing whitespace to match Python's str.strip().
    while (n > 0 && is_whitespace((unsigned char)line[n - 1])) {
        n--;
    }
    size_t start = 0;
    while (start < n && is_whitespace((unsigned char)line[start])) {
        start++;
    }
    size_t stripped_len = n - start;
    if (dynbuf_reserve(buf, stripped_len + 1) != 0) {
        return -1;
    }
    if (stripped_len > 0) {
        memcpy(buf->data, line + start, stripped_len);
    }
    buf->len = stripped_len;
    buf->data[stripped_len] = '\0';
    return 0;
}

static int dynbuf_append_seq_line(DynBuf *buf, const char *line, size_t n) {
    if (n == 0) {
        return 0;
    }
    size_t needed = buf->len + n + 1;
    if (dynbuf_reserve(buf, needed) != 0) {
        return -1;
    }
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)line[i];
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t' || c == '\v' || c == '\f') {
            continue;
        }
        buf->data[buf->len++] = (char)c;
    }
    buf->data[buf->len] = '\0';
    return 0;
}

static int write_record(FILE *out, const DynBuf *header, const DynBuf *seq) {
    if (fputc('>', out) == EOF) {
        return -1;
    }
    if (header->len > 0 && fwrite(header->data, 1, header->len, out) != header->len) {
        return -1;
    }
    if (fputc('\n', out) == EOF) {
        return -1;
    }
    if (seq->len > 0 && fwrite(seq->data, 1, seq->len, out) != seq->len) {
        return -1;
    }
    if (fputc('\n', out) == EOF) {
        return -1;
    }
    return 0;
}

static int process_filter_record(
    FILE *out,
    const DynBuf *header,
    const DynBuf *seq,
    int min_seq_len,
    int max_seq_len,
    long long seqs_num,
    long long skip_n,
    long long *skipped,
    long long *processed,
    long long *passed,
    int *stop_now,
    char *error_msg,
    size_t error_cap) {
    if (*skipped < skip_n) {
        (*skipped)++;
        return 0;
    }

    (*processed)++;

    if (seqs_num != -1 && *passed >= seqs_num) {
        *stop_now = 1;
        return 0;
    }

    if ((long long)seq->len >= (long long)min_seq_len &&
        (long long)seq->len <= (long long)max_seq_len) {
        if (write_record(out, header, seq) != 0) {
            set_error(error_msg, error_cap, "Failed writing filtered FASTA record: %s", strerror(errno));
            return -1;
        }
        (*passed)++;
    }

    return 0;
}

API_EXPORT int nanoplm_filter_fasta(
    const char *input_path,
    const char *output_path,
    int min_seq_len,
    int max_seq_len,
    long long seqs_num,
    long long skip_n,
    progress_cb_t progress_cb,
    long long *out_processed,
    long long *out_passed,
    char *error_msg,
    size_t error_cap) {
    if (!input_path || !output_path || !out_processed || !out_passed) {
        set_error(error_msg, error_cap, "Invalid null argument");
        return -1;
    }
    if (min_seq_len < 0 || max_seq_len < 0 || min_seq_len > max_seq_len) {
        set_error(error_msg, error_cap, "Invalid min/max sequence lengths");
        return -1;
    }
    if (seqs_num < -1 || seqs_num == 0) {
        set_error(error_msg, error_cap, "seqs_num must be -1 or positive");
        return -1;
    }
    if (skip_n < 0) {
        set_error(error_msg, error_cap, "skip_n must be non-negative");
        return -1;
    }

    int in_fd = open(input_path, O_RDONLY);
    if (in_fd < 0) {
        set_error(error_msg, error_cap, "Failed to open input FASTA %s: %s", input_path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(in_fd, &st) != 0) {
        close(in_fd);
        set_error(error_msg, error_cap, "Failed to stat input FASTA %s: %s", input_path, strerror(errno));
        return -1;
    }
    long long input_size = (long long)st.st_size;

    FILE *in = fdopen(in_fd, "rb");
    if (!in) {
        close(in_fd);
        set_error(error_msg, error_cap, "Failed to open input FASTA stream %s: %s", input_path, strerror(errno));
        return -1;
    }
    FILE *out = fopen(output_path, "wb");
    if (!out) {
        fclose(in);
        set_error(error_msg, error_cap, "Failed to open output FASTA %s: %s", output_path, strerror(errno));
        return -1;
    }

    setvbuf(in, NULL, _IOFBF, 4 * 1024 * 1024);
    setvbuf(out, NULL, _IOFBF, 4 * 1024 * 1024);

    DynBuf header;
    DynBuf seq;
    if (dynbuf_init(&header, 256) != 0 || dynbuf_init(&seq, 1024) != 0) {
        dynbuf_free(&header);
        dynbuf_free(&seq);
        fclose(in);
        fclose(out);
        set_error(error_msg, error_cap, "Out of memory");
        return -1;
    }

    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len = 0;
    int seen_header = 0;
    int stop_now = 0;
    int status = 0;
    long long bytes_read = 0;
    long long last_reported_percent = -1;

    long long skipped = 0;
    long long processed = 0;
    long long passed = 0;

    while ((line_len = getline(&line, &line_cap, in)) != -1) {
        bytes_read += (long long)line_len;
        if (line_len > 0 && line[0] == '>') {
            if (seen_header) {
                if (process_filter_record(
                        out, &header, &seq, min_seq_len, max_seq_len, seqs_num, skip_n, &skipped,
                        &processed, &passed, &stop_now, error_msg, error_cap) != 0) {
                    status = -1;
                    goto cleanup;
                }
                if (stop_now) {
                    break;
                }
            }
            if (dynbuf_set_header_from_line(&header, line + 1, (size_t)(line_len - 1)) != 0) {
                set_error(error_msg, error_cap, "Out of memory while reading FASTA header");
                status = -1;
                goto cleanup;
            }
            seq.len = 0;
            if (seq.data != NULL) {
                seq.data[0] = '\0';
            }
            seen_header = 1;
        } else if (seen_header) {
            if (dynbuf_append_seq_line(&seq, line, (size_t)line_len) != 0) {
                set_error(error_msg, error_cap, "Out of memory while reading FASTA sequence");
                status = -1;
                goto cleanup;
            }
        }
        report_progress(progress_cb, 1, bytes_read, input_size, &last_reported_percent);
    }

    if (status == 0 && seen_header && !stop_now) {
        if (process_filter_record(
                out, &header, &seq, min_seq_len, max_seq_len, seqs_num, skip_n, &skipped,
                &processed, &passed, &stop_now, error_msg, error_cap) != 0) {
            status = -1;
            goto cleanup;
        }
    }
    report_progress(progress_cb, 1, input_size, input_size, &last_reported_percent);

    if (ferror(in)) {
        set_error(error_msg, error_cap, "Error while reading input FASTA");
        status = -1;
        goto cleanup;
    }

cleanup:
    free(line);
    dynbuf_free(&header);
    dynbuf_free(&seq);

    if (fclose(in) != 0 && status == 0) {
        set_error(error_msg, error_cap, "Failed to close input FASTA: %s", strerror(errno));
        status = -1;
    }
    if (fclose(out) != 0 && status == 0) {
        set_error(error_msg, error_cap, "Failed to close output FASTA: %s", strerror(errno));
        status = -1;
    }

    if (status == 0) {
        *out_processed = processed;
        *out_passed = passed;
    }
    return status;
}

API_EXPORT int nanoplm_split_fasta(
    const char *input_path,
    const char *train_path,
    const char *val_path,
    double val_ratio,
    progress_cb_t progress_cb,
    long long *out_train_size,
    long long *out_val_size,
    char *error_msg,
    size_t error_cap) {
    if (!input_path || !train_path || !val_path || !out_train_size || !out_val_size) {
        set_error(error_msg, error_cap, "Invalid null argument");
        return -1;
    }
    if (!(val_ratio >= 0.0 && val_ratio <= 1.0)) {
        set_error(error_msg, error_cap, "val_ratio must be in [0, 1]");
        return -1;
    }

    FILE *in = fopen(input_path, "rb");
    if (!in) {
        set_error(error_msg, error_cap, "Failed to open input FASTA %s: %s", input_path, strerror(errno));
        return -1;
    }
    setvbuf(in, NULL, _IOFBF, 4 * 1024 * 1024);
    struct stat st;
    if (fstat(fileno(in), &st) != 0) {
        fclose(in);
        set_error(error_msg, error_cap, "Failed to stat input FASTA %s: %s", input_path, strerror(errno));
        return -1;
    }
    long long input_size = (long long)st.st_size;

    long long total = 0;
    char *line = NULL;
    size_t line_cap = 0;
    ssize_t line_len = 0;
    long long bytes_read = 0;
    long long last_reported_percent = -1;
    while ((line_len = getline(&line, &line_cap, in)) != -1) {
        bytes_read += (long long)line_len;
        if (line_len > 0 && line[0] == '>') {
            total++;
        }
        report_progress(progress_cb, 1, bytes_read, input_size, &last_reported_percent);
    }
    if (ferror(in)) {
        free(line);
        fclose(in);
        set_error(error_msg, error_cap, "Error while counting FASTA records");
        return -1;
    }

    long long val_size = (long long)((double)total * val_ratio);
    long long train_size = total - val_size;

    if (fseek(in, 0, SEEK_SET) != 0) {
        free(line);
        fclose(in);
        set_error(error_msg, error_cap, "Failed to rewind input FASTA");
        return -1;
    }

    FILE *train = fopen(train_path, "wb");
    if (!train) {
        free(line);
        fclose(in);
        set_error(error_msg, error_cap, "Failed to open train FASTA %s: %s", train_path, strerror(errno));
        return -1;
    }
    FILE *val = fopen(val_path, "wb");
    if (!val) {
        free(line);
        fclose(in);
        fclose(train);
        set_error(error_msg, error_cap, "Failed to open val FASTA %s: %s", val_path, strerror(errno));
        return -1;
    }

    setvbuf(train, NULL, _IOFBF, 4 * 1024 * 1024);
    setvbuf(val, NULL, _IOFBF, 4 * 1024 * 1024);

    long long rec_idx = -1;
    FILE *cur_out = NULL;
    int status = 0;
    bytes_read = 0;
    last_reported_percent = -1;
    while ((line_len = getline(&line, &line_cap, in)) != -1) {
        bytes_read += (long long)line_len;
        if (line_len > 0 && line[0] == '>') {
            rec_idx++;
            cur_out = (rec_idx < train_size) ? train : val;
        }
        if (rec_idx >= 0 && cur_out != NULL) {
            if (fwrite(line, 1, (size_t)line_len, cur_out) != (size_t)line_len) {
                set_error(error_msg, error_cap, "Failed while writing split FASTA: %s", strerror(errno));
                status = -1;
                break;
            }
        }
        report_progress(progress_cb, 2, bytes_read, input_size, &last_reported_percent);
    }

    if (status == 0 && ferror(in)) {
        set_error(error_msg, error_cap, "Error while splitting FASTA");
        status = -1;
    }

    free(line);

    if (fclose(in) != 0 && status == 0) {
        set_error(error_msg, error_cap, "Failed to close input FASTA: %s", strerror(errno));
        status = -1;
    }
    if (fclose(train) != 0 && status == 0) {
        set_error(error_msg, error_cap, "Failed to close train FASTA: %s", strerror(errno));
        status = -1;
    }
    if (fclose(val) != 0 && status == 0) {
        set_error(error_msg, error_cap, "Failed to close val FASTA: %s", strerror(errno));
        status = -1;
    }

    if (status == 0) {
        report_progress(progress_cb, 2, input_size, input_size, &last_reported_percent);
        *out_train_size = train_size;
        *out_val_size = val_size;
    }
    return status;
}

API_EXPORT int nanoplm_shuffle_fasta(
    const char *input_path,
    const char *output_path,
    unsigned long long seed,
    int num_threads,
    long long batch_records,
    shuffle_progress_cb_t progress_cb,
    long long *out_num_records,
    char *error_msg,
    size_t error_cap) {
    if (!input_path || !output_path || !out_num_records) {
        set_error(error_msg, error_cap, "Invalid null argument");
        return -1;
    }
    if (num_threads < 1) {
        num_threads = 1;
    }
    if (batch_records < 1) {
        batch_records = 262144;
    }

    int in_fd = open(input_path, O_RDONLY);
    if (in_fd < 0) {
        set_error(error_msg, error_cap, "Failed to open input FASTA %s: %s", input_path, strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(in_fd, &st) != 0) {
        close(in_fd);
        set_error(error_msg, error_cap, "Failed to stat input FASTA %s: %s", input_path, strerror(errno));
        return -1;
    }
    if (st.st_size <= 0) {
        close(in_fd);
        set_error(error_msg, error_cap, "Input FASTA is empty");
        return -1;
    }
    size_t file_size = (size_t)st.st_size;

    unsigned char *in_map = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, in_fd, 0);
    if (in_map == MAP_FAILED) {
        close(in_fd);
        set_error(error_msg, error_cap, "Failed to mmap input FASTA: %s", strerror(errno));
        return -1;
    }

    uint64_t num_records = 0;
    if (count_fasta_records(in_map, file_size, &num_records, progress_cb) != 0 || num_records == 0) {
        munmap(in_map, file_size);
        close(in_fd);
        set_error(error_msg, error_cap, "No FASTA records found in input");
        return -1;
    }
    if (num_records > UINT32_MAX) {
        munmap(in_map, file_size);
        close(in_fd);
        set_error(error_msg, error_cap, "FASTA contains too many records for in-memory shuffle");
        return -1;
    }

    uint64_t *starts = (uint64_t *)malloc((size_t)(num_records + 1) * sizeof(uint64_t));
    uint32_t *perm = (uint32_t *)malloc((size_t)num_records * sizeof(uint32_t));
    CopyTask *tasks = (CopyTask *)malloc((size_t)batch_records * sizeof(CopyTask));
    if (!starts || !perm || !tasks) {
        free(starts);
        free(perm);
        free(tasks);
        munmap(in_map, file_size);
        close(in_fd);
        set_error(error_msg, error_cap, "Out of memory while preparing FASTA shuffle");
        return -1;
    }

    if (fill_fasta_record_starts(in_map, file_size, starts, num_records, progress_cb) != 0) {
        free(starts);
        free(perm);
        free(tasks);
        munmap(in_map, file_size);
        close(in_fd);
        set_error(error_msg, error_cap, "Failed to index FASTA record starts");
        return -1;
    }

    for (uint64_t i = 0; i < num_records; i++) {
        perm[i] = (uint32_t)i;
    }

    uint64_t rng_state = (uint64_t)seed;
    if (rng_state == 0) {
        rng_state = 1;
    }
    for (uint64_t i = num_records - 1; i > 0; i--) {
        uint64_t j = splitmix64_next(&rng_state) % (i + 1);
        uint32_t tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    int out_fd = open(output_path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (out_fd < 0) {
        free(starts);
        free(perm);
        free(tasks);
        munmap(in_map, file_size);
        close(in_fd);
        set_error(error_msg, error_cap, "Failed to open output FASTA %s: %s", output_path, strerror(errno));
        return -1;
    }
    if (ftruncate(out_fd, (off_t)file_size) != 0) {
        free(starts);
        free(perm);
        free(tasks);
        munmap(in_map, file_size);
        close(in_fd);
        close(out_fd);
        set_error(error_msg, error_cap, "Failed to size output FASTA %s: %s", output_path, strerror(errno));
        return -1;
    }

    unsigned char *out_map = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    if (out_map == MAP_FAILED) {
        free(starts);
        free(perm);
        free(tasks);
        munmap(in_map, file_size);
        close(in_fd);
        close(out_fd);
        set_error(error_msg, error_cap, "Failed to mmap output FASTA: %s", strerror(errno));
        return -1;
    }

#if defined(_OPENMP)
    omp_set_num_threads(num_threads);
#endif

    uint64_t dst_offset = 0;
    long long last_reported_percent = -1;
    for (uint64_t batch_start = 0; batch_start < num_records; batch_start += (uint64_t)batch_records) {
        uint64_t batch_n = num_records - batch_start;
        if (batch_n > (uint64_t)batch_records) {
            batch_n = (uint64_t)batch_records;
        }

        for (uint64_t j = 0; j < batch_n; j++) {
            uint32_t rec = perm[batch_start + j];
            uint64_t src = starts[rec];
            uint64_t len = starts[(uint64_t)rec + 1] - src;
            tasks[j].src = src;
            tasks[j].dst = dst_offset;
            tasks[j].len = len;
            dst_offset += len;
        }

        qsort(tasks, (size_t)batch_n, sizeof(CopyTask), compare_copy_task_by_src);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (uint64_t j = 0; j < batch_n; j++) {
            memcpy(out_map + tasks[j].dst, in_map + tasks[j].src, (size_t)tasks[j].len);
        }

        report_shuffle_progress(
            progress_cb,
            3,
            (long long)(batch_start + batch_n),
            (long long)num_records,
            (long long)(batch_start + batch_n),
            &last_reported_percent);
    }

    munmap(out_map, file_size);
    free(starts);
    free(perm);
    free(tasks);
    munmap(in_map, file_size);
    close(in_fd);
    if (close(out_fd) != 0) {
        set_error(error_msg, error_cap, "Failed to close output FASTA: %s", strerror(errno));
        return -1;
    }

    *out_num_records = (long long)num_records;
    return 0;
}
