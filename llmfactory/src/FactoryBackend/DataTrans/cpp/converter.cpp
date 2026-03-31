#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fstream>
#include <charconv>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// Global tables for fast escaping
static const char* ESC_SEQ[256] = { nullptr };
static uint8_t ESC_LEN[256] = { 0 };

struct TableInit {
    char buffer[32][7];

    TableInit() {
        ESC_SEQ['\b'] = "\\b"; ESC_LEN['\b'] = 2;
        ESC_SEQ['\f'] = "\\f"; ESC_LEN['\f'] = 2;
        ESC_SEQ['\n'] = "\\n"; ESC_LEN['\n'] = 2;
        ESC_SEQ['\r'] = "\\r"; ESC_LEN['\r'] = 2;
        ESC_SEQ['\t'] = "\\t"; ESC_LEN['\t'] = 2;
        ESC_SEQ['\"'] = "\\\""; ESC_LEN['\"'] = 2;
        ESC_SEQ['\\'] = "\\\\"; ESC_LEN['\\'] = 2;

        for (int i = 0; i < 32; ++i) {
            if (ESC_SEQ[i]) continue;
            snprintf(buffer[i], 7, "\\u%04x", i);
            ESC_SEQ[i] = buffer[i];
            ESC_LEN[i] = 6;
        }
    }
} _init_tab;

// Optimized escape using C++17 std::string::data()
static void fast_escape(const char* s, size_t len, std::string& out) noexcept {
    if (len == 0) return;

    size_t extra_len = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (ESC_SEQ[c]) {
            extra_len += ESC_LEN[c] - 1;
        }
    }

    size_t start_pos = out.size();
    out.resize(start_pos + len + extra_len);

    // C++17: data() returns non-const pointer
    char* dst = out.data() + start_pos;

    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (ESC_SEQ[c]) {
            std::memcpy(dst, ESC_SEQ[c], ESC_LEN[c]);
            dst += ESC_LEN[c];
        } else {
            *dst++ = s[i];
        }
    }
}

enum class ColType { INT64, DOUBLE, STRING };

struct Column {
    const char* data;
    int64_t stride;
    ColType type;
    py::array _ref;
};

// Updated to use std::string_view
bool format_val(const Column& col, int row, char* buf, std::string_view s_view,
                const char*& p, size_t& l) noexcept {
    if (col.type == ColType::STRING) {
        if (s_view.empty()) return false;
        p = s_view.data();
        l = s_view.length();
        return true;
    }

    auto* ptr = col.data + row * col.stride;
    std::to_chars_result res;

    if (col.type == ColType::INT64) {
        res = std::to_chars(buf, buf + 64, *reinterpret_cast<const int64_t*>(ptr));
    } else {
        double v = *reinterpret_cast<const double*>(ptr);
        if (std::isnan(v)) return false;
        res = std::to_chars(buf, buf + 64, v, std::chars_format::general);
    }

    if (res.ec != std::errc()) return false;
    p = buf; l = res.ptr - buf; return true;
}

void process_chunk_to_json(py::dict columns_data, py::dict mapping,
                           std::string out_path, bool is_first, size_t num_rows) {
    std::vector<Column> cols;
    std::unordered_map<std::string, int> name_map;

    // 1. Prepare Column Metadata
    for (auto const& [key_obj, val_obj] : columns_data) {
        auto arr = val_obj.cast<py::array>();
        ColType t = arr.dtype().is(py::dtype::of<int64_t>()) ? ColType::INT64 :
                    arr.dtype().is(py::dtype::of<double>())  ? ColType::DOUBLE : ColType::STRING;
        name_map[py::str(key_obj)] = cols.size();
        cols.push_back({(const char*)arr.data(), arr.strides(0), t, arr});
    }

    // 2. Prepare Mapping
    std::vector<std::string> col_headers;
    std::vector<std::vector<int>> task_map;

    for (auto const& [key_obj, val_obj] : mapping) {
        std::string key = py::str(key_obj);
        std::vector<int> indices;
        try {
            for (auto s : val_obj.cast<std::vector<std::string>>())
                if (name_map.count(s)) indices.push_back(name_map[s]);
        } catch (...) {
            std::string s = val_obj.cast<std::string>();
            if (name_map.count(s)) indices.push_back(name_map[s]);
        }
        task_map.push_back(indices);
        col_headers.push_back((col_headers.empty() ? "\"" : ", \"") + key + "\": ");
    }

    const size_t num_cols = cols.size();

    // 3. Pre-extract Strings (Zero-Copy)
    std::vector<std::string_view> s_mat(num_rows * num_cols);
    for (size_t i = 0; i < num_rows; ++i) {
        auto* row_base = &s_mat[i * num_cols];
        for (size_t c = 0; c < num_cols; ++c) {
            if (cols[c].type != ColType::STRING) continue;
            auto* obj = *reinterpret_cast<PyObject**>(const_cast<char*>(cols[c].data + i * cols[c].stride));
            if (obj && obj != Py_None) {
                Py_ssize_t len;
                const char* s = PyUnicode_AsUTF8AndSize(obj, &len);
                if (s) row_base[c] = std::string_view(s, len);
            }
        }
    }

    // 4. Parallel Build & Sequential Merge (Optimized)
    {
        py::gil_scoped_release release;

        // A. Static Storage: Reuse memory across chunks to avoid allocation
        static std::vector<std::string> buffers;
        int num_threads = omp_get_max_threads();
        if (buffers.size() != (size_t)num_threads) buffers.resize(num_threads);

        // B. Precise Pre-allocation
        size_t rows_per_thread = (num_rows + num_threads - 1) / num_threads;
        size_t target_cap = rows_per_thread * 2048; // 2KB per row estimate

        // C. Open File with Binary Mode
        std::ofstream out(out_path, std::ios::binary | std::ios::app);
        if (!out) throw std::runtime_error("Failed to open output file");

        if (is_first) {
            out.write("[\n", 2);
        }

        // D. Parallel Processing
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& my_buf = buffers[tid];

            // Clear content but keep capacity (Optimization)
            my_buf.clear();
            if (my_buf.capacity() < target_cap) my_buf.reserve(target_cap);

            char n_buf[64]; // Stack buffer for number conversion

            // E. Inlined Logic (Removed Lambda overhead)
            #pragma omp for schedule(static) nowait
            for (int i = 0; i < (int)num_rows; ++i) {
                // Row Start
                if (is_first && i == 0) {
                    my_buf += "  {";
                } else {
                    my_buf += ",\n  {";
                }

                // Columns
                const size_t row_offset = i * num_cols;
                for (size_t j = 0; j < col_headers.size(); ++j) {
                    // Append Header
                    my_buf += col_headers[j];

                    bool has_v = false;
                    for (int idx : task_map[j]) {
                        const char* p; size_t l;
                        if (format_val(cols[idx], i, n_buf, s_mat[row_offset + idx], p, l)) {
                            if (!has_v) { my_buf += "\""; has_v = true; }
                            else { my_buf += "\\n"; }

                            if (cols[idx].type == ColType::STRING)
                                fast_escape(p, l, my_buf);
                            else
                                my_buf.append(p, l);
                        }
                    }
                    my_buf += has_v ? "\"" : "null";
                }
                my_buf += "}";
            }
        }

        // F. Sequential Merge with Binary Write
        for (auto& buf : buffers) {
            if (!buf.empty()) {
                out.write(buf.data(), buf.size());
            }
        }
    }
}

PYBIND11_MODULE(fast_converter, m) {
    m.def("process_chunk_to_json", &process_chunk_to_json);
}
