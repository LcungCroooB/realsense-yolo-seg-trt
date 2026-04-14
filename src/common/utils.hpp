#pragma once

#include <string>
#include <vector>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <tuple>

namespace utils
{
    std::string format(const char *fmt, ...);
    inline int upbound(int n, int align = 32) { return (n + align - 1) / align * align; }
    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v);
    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id);
}

namespace utils::fs
{
    bool exists(const std::string &files);
    bool mkdir(const std::string &path);
    bool mkdirs(const std::string &path);
    FILE *fopen_mkdirs(const std::string &path, const std::string &mode);
    std::string file_name(const std::string &path, bool include_suffix);
    std::vector<unsigned char> load_file(const std::string &file);
    std::vector<std::string> find_files(const std::string &directory, const std::string &filter,
                                        bool findDirectory = false, bool includeSubDirectory = false);
    bool pattern_match(const char *str, const char *matcher, bool ignore_case = true);
    bool rmtree(const std::string &directory, bool ignore_fail = false);
}

namespace utils::path
{
    bool is_absolute(const std::string &path);
    std::string join(const std::string &base, const std::string &rel);
    std::string directory_name(const std::string &path);
    std::string parent(const std::string &path);
    std::string cwd();
    std::string executable_dir();

    bool resolve_input_path(const std::string &path, std::string &resolved, std::string &prefix);
    std::string resolve_output_path(const std::string &path, const std::string &preferred_prefix);
}

namespace utils::time
{
    long long timestamp_millisecond();

}