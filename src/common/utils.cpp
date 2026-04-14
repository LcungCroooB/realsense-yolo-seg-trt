#include "utils.hpp"

#include <sys/stat.h> // For mkdirs
#include <unistd.h>   // For access
#include <chrono>
#include <ctime>
#include <stdarg.h>
#include <dirent.h>
#include <errno.h>
#include <sys/types.h>
#include <stack>
#include <fstream>

namespace utils
{
    std::string format(const char *fmt, ...)
    {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }

    std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
    {
        const int h_i = static_cast<int>(h * 6);
        const float f = h * 6 - h_i;
        const float p = v * (1 - s);
        const float q = v * (1 - f * s);
        const float t = v * (1 - (1 - f) * s);
        float r, g, b;
        switch (h_i)
        {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
        default:
            r = 1;
            g = 1;
            b = 1;
            break;
        }
        return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
    }

    std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
    {
        float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
        ;
        float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
        return hsv2bgr(h_plane, s_plane, 1);
    }

    namespace fs
    {
        namespace
        {
            bool is_directory(const std::string &path)
            {
                struct stat st;
                if (::stat(path.c_str(), &st) != 0)
                    return false;
                return S_ISDIR(st.st_mode);
            }
        }

        bool exists(const std::string &files)
        {
            return access(files.c_str(), R_OK) == 0; // linux
        }

        bool mkdir(const std::string &path)
        {
            return ::mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // linux
        }

        bool mkdirs(const std::string &path)
        {
            if (path.empty())
                return false;

            if (is_directory(path))
                return true;

            std::string current;
            if (path[0] == '/')
                current = "/";

            size_t begin = (path[0] == '/') ? 1 : 0;
            while (begin <= path.size())
            {
                size_t end = path.find('/', begin);
                std::string part = path.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
                if (!part.empty())
                {
                    if (!current.empty() && current.back() != '/')
                        current += "/";
                    current += part;

                    if (!is_directory(current) && ::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
                        return false;
                }

                if (end == std::string::npos)
                    break;
                begin = end + 1;
            }
            return is_directory(path);
        }

        FILE *fopen_mkdirs(const std::string &path, const std::string &mode)
        {
            FILE *f = fopen(path.c_str(), mode.c_str());
            if (f)
                return f;

            int p = path.rfind('/');
            if (p == -1)
                return nullptr;

            std::string directory = path.substr(0, p);
            if (!mkdirs(directory))
                return nullptr;
            return fopen(path.c_str(), mode.c_str());
        }

        std::string file_name(const std::string &path, bool include_suffix)
        {
            if (path.empty())
                return "";

            int p = path.rfind('/');
            p += 1;

            if (include_suffix)
                return path.substr(p);

            int u = path.rfind(".");
            if (u == -1)
                return path.substr(p);

            if (u < p)
                u = path.size();
            return path.substr(p, u - p);
        }

        std::vector<unsigned char> load_file(const std::string &file)
        {
            std::ifstream in(file, std::ios::in | std::ios::binary);
            if (!in.is_open())
            {
                // INFOE("Failed to load trtmodel from %d", file);
                return {};
            }

            in.seekg(0, std::ios::end);
            size_t length = in.tellg();
            std::vector<unsigned char> data;
            if (length > 0)
            {
                in.seekg(0, std::ios::beg);
                data.resize(length);
                in.read((char *)&data[0], length);
            }
            in.close();
            return data;
        }

        std::vector<std::string> find_files(const std::string &directory, const std::string &filter, bool findDirectory, bool includeSubDirectory)
        {
            std::string real_path = directory;
            if (real_path.empty())
                real_path = "./";

            char backchar = real_path.back();
            if (backchar not_eq '/')
                real_path += "/";

            struct dirent *fileinfo;
            DIR *handle;
            std::stack<std::string> ps;
            std::vector<std::string> out;
            ps.push(real_path);

            while (!ps.empty())
            {
                std::string search_path = ps.top();
                ps.pop();

                handle = opendir(search_path.c_str());
                if (handle not_eq 0)
                {
                    while (fileinfo = readdir(handle))
                    {
                        struct stat file_stat;
                        if (strcmp(fileinfo->d_name, ".") == 0 or strcmp(fileinfo->d_name, "..") == 0)
                            continue;

                        if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
                            continue;

                        if (!findDirectory and !S_ISDIR(file_stat.st_mode) or
                            findDirectory and S_ISDIR(file_stat.st_mode))
                        {
                            if (pattern_match(fileinfo->d_name, filter.c_str()))
                                out.push_back(search_path + fileinfo->d_name);
                        }

                        if (includeSubDirectory and S_ISDIR(file_stat.st_mode))
                            ps.push(search_path + fileinfo->d_name + "/");
                    }
                    closedir(handle);
                }
            }
            return out;
        }

        bool alphabet_equal(char a, char b, bool ignore_case)
        {
            if (ignore_case)
            {
                a = a > 'a' and a < 'z' ? a - 'a' + 'A' : a;
                b = b > 'a' and b < 'z' ? b - 'a' + 'A' : b;
            }
            return a == b;
        }

        static bool pattern_match_body(const char *str, const char *matcher, bool ignore_case)
        {
            if (!matcher or !*matcher or !str or !*str)
                return false;

            const char *ptr_matcher = matcher;
            while (*str)
            {
                if (*ptr_matcher == '?')
                {
                    ptr_matcher++;
                }
                else if (*ptr_matcher == '*')
                {
                    if (*(ptr_matcher + 1))
                    {
                        if (pattern_match_body(str, ptr_matcher + 1, ignore_case))
                            return true;
                    }
                    else
                    {
                        return true;
                    }
                }
                else if (!alphabet_equal(*ptr_matcher, *str, ignore_case))
                {
                    return false;
                }
                else
                {
                    if (*ptr_matcher)
                        ptr_matcher++;
                    else
                        return false;
                }
                str++;
            }

            while (*ptr_matcher)
            {
                if (*ptr_matcher not_eq '*')
                    return false;
                ptr_matcher++;
            }
            return true;
        }

        bool pattern_match(const char *str, const char *matcher, bool ignore_case)
        {
            //   abcdefg.pnga          *.png      > false
            //   abcdefg.png           *.png      > true
            //   abcdefg.png          a?cdefg.png > true
            if (!matcher or !*matcher or !str or !*str)
            {
                return false;
            }

            char filter[500];
            strcpy(filter, matcher);

            std::vector<const char *> arr;
            char *ptr_str = filter;
            char *ptr_prev_str = ptr_str;
            while (*ptr_str)
            {
                if (*ptr_str == ';')
                {
                    *ptr_str = 0;
                    arr.push_back(ptr_prev_str);
                    ptr_prev_str = ptr_str + 1;
                }
                ptr_str++;
            }

            if (*ptr_prev_str)
                arr.push_back(ptr_prev_str);

            for (int i = 0; i < arr.size(); ++i)
            {
                if (pattern_match_body(str, arr[i], ignore_case))
                    return true;
            }
            return false;
        }

        bool rmtree(const std::string &directory, bool ignore_fail)
        {
            if (directory.empty())
                return false;
            auto files = find_files(directory, "*", false);
            auto dirs = find_files(directory, "*", true);
            bool success = true;
            for (int i = 0; i < files.size(); ++i)
            {
                if (::remove(files[i].c_str()) != 0)
                {
                    success = false;
                    if (!ignore_fail)
                        return false;
                }
            }

            dirs.insert(dirs.begin(), directory);
            for (int i = (int)dirs.size() - 1; i >= 0; --i)
            {
                if (::rmdir(dirs[i].c_str()) != 0)
                {
                    success = false;
                    if (!ignore_fail)
                        return false;
                }
            }
            return success;
        }
    }

    namespace path
    {
        bool is_absolute(const std::string &path)
        {
            return !path.empty() && path[0] == '/';
        }

        std::string join(const std::string &base, const std::string &rel)
        {
            if (base.empty())
                return rel;
            if (base.back() == '/')
                return base + rel;
            return base + "/" + rel;
        }

        std::string directory_name(const std::string &path)
        {
            const size_t pos = path.find_last_of('/');
            if (pos == std::string::npos)
                return std::string();
            if (pos == 0)
                return "/";
            return path.substr(0, pos);
        }

        std::string parent(const std::string &path)
        {
            return directory_name(path);
        }

        std::string cwd()
        {
            char buffer[PATH_MAX] = {0};
            if (getcwd(buffer, sizeof(buffer)) == nullptr)
                return std::string();
            return std::string(buffer);
        }

        std::string executable_dir()
        {
            char buffer[PATH_MAX] = {0};
            const ssize_t n = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
            if (n <= 0)
                return std::string();
            buffer[n] = '\0';
            return directory_name(std::string(buffer));
        }

        std::vector<std::string> candidate_prefixes()
        {
            std::vector<std::string> prefixes;
            const std::string cwd_path = utils::path::cwd();
            const std::string exe_dir = utils::path::executable_dir();

            prefixes.emplace_back(std::string());
            if (!cwd_path.empty())
            {
                prefixes.emplace_back(cwd_path);
                prefixes.emplace_back(utils::path::parent(cwd_path));
                prefixes.emplace_back(utils::path::parent(utils::path::parent(cwd_path)));
            }
            if (!exe_dir.empty())
            {
                prefixes.emplace_back(exe_dir);
                prefixes.emplace_back(utils::path::parent(exe_dir));
                prefixes.emplace_back(utils::path::parent(utils::path::parent(exe_dir)));
            }
            return prefixes;
        }

        bool resolve_input_path(const std::string &path,std::string &resolved,std::string &prefix)
        {
            if (utils::path::is_absolute(path))
            {
                if (!utils::fs::exists(path))
                    return false;
                resolved = path;
                prefix.clear();
                return true;
            }

            const std::vector<std::string> prefixes = candidate_prefixes();
            for (const std::string &p : prefixes)
            {
                const std::string candidate = utils::path::join(p, path);
                if (utils::fs::exists(candidate))
                {
                    resolved = candidate;
                    prefix = p;
                    return true;
                }
            }
            return false;
        }

        std::string resolve_output_path(const std::string &path,const std::string &preferred_prefix)
        {
            if (utils::path::is_absolute(path))
                return path;
            if (!preferred_prefix.empty())
                return utils::path::join(preferred_prefix, path);
            return path;
        }
    }

    namespace time
    {
        std::string timenow_format()
        {
            char time_string[20];
            std::time_t timestamp;
            std::time(&timestamp);
            tm &t = *(tm *)std::localtime(&timestamp);

            sprintf(time_string, "%04d-%02d-%02d_%02d-%02d-%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
            return time_string;
        }

        long long timestamp_millisecond() // Calculation time
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        }

        double timestamp_second()
        {
            return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
        }
    }
}