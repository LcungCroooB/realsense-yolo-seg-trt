#pragma once

#include <yaml-cpp/yaml.h>
#include <string>

// ---------------------------------------------------------------------------
// Lightweight YAML::Node accessors shared by all config loaders in src/apps.
// These are intentionally kept header-only so they stay in the apps target
// (yaml-cpp is not linked into PerceptFusion).
// ---------------------------------------------------------------------------
namespace yaml_node
{
    inline std::string read_str(const YAML::Node &node, const std::string &key,
                                const std::string &def = "")
    {
        if (!node || !node[key]) return def;
        return node[key].as<std::string>(def);
    }

    inline int read_int(const YAML::Node &node, const std::string &key, int def = 0)
    {
        if (!node || !node[key]) return def;
        return node[key].as<int>(def);
    }

    inline bool read_bool(const YAML::Node &node, const std::string &key, bool def = false)
    {
        if (!node || !node[key]) return def;
        return node[key].as<bool>(def);
    }

    inline float read_float(const YAML::Node &node, const std::string &key, float def = 0.0f)
    {
        if (!node || !node[key]) return def;
        return node[key].as<float>(def);
    }

    inline std::vector<int> read_int_list(const YAML::Node &node, const std::string &key)
    {
        if (!node || !node[key] || !node[key].IsSequence())
            return {};
        std::vector<int> values;
        values.reserve(node[key].size());
        for (const YAML::Node &item : node[key])
        {
            if (!item)
                continue;
            values.emplace_back(item.as<int>());
        }
        return values;
    }
}
