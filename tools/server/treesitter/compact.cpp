#include "compact.h"

#include <tree_sitter/api.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <regex>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

// Language grammar declarations (defined in the grammar .c files)
extern "C" const TSLanguage *tree_sitter_python();
extern "C" const TSLanguage *tree_sitter_javascript();
extern "C" const TSLanguage *tree_sitter_typescript();
extern "C" const TSLanguage *tree_sitter_rust();
extern "C" const TSLanguage *tree_sitter_go();
extern "C" const TSLanguage *tree_sitter_c();
extern "C" const TSLanguage *tree_sitter_cpp();
extern "C" const TSLanguage *tree_sitter_java();
extern "C" const TSLanguage *tree_sitter_bash();

namespace compact {

//------------------------------------------------------------------------------
// Language registry
//------------------------------------------------------------------------------

struct BodyRule {
    const char *body_type;    // tree-sitter node type for the body
    const char *replacement;  // what to replace the body with (e.g. "{ /* ... */ }")
};

struct LangEntry {
    const char *name;
    std::vector<const char *> aliases;
    const TSLanguage *(*parser)();
    std::vector<const char *> def_types; // node types that are definitions with bodies
    BodyRule body;
};

static const LangEntry s_languages[] = {
    {
        "python",
        {"py", "python3", "python2"},
        tree_sitter_python,
        {"function_definition", "class_definition", "decorated_definition"},
        {"block", ": ..."}
    },
    {
        "javascript",
        {"js", "javascript", "node", "ecmascript"},
        tree_sitter_javascript,
        {"function_declaration", "method_definition", "arrow_function", "class_declaration"},
        {"statement_block", "{ /* ... */ }"}
    },
    {
        "typescript",
        {"ts", "typescript", "tsx"},
        tree_sitter_typescript,
        {"function_declaration", "method_definition", "arrow_function", "class_declaration"},
        {"statement_block", "{ /* ... */ }"}
    },
    {
        "rust",
        {"rs", "rust"},
        tree_sitter_rust,
        {"function_item", "struct_item", "impl_item", "trait_item", "enum_item", "mod_item"},
        {"block", "{ /* ... */ }"}
    },
    {
        "go",
        {"go", "golang"},
        tree_sitter_go,
        {"function_declaration", "method_declaration"},
        {"block", "{ /* ... */ }"}
    },
    {
        "c",
        {"c", "h"},
        tree_sitter_c,
        {"function_definition"},
        {"compound_statement", "{ /* ... */ }"}
    },
    {
        "cpp",
        {"cpp", "c++", "cc", "cxx", "hpp", "hxx", "c++", "cxx"},
        tree_sitter_cpp,
        {"function_definition", "class_specifier", "struct_specifier"},
        {"compound_statement", "{ /* ... */ }"}
    },
    {
        "java",
        {"java"},
        tree_sitter_java,
        {"method_declaration", "class_declaration"},
        {"block", "{ /* ... */ }"}
    },
    {
        "bash",
        {"bash", "sh", "shell", "zsh"},
        tree_sitter_bash,
        {"function_definition"},
        {"compound_statement", "{ /* ... */ }"}
    },
};

static constexpr size_t s_language_count = sizeof(s_languages) / sizeof(s_languages[0]);

//------------------------------------------------------------------------------
// Language lookup
//------------------------------------------------------------------------------

static const LangEntry *find_lang(const std::string & name) {
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) {
        lower.push_back(std::tolower((unsigned char)c));
    }
    for (size_t i = 0; i < s_language_count; i++) {
        if (lower == s_languages[i].name) return &s_languages[i];
        for (auto alias : s_languages[i].aliases) {
            if (lower == alias) return &s_languages[i];
        }
    }
    return nullptr;
}

//------------------------------------------------------------------------------
// Body range collector (recursive tree walk)
//------------------------------------------------------------------------------

static void collect_body_ranges(
    TSNode node,
    const LangEntry *lang,
    std::vector<std::pair<uint32_t, uint32_t>> & ranges,
    std::vector<std::pair<uint32_t, uint32_t>> & class_ranges // ranges to skip (class bodies contain methods)
) {
    uint32_t n_children = ts_node_child_count(node);

    // Check if this node is a definition type for this language
    const char *type = ts_node_type(node);
    bool is_def = false;
    for (auto dt : lang->def_types) {
        if (strcmp(type, dt) == 0) { is_def = true; break; }
    }

    // For class-like definitions, mark their full range so inner method bodies aren't double-stripped
    bool is_class = (strcmp(type, "class_definition") == 0 ||
                     strcmp(type, "class_specifier") == 0 ||
                     strcmp(type, "struct_specifier") == 0 ||
                     strcmp(type, "class_declaration") == 0 ||
                     strcmp(type, "impl_item") == 0);

    if (is_def) {
        // Find the body child node
        for (uint32_t i = 0; i < n_children; i++) {
            TSNode child = ts_node_child(node, i);
            if (strcmp(ts_node_type(child), lang->body.body_type) == 0) {
                uint32_t start = ts_node_start_byte(child);
                uint32_t end   = ts_node_end_byte(child);
                // Check if this body is inside a class range we already plan to handle
                bool skip = false;
                for (auto & cr : class_ranges) {
                    if (start >= cr.first && end <= cr.second) {
                        skip = true;
                        break;
                    }
                }
                if (!skip) {
                    ranges.push_back({start, end});
                }
                break;
            }
        }
    }

    // If this is a class/impl, recurse into children but the body child of THIS node
    // is the class body, not individual methods. The method definitions inside will be
    // caught by their own definition types above.
    for (uint32_t i = 0; i < n_children; i++) {
        collect_body_ranges(ts_node_child(node, i), lang, ranges, class_ranges);
    }
}

//------------------------------------------------------------------------------
// Tree-sitter based code block compaction
//------------------------------------------------------------------------------

static std::string compact_with_treesitter(const std::string & code, const LangEntry *lang) {
    TSParser *parser = ts_parser_new();
    if (!parser) return code;

    ts_parser_set_language(parser, lang->parser());

    TSTree *tree = ts_parser_parse_string(parser, nullptr, code.data(), code.size());
    if (!tree) {
        ts_parser_delete(parser);
        return code;
    }

    TSNode root = ts_tree_root_node(tree);

    // Collect body ranges to strip (top-level only to avoid nested double-stripping)
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    std::vector<std::pair<uint32_t, uint32_t>> class_ranges; // unused at top level
    collect_body_ranges(root, lang, ranges, class_ranges);

    // Sort ranges by start position (ascending)
    std::sort(ranges.begin(), ranges.end());

    // Apply replacements in reverse order to preserve positions
    std::string result = code;
    for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
        result.replace(it->first, it->second - it->first, lang->body.replacement);
    }

    ts_tree_delete(tree);
    ts_parser_delete(parser);

    return result;
}

//------------------------------------------------------------------------------
// Heuristic fallback (brace counting)
//------------------------------------------------------------------------------

static std::string compact_heuristic(const std::string & code) {
    // Simple heuristic: find lines starting with def/function/class keywords,
    // strip body between first { and matching }.
    // This handles C-like languages and Python-like languages roughly.

    std::istringstream stream(code);
    std::string line, result;
    int brace_depth = 0;
    bool in_function = false;
    int func_braces = 0;

    // Try Python-style first: def foo(): ... strip indented block
    std::string python_result;
    std::istringstream py_stream(code);
    bool in_py_func = false;
    int py_indent = 0;

    while (std::getline(py_stream, line)) {
        // Count leading whitespace
        size_t indent = 0;
        for (char c : line) { if (c == ' ' || c == '\t') indent++; else break; }

        if (!in_py_func) {
            // Check for function/class definition
            std::string trimmed = line;
            trimmed.erase(0, trimmed.find_first_not_of(" \t"));
            if (trimmed.rfind("def ", 0) == 0 || trimmed.rfind("class ", 0) == 0 ||
                trimmed.rfind("async def ", 0) == 0) {
                in_py_func = true;
                py_indent = (int)indent;
                python_result += line + "\n";
                python_result += std::string(py_indent + 1, ' ') + "# ...\n";
                continue;
            }
            python_result += line + "\n";
        } else {
            // Inside a function: skip lines with same or greater indent than function's first line
            if ((int)indent <= py_indent && !line.empty() &&
                line.find_first_not_of(" \t") != std::string::npos) {
                in_py_func = false;
                python_result += line + "\n";
            }
            // else: skip this line (strip body)
        }
    }

    if (!in_py_func && (python_result.size() < code.size() * 0.7f)) {
        // Python heuristic worked (reduced by at least 30%)
        return python_result;
    }

    // Fall through to brace-based heuristic for C-like
    std::string brace_result;
    std::istringstream br_stream(code);
    int braces_needed = 0;
    bool in_braces = false;

    while (std::getline(br_stream, line)) {
        if (!in_braces) {
            brace_result += line + "\n";
            // Count opening braces in this line
            for (char c : line) {
                if (c == '{') {
                    if (braces_needed == 0) {
                        in_braces = true;
                    }
                    braces_needed++;
                }
                if (c == '}') braces_needed--;
            }
            if (in_braces) {
                brace_result += "    /* ... */\n";
            }
        } else {
            for (char c : line) {
                if (c == '{') braces_needed++;
                if (c == '}') {
                    braces_needed--;
                    if (braces_needed <= 0) {
                        brace_result += line + "\n";
                        in_braces = false;
                        goto next_line;
                    }
                }
            }
        }
        next_line:;
    }

    return brace_result;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------

void init_parsers() {
    // Parsers are initialized on first use by ts_parser_set_language.
    // No explicit initialization needed.
}

std::string compact_code_block(const std::string & code, const std::string & lang_name) {
    const LangEntry *lang = find_lang(lang_name);
    if (lang) {
        std::string result = compact_with_treesitter(code, lang);
        // If tree-sitter produced a meaningful reduction, use it
        if (result.size() < code.size() * 0.95f) {
            return result;
        }
        // Otherwise fall through to heuristic
    }
    return compact_heuristic(code);
}

std::string compact_prompt(const std::string & prompt) {
    // Regex to find fenced code blocks: ```language\n content ```
    // Handles both ```lang and ``` (no language tag)
    // Uses [\\s\\S] instead of . to match across newlines (no std::regex::dotall in C++17)
    static const std::regex fenced_block(
        "```([a-zA-Z0-9+#._-]*)\\s*\n"   // opening fence with optional language
        "([\\s\\S]*?)"                     // content (non-greedy, matches newlines)
        "```",                             // closing fence
        std::regex::nosubs
    );

    std::string result;
    size_t last_end = 0;
    auto begin = std::sregex_iterator(prompt.begin(), prompt.end(), fenced_block);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        // Get match positions manually since we used nosubs
        auto match = *it;
        size_t pos = match.position();
        size_t len = match.length();

        // Extract full match text
        std::string full_match = prompt.substr(pos, len);

        // Find first newline to extract language tag and content
        size_t nl_pos = full_match.find('\n');
        std::string lang_tag;
        std::string content;
        if (nl_pos != std::string::npos) {
            lang_tag = full_match.substr(3, nl_pos - 3); // skip ```
            // Trim whitespace from lang tag
            lang_tag.erase(lang_tag.find_last_not_of(" \t\r") + 1);
            content = full_match.substr(nl_pos + 1);
            // Remove trailing ```
            if (content.size() >= 3) {
                content = content.substr(0, content.size() - 3);
            }
            // Trim trailing whitespace/newlines from content
            content.erase(content.find_last_not_of(" \t\n\r") + 1);
        }

        // Append text before this block
        result.append(prompt, last_end, pos - last_end);

        // Compact the code block
        std::string compacted;
        if (!content.empty()) {
            compacted = compact_code_block(content, lang_tag);
        } else {
            compacted = content; // empty block, keep as-is
        }

        // Reconstruct fenced block
        if (lang_tag.empty()) {
            result += "```\n" + compacted + "\n```";
        } else {
            result += "```" + lang_tag + "\n" + compacted + "\n```";
        }

        last_end = pos + len;
    }

    // Append remaining text
    result.append(prompt, last_end, prompt.size() - last_end);

    return result;
}

} // namespace compact
