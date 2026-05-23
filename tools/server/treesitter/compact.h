#pragma once

#include <string>

namespace compact {

// Compact code blocks in the prompt using tree-sitter AST analysis.
// Strips function/class/method bodies, keeps signatures and docstrings.
std::string compact_prompt(const std::string & prompt);

// Per-code-block compaction with language detection
std::string compact_code_block(const std::string & code, const std::string & lang);

// Initialize all registered language parsers (called once at startup)
void init_parsers();

} // namespace compact
