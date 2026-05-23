#define _POSIX_C_SOURCE 200112L
#define _DEFAULT_SOURCE

// Custom amalgamation for tree-sitter (without WebAssembly support)

#include "./alloc.c"
#include "./get_changed_ranges.c"
#include "./language.c"
#include "./lexer.c"
#include "./node.c"
#include "./parser.c"
#include "./query.c"
#include "./stack.c"
#include "./subtree.c"
#include "./tree_cursor.c"
#include "./tree.c"
