#!/usr/bin/env bash
# Locate the Claude Code command-line binary on this machine and call it the way README 6.7 specifies for
# the judge: custom system prompt, no tools, no user/project settings or memory, no MCP servers, no saved
# session, clean environment, empty working directory. The prompt is read from stdin.
#
# usage: claude_cli.sh <system-prompt-file> [model] [effort] < prompt.txt   -> prints one JSON result object
#
# The binary ships inside the VS Code extension, whose folder name contains a version number that changes
# on every extension update, so the path is resolved at call time rather than stored.
set -euo pipefail
find_claude() {
  if [[ -n "${CLAUDE_CLI:-}" && -x "${CLAUDE_CLI}" ]]; then echo "$CLAUDE_CLI"; return; fi
  if command -v claude >/dev/null 2>&1; then command -v claude; return; fi
  ls -d "$HOME"/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude 2>/dev/null | sort -V | tail -1
}
if [[ "${1:-}" == "--where" ]]; then find_claude; exit 0; fi
system_file=${1:?system prompt file required}; model=${2:-claude-opus-5-5}; effort=${3:-high}
bin=$(find_claude); [[ -x "$bin" ]] || { echo "Claude Code binary not found" >&2; exit 3; }
work=$(mktemp -d "/tmp/agent4-judge.${UID}.XXXXXX"); trap 'rm -rf "$work"' EXIT
cd "$work"
env -i HOME="$HOME" PATH=/usr/bin:/bin TERM=dumb "$bin" -p \
  --model "$model" --effort "$effort" --system-prompt "$(cat "$system_file")" \
  --tools "" --setting-sources "" --strict-mcp-config --disable-slash-commands --safe-mode \
  --no-session-persistence --output-format json
