#!/usr/bin/env bash
# Keyless outbound HTTPS check. Any HTTP status code means the host is reachable; 000 means it is not.
for url in https://generativelanguage.googleapis.com/ https://router.huggingface.co/ https://api.anthropic.com/ https://pypi.org/simple/ ; do
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 15 "$url" 2>/dev/null || true)
  echo "NET $(hostname) $url http=${code:-000}"
done
