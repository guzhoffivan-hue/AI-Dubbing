#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/macos/build_app.sh"

APP_BUNDLE="$ROOT/AI Dubber.app"
DMG_PATH="$ROOT/AI-Dubber.dmg"
STAGING_DIR="$(mktemp -d)"

cp -R "$APP_BUNDLE" "$STAGING_DIR/AI Dubber.app"
ln -s /Applications "$STAGING_DIR/Applications"

rm -f "$DMG_PATH"
/usr/bin/hdiutil create \
  -volname "AI Dubber" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

rm -rf "$STAGING_DIR"
echo "Created DMG installer: $DMG_PATH"
