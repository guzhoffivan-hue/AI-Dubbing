#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_BUNDLE="$ROOT/AI Dubber.app"

mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"
cp "$ROOT/macos/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
cp "$ROOT/core/app/ui/index.html" "$APP_BUNDLE/Contents/Resources/index.html"

# Generate AppIcon.icns if not present
if [ ! -f "$APP_BUNDLE/Contents/Resources/AppIcon.icns" ]; then
  ICONSET_TMP="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$ICONSET_TMP"
  /usr/bin/swift "$ROOT/macos/make_icon.swift" "$ICONSET_TMP"
  /usr/bin/iconutil -c icns "$ICONSET_TMP" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
  rm -rf "$(dirname "$ICONSET_TMP")"
fi

/usr/bin/swiftc -O -swift-version 5 \
  "$ROOT/macos/AIDubberApp.swift" \
  -o "$APP_BUNDLE/Contents/MacOS/AI Dubber" \
  -framework Cocoa \
  -framework WebKit \
  -framework UniformTypeIdentifiers

codesign --force --deep -s - "$APP_BUNDLE"
echo "Built native macOS app: $APP_BUNDLE"
