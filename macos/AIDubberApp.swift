import Cocoa
import WebKit
import UniformTypeIdentifiers

final class DropTargetWebView: WKWebView {
    private let allowedExtensions: Set<String> = ["mp4", "mov", "mkv", "m4v", "avi"]
    var lastMouseDownEvent: NSEvent?

    override init(frame: CGRect, configuration: WKWebViewConfiguration) {
        super.init(frame: frame, configuration: configuration)
        registerForDraggedTypes([.fileURL])
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        registerForDraggedTypes([.fileURL])
    }

    override func mouseDown(with event: NSEvent) {
        self.lastMouseDownEvent = event
        let winLoc = event.locationInWindow
        let winHeight = window?.frame.height ?? 459
        let winWidth = window?.frame.width ?? 680
        // Top titlebar strip (excluding traffic lights on left and HF/Models pills on right)
        if winLoc.y > winHeight - 40 && winLoc.x > 75 && winLoc.x < winWidth - 155 {
            window?.performDrag(with: event)
            return
        }
        super.mouseDown(with: event)
    }

    private func extractVideoURL(from info: NSDraggingInfo) -> URL? {
        guard let urls = info.draggingPasteboard.readObjects(
            forClasses: [NSURL.self],
            options: [.urlReadingFileURLsOnly: true]
        ) as? [URL] else {
            return nil
        }
        for url in urls {
            if allowedExtensions.contains(url.pathExtension.lowercased()) {
                return url
            }
        }
        return urls.first
    }

    override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        if extractVideoURL(from: sender) != nil {
            evaluateJavaScript("document.getElementById('wellCard')?.classList.add('drag-over');", completionHandler: nil)
            return .copy
        }
        return []
    }

    override func draggingUpdated(_ sender: NSDraggingInfo) -> NSDragOperation {
        if extractVideoURL(from: sender) != nil {
            return .copy
        }
        return []
    }

    override func draggingExited(_ sender: NSDraggingInfo?) {
        evaluateJavaScript("document.getElementById('wellCard')?.classList.remove('drag-over');", completionHandler: nil)
    }

    override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        evaluateJavaScript("document.getElementById('wellCard')?.classList.remove('drag-over');", completionHandler: nil)
        guard let fileURL = extractVideoURL(from: sender) else {
            return false
        }
        let escapedPath = fileURL.path
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
        evaluateJavaScript("window.onNativeFileSelected && window.onNativeFileSelected('\(escapedPath)');", completionHandler: nil)
        return true
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKScriptMessageHandler {
    var window: NSWindow!
    var webView: DropTargetWebView!
    var backendProcess: Process?
    let apiPort = 8765

    private func resolveProjectRoot() -> URL {
        // 1. Check relative to the .app bundle (e.g. /Volumes/develop/AI-Dubbing/AI Dubber.app)
        let bundleURL = Bundle.main.bundleURL
        let parentURL = bundleURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: parentURL.appendingPathComponent("core/app/main.py").path) {
            return parentURL
        }
        // 2. Fallback to standard project workspace path
        let defaultRoot = URL(fileURLWithPath: "/Volumes/develop/AI-Dubbing")
        if FileManager.default.fileExists(atPath: defaultRoot.appendingPathComponent("core/app/main.py").path) {
            return defaultRoot
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private func setupMainMenu() {
        let mainMenu = NSMenu()

        // App Menu
        let appMenuItem = NSMenuItem()
        mainMenu.addItem(appMenuItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "О программе AI Dubber", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Скрыть AI Dubber", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(NSMenuItem.separator())
        appMenu.addItem(withTitle: "Завершить AI Dubber", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appMenuItem.submenu = appMenu

        // Edit Menu (enables Cmd+C, Cmd+V, Cmd+X, Cmd+A in WKWebView inputs)
        let editMenuItem = NSMenuItem()
        mainMenu.addItem(editMenuItem)
        let editMenu = NSMenu(title: "Правка")
        editMenu.addItem(withTitle: "Отменить", action: Selector(("undo:")), keyEquivalent: "z")
        editMenu.addItem(withTitle: "Повторить", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(NSMenuItem.separator())
        editMenu.addItem(withTitle: "Вырезать", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        editMenu.addItem(withTitle: "Копировать", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Вставить", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editMenu.addItem(withTitle: "Выделить всё", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editMenuItem.submenu = editMenu

        NSApp.mainMenu = mainMenu
    }

    private func killStaleBackendOnPort() {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        task.arguments = ["-t", "-i", ":\(apiPort)"]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            task.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            if let output = String(data: data, encoding: .utf8) {
                for line in output.split(whereSeparator: \.isNewline) {
                    if let pid = Int32(line.trimmingCharacters(in: .whitespaces)), pid != getpid() {
                        kill(pid, SIGTERM)
                    }
                }
            }
        } catch {}
    }

    private func startBackendIfNeeded(completion: @escaping (Bool) -> Void) {
        let statusURL = URL(string: "http://127.0.0.1:\(apiPort)/api/dub/status")!
        var req = URLRequest(url: statusURL)
        req.timeoutInterval = 0.4

        URLSession.shared.dataTask(with: req) { [weak self] data, response, _ in
            guard let self = self else { return }
            if let http = response as? HTTPURLResponse, http.statusCode == 200,
               let data = data,
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let isRunning = json["is_running"] as? Bool, isRunning == true {
                DispatchQueue.main.async { completion(true) }
                return
            }
            self.killStaleBackendOnPort()
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) {
                self.spawnBackendProcess(completion: completion)
            }
        }.resume()
    }

    private func spawnBackendProcess(completion: @escaping (Bool) -> Void) {
        let root = resolveProjectRoot()
        let pythonPath = root.appendingPathComponent(".venvs/asr/bin/python").path
        guard FileManager.default.fileExists(atPath: pythonPath) else {
            DispatchQueue.main.async { completion(false) }
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.currentDirectoryURL = root
        proc.arguments = [
            "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1",
            "--port", "\(apiPort)",
            "--log-level", "warning"
        ]

        var env = ProcessInfo.processInfo.environment
        let corePath = root.appendingPathComponent("core").path
        env["PYTHONPATH"] = "\(corePath):\(root.path)"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        proc.environment = env

        do {
            try proc.run()
            self.backendProcess = proc
        } catch {
            DispatchQueue.main.async { completion(false) }
            return
        }

        // Poll health until ready (up to ~6 seconds)
        func pollReady(attemptsLeft: Int) {
            guard attemptsLeft > 0 else {
                DispatchQueue.main.async { completion(false) }
                return
            }
            let healthURL = URL(string: "http://127.0.0.1:\(self.apiPort)/api/health")!
            var req = URLRequest(url: healthURL)
            req.timeoutInterval = 0.35
            URLSession.shared.dataTask(with: req) { _, response, _ in
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    DispatchQueue.main.async { completion(true) }
                } else {
                    DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) {
                        pollReady(attemptsLeft: attemptsLeft - 1)
                    }
                }
            }.resume()
        }

        pollReady(attemptsLeft: 24)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMainMenu()

        let rect = NSRect(x: 0, y: 0, width: 680, height: 459)
        window = NSWindow(
            contentRect: rect,
            styleMask: [.titled, .closable, .miniaturizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "AI Dubber"
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.backgroundColor = NSColor(calibratedWhite: 0.96, alpha: 1.0)
        window.setContentSize(NSSize(width: 680, height: 459))
        window.minSize = NSSize(width: 680, height: 459)
        window.maxSize = NSSize(width: 680, height: 459)
        window.center()

        let contentController = WKUserContentController()
        contentController.add(self, name: "nativeApp")

        let config = WKWebViewConfiguration()
        config.userContentController = contentController

        webView = DropTargetWebView(frame: rect, configuration: config)
        webView.autoresizingMask = [.width, .height]
        webView.setValue(false, forKey: "drawsBackground")

        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        startBackendIfNeeded { [weak self] backendReady in
            guard let self = self else { return }
            let root = self.resolveProjectRoot()
            let localHTML = root.appendingPathComponent("core/app/ui/index.html")
            let bundledHTML = Bundle.main.resourceURL?.appendingPathComponent("index.html")
            if backendReady, let url = URL(string: "http://127.0.0.1:\(self.apiPort)/") {
                self.webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
            } else if FileManager.default.fileExists(atPath: localHTML.path) {
                self.webView.loadFileURL(localHTML, allowingReadAccessTo: localHTML.deletingLastPathComponent())
            } else if let bundledHTML = bundledHTML, FileManager.default.fileExists(atPath: bundledHTML.path) {
                self.webView.loadFileURL(bundledHTML, allowingReadAccessTo: bundledHTML.deletingLastPathComponent())
            }
        }
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any],
              let action = body["action"] as? String else {
            return
        }

        switch action {
        case "pickFile":
            let panel = NSOpenPanel()
            panel.title = "Выберите видео для дубляжа"
            panel.canChooseFiles = true
            panel.canChooseDirectories = false
            panel.allowsMultipleSelection = false
            panel.allowedContentTypes = [.mpeg4Movie, .quickTimeMovie, .movie, .video, .avi]
            let root = resolveProjectRoot()
            let inputDir = root.appendingPathComponent("input")
            if FileManager.default.fileExists(atPath: inputDir.path) {
                panel.directoryURL = inputDir
            }
            if panel.runModal() == .OK, let url = panel.url {
                let escapedPath = url.path
                    .replacingOccurrences(of: "\\", with: "\\\\")
                    .replacingOccurrences(of: "'", with: "\\'")
                webView.evaluateJavaScript("window.onNativeFileSelected && window.onNativeFileSelected('\(escapedPath)');", completionHandler: nil)
            }

        case "revealFile":
            if let path = body["path"] as? String, !path.isEmpty {
                let fileURL = URL(fileURLWithPath: path)
                if FileManager.default.fileExists(atPath: fileURL.path) {
                    NSWorkspace.shared.activateFileViewerSelecting([fileURL])
                    return
                }
            }
            let outDir = resolveProjectRoot().appendingPathComponent("output")
            NSWorkspace.shared.open(outDir)

        case "openUrl":
            if let urlStr = body["url"] as? String, let url = URL(string: urlStr) {
                NSWorkspace.shared.open(url)
            }

        case "startDrag":
            if let event = webView.lastMouseDownEvent ?? NSApp.currentEvent {
                window.performDrag(with: event)
            }

        default:
            break
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        let instances = NSWorkspace.shared.runningApplications.filter {
            $0.bundleIdentifier == "com.aidubbing.mac"
        }
        if instances.count <= 1, let proc = backendProcess, proc.isRunning {
            proc.terminate()
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
