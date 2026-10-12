import Cocoa

let args = CommandLine.arguments
guard args.count > 1 else { exit(1) }
let iconsetDir = URL(fileURLWithPath: args[1])

let sizes: [(Int, String)] = [
    (16, "icon_16x16.png"),
    (32, "icon_16x16@2x.png"),
    (32, "icon_32x32.png"),
    (64, "icon_32x32@2x.png"),
    (128, "icon_128x128.png"),
    (256, "icon_128x128@2x.png"),
    (256, "icon_256x256.png"),
    (512, "icon_256x256@2x.png"),
    (512, "icon_512x512.png"),
    (1024, "icon_512x512@2x.png")
]

for (px, filename) in sizes {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: px,
        pixelsHigh: px,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    )!

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    let ctx = NSGraphicsContext.current!.cgContext

    let s = CGFloat(px)
    let inset = s * 0.09
    let rect = CGRect(x: inset, y: inset, width: s - inset * 2, height: s - inset * 2)
    let radius = rect.width * 0.225

    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    path.addClip()

    // Glossy Apple Blue gradient background
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bgColors = [
        CGColor(red: 0.30, green: 0.68, blue: 1.00, alpha: 1.0),
        CGColor(red: 0.00, green: 0.44, blue: 0.92, alpha: 1.0)
    ] as CFArray
    let grad = CGGradient(colorsSpace: colorSpace, colors: bgColors, locations: [0.0, 1.0])!
    ctx.drawLinearGradient(
        grad,
        start: CGPoint(x: s / 2, y: rect.maxY),
        end: CGPoint(x: s / 2, y: rect.minY),
        options: []
    )

    // Soundwave bars in white
    let barHeights: [CGFloat] = [0.22, 0.42, 0.58, 0.36, 0.24]
    let barWidth = rect.width * 0.075
    let gap = rect.width * 0.055
    let totalW = CGFloat(barHeights.count) * barWidth + CGFloat(barHeights.count - 1) * gap
    var x = s / 2 - totalW / 2

    NSColor.white.setFill()
    for hFactor in barHeights {
        let bh = rect.height * hFactor
        let barRect = CGRect(x: x, y: (s - bh) / 2, width: barWidth, height: bh)
        let barPath = NSBezierPath(roundedRect: barRect, xRadius: barWidth / 2, yRadius: barWidth / 2)
        barPath.fill()
        x += barWidth + gap
    }

    NSGraphicsContext.restoreGraphicsState()

    if let pngData = rep.representation(using: .png, properties: [:]) {
        try? pngData.write(to: iconsetDir.appendingPathComponent(filename))
    }
}
