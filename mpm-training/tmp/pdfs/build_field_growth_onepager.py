from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "field-integrated-material-growth.pdf"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE = landscape((720, 405))
W, H = PAGE

INK = HexColor("#172126")
MUTED = HexColor("#637077")
PAPER = HexColor("#F6F5F0")
WHITE = HexColor("#FFFFFF")
GREEN = HexColor("#20A66A")
GREEN_DARK = HexColor("#11754A")
GREEN_PALE = HexColor("#DDF2E7")
RED = HexColor("#D25C51")
RED_PALE = HexColor("#F7E3DF")
BLUE = HexColor("#3D6F8E")
GRID = HexColor("#CAD4D1")


def rounded(c, x, y, w, h, radius=12, fill=WHITE, stroke=None, width=1):
    c.setLineWidth(width)
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1 if stroke else 0)


def label(c, text, x, y, size, color=INK, font="Helvetica", align="left"):
    c.setFillColor(color)
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def dot(c, x, y, r=5, fill=BLUE, stroke=WHITE, width=1.2):
    c.setLineWidth(width)
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.circle(x, y, r, fill=1, stroke=1)


def arrow(c, x1, y1, x2, y2, color=GREEN_DARK, width=2):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    s = 7
    c.line(x2, y2, x2 - ux * s + px * 3.5, y2 - uy * s + py * 3.5)
    c.line(x2, y2, x2 - ux * s - px * 3.5, y2 - uy * s - py * 3.5)


def step_bad(c, x, y):
    rounded(c, x, y, 200, 218, fill=RED_PALE)
    label(c, "SAMPLE-DRIVEN GROWTH", x + 18, y + 188, 9, RED, "Helvetica-Bold")
    label(c, "Every nearby point", x + 18, y + 164, 14, INK, "Helvetica-Bold")
    label(c, "makes its own growth decision.", x + 18, y + 145, 12, INK, "Helvetica-Bold")

    source = [(x + 53, y + 87), (x + 82, y + 103), (x + 84, y + 72)]
    targets = [(x + 136, y + 90), (x + 141, y + 94), (x + 137, y + 86)]
    for sx, sy in source:
        dot(c, sx, sy, 7)
    for (sx, sy), (tx, ty) in zip(source, targets):
        arrow(c, sx + 8, sy, tx - 8, ty, RED, 1.5)
        dot(c, tx, ty, 7, RED, WHITE)

    c.setStrokeColor(RED)
    c.setLineWidth(1.5)
    c.circle(x + 138, y + 90, 19, fill=0, stroke=1)
    label(c, "overlap", x + 138, y + 48, 10, RED, "Helvetica-Bold", "center")
    label(c, "More samples can accidentally mean more growth.", x + 18, y + 18, 8.5, MUTED)


def vote_icon(c, cx, cy):
    for dx, dy in [(-18, 8), (0, 17), (18, 5), (-8, -12), (14, -14)]:
        dot(c, cx + dx, cy + dy, 4.5)
        c.setStrokeColor(GREEN)
        c.setLineWidth(1)
        c.line(cx + dx, cy + dy + 7, cx + dx, cy + dy + 14)


def field_icon(c, cx, cy):
    s = 11
    values = [0.10, 0.22, 0.35, 0.18, 0.30, 0.62, 0.82, 0.42, 0.14, 0.40, 0.66, 0.31]
    for row in range(3):
        for col in range(4):
            alpha = values[row * 4 + col]
            c.setFillColor(Color(GREEN.red, GREEN.green, GREEN.blue, alpha=alpha))
            c.setStrokeColor(WHITE)
            c.rect(cx - 2 * s + col * s, cy - 1.5 * s + row * s, s, s, fill=1, stroke=1)
    label(c, "average", cx, cy - 29, 7.5, GREEN_DARK, "Helvetica-Bold", "center")


def volume_icon(c, cx, cy):
    c.setFillColor(GREEN_PALE)
    c.setStrokeColor(GREEN_DARK)
    c.setLineWidth(1.5)
    c.circle(cx, cy, 23, fill=1, stroke=1)
    c.setDash(3, 2)
    c.circle(cx, cy, 15, fill=0, stroke=1)
    c.setDash()
    for dx, dy in [(-8, 5), (8, 6), (0, -8)]:
        dot(c, cx + dx, cy + dy, 4.5)


def deficit_icon(c, cx, cy):
    for angle in range(0, 360, 45):
        import math
        rad = math.radians(angle)
        radius = 17 if angle != 0 else 8
        dot(c, cx + math.cos(rad) * radius, cy + math.sin(rad) * radius, 3.8)
    arrow(c, cx + 10, cy, cx + 34, cy, GREEN_DARK, 1.8)
    c.setFillColor(Color(GREEN.red, GREEN.green, GREEN.blue, alpha=0.14))
    c.setStrokeColor(GREEN)
    c.wedge(cx - 26, cy - 26, cx + 26, cy + 26, -28, 56, fill=1, stroke=0)


def insert_icon(c, cx, cy):
    for dx, dy in [(-19, 8), (-6, 17), (9, 12), (-13, -10), (3, -8)]:
        dot(c, cx + dx, cy + dy, 4.5)
    dot(c, cx + 24, cy, 6.5, GREEN, WHITE, 1.5)
    c.setStrokeColor(GREEN_DARK)
    c.setLineWidth(1.2)
    c.circle(cx + 24, cy, 11, fill=0, stroke=1)


def step_good(c, x, y):
    rounded(c, x, y, 432, 218, fill=WHITE, stroke=GREEN_PALE, width=1.2)
    label(c, "FIELD-INTEGRATED GROWTH", x + 18, y + 188, 9, GREEN_DARK, "Helvetica-Bold")
    label(c, "The material decides once. Samples follow.", x + 18, y + 164, 15, INK, "Helvetica-Bold")

    centers = [x + 47, x + 130, x + 216, x + 302, x + 386]
    cy = y + 95
    vote_icon(c, centers[0], cy)
    field_icon(c, centers[1], cy)
    volume_icon(c, centers[2], cy)
    deficit_icon(c, centers[3], cy)
    insert_icon(c, centers[4], cy)
    for a, b in zip(centers[:-1], centers[1:]):
        arrow(c, a + 29, cy, b - 30, cy, GREEN_DARK, 1.3)

    captions = [
        ("1", "Samples vote"),
        ("2", "Field integrates"),
        ("3", "Volume expands"),
        ("4", "Deficit guides"),
        ("5", "One sample added"),
    ]
    for cx, (num, text) in zip(centers, captions):
        c.setFillColor(GREEN_DARK)
        c.circle(cx, y + 36, 7, fill=1, stroke=0)
        label(c, num, cx, y + 33.3, 7, WHITE, "Helvetica-Bold", "center")
        label(c, text, cx, y + 16, 8.2, INK, "Helvetica-Bold", "center")


c = Canvas(str(OUTPUT), pagesize=PAGE)
c.setTitle("Field-integrated material growth")
c.setAuthor("Neural Graph")
c.setFillColor(PAPER)
c.rect(0, 0, W, H, fill=1, stroke=0)

label(c, "Growing material, not dividing particles", 36, 356, 26, INK, "Helvetica-Bold")
label(c, "A sampling-aware model for morphogenesis with MPM", 36, 334, 11.5, MUTED)

step_bad(c, 36, 91)
step_good(c, 252, 91)

rounded(c, 36, 35, 648, 38, radius=10, fill=INK)
label(c, "CORE PRINCIPLE", 52, 51, 8, GREEN_PALE, "Helvetica-Bold")
label(c, "Biology controls continuum growth. Particle creation only maintains numerical coverage.", 135, 48.5, 11, WHITE, "Helvetica-Bold")

label(c, "Nearby intentions are averaged, persistent growth credit belongs to space, and target claims prevent duplicate insertion.", 36, 16, 8, MUTED)
label(c, "FIELD-INTEGRATED MATERIAL GROWTH", 684, 16, 7.5, GREEN_DARK, "Helvetica-Bold", "right")

c.showPage()
c.save()
print(OUTPUT)
