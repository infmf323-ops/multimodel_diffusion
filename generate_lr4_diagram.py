from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


TITLE = load_font(28)
TEXT = load_font(20)
SMALL = load_font(16)


def box(draw, xy, text, fill):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline="#1f2937", width=3)
    x0, y0, x1, y1 = xy
    bbox = draw.multiline_textbbox((0, 0), text, font=TEXT, spacing=6, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2),
        text,
        font=TEXT,
        fill="#111827",
        spacing=6,
        align="center",
    )


def arrow(draw, start, end, text=""):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill="#374151", width=4)
    if x1 >= x0:
        pts = [(x1, y1), (x1 - 16, y1 - 10), (x1 - 16, y1 + 10)]
    else:
        pts = [(x1, y1), (x1 + 16, y1 - 10), (x1 + 16, y1 + 10)]
    draw.polygon(pts, fill="#374151")
    if text:
        draw.text(((x0 + x1) / 2 + 6, (y0 + y1) / 2 - 16), text, font=SMALL, fill="#374151")


img = Image.new("RGB", (1700, 980), "#faf7f1")
draw = ImageDraw.Draw(img)
draw.text((60, 28), "Финальная архитектура демо-сервиса (ЛР4)", font=TITLE, fill="#111827")

box(draw, (90, 200, 320, 290), "Пользователь", "#dbeafe")
box(draw, (420, 180, 730, 300), "Frontend\nHTML + JS\nпорт 3000", "#e0f2fe")
box(draw, (840, 150, 1180, 330), "Inference Service\nFastAPI + финальная модель ЛР3\nпорт 8000", "#ede9fe")
box(draw, (1280, 150, 1570, 250), "PostgreSQL\nистория запросов", "#f3f4f6")
box(draw, (1280, 310, 1570, 410), "Lab3 artifacts\nsummary + logs + model", "#dcfce7")
box(draw, (840, 470, 1180, 610), "Prometheus\nметрики /metrics\nпорт 9090", "#fef3c7")

arrow(draw, (320, 245), (420, 245), "открывает UI")
arrow(draw, (730, 245), (840, 245), "HTTP /predict")
arrow(draw, (1180, 210), (1280, 210), "SQL")
arrow(draw, (1180, 350), (1280, 350), "reads")
arrow(draw, (1180, 540), (1280, 540), "")
arrow(draw, (1010, 330), (1010, 470), "/metrics")
draw.text((1260, 520), "скрапинг\nметрик", font=SMALL, fill="#374151")

draw.text(
    (90, 700),
    "В финальном демо-контуре ЛР4 разворачивается именно serving-срез системы:\n"
    "UI, inference API, база истории запросов, monitoring и доступ к артефактам ЛР3.\n"
    "Генерация синтетических данных и fine-tuning остаются частью offline pipeline из ЛР1-ЛР3.",
    font=TEXT,
    fill="#374151",
    spacing=8,
)

img.save(ROOT / "lr4_deployment_diagram.png")
