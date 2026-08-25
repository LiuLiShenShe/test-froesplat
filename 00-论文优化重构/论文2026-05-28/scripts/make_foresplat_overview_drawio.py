#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import html
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom


OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "fig1_foresplat_overview.drawio"


class Drawio:
    def __init__(self):
        self.next_id = 2
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": "2026-05-28T12:00:00.000Z",
                "agent": "Codex",
                "version": "26.0.0",
                "type": "device",
            },
        )
        self.diagram = ET.SubElement(self.mxfile, "diagram", {"name": "ForeSplat Overview"})
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1680",
                "dy": "930",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": "1680",
                "pageHeight": "930",
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _id(self) -> str:
        out = str(self.next_id)
        self.next_id += 1
        return out

    def rect(self, x, y, w, h, value="", style="", parent="1", rounded=True, rid=None):
        rid = rid or self._id()
        base = "rounded=1;whiteSpace=wrap;html=1;" if rounded else "whiteSpace=wrap;html=1;"
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": rid,
                "value": value,
                "style": base + style,
                "vertex": "1",
                "parent": parent,
            },
        )
        geo = ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return rid

    def ellipse(self, x, y, w, h, value="", style="", parent="1", rid=None):
        rid = rid or self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": rid,
                "value": value,
                "style": "ellipse;whiteSpace=wrap;html=1;" + style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return rid

    def text(self, x, y, w, h, value, style="", parent="1", rid=None):
        rid = rid or self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": rid,
                "value": value,
                "style": "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;" + style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"})
        return rid

    def line(self, points, style="", value="", parent="1", rid=None):
        rid = rid or self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": rid,
                "value": value,
                "style": "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;" + style,
                "edge": "1",
                "parent": parent,
            },
        )
        geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            ET.SubElement(geo, "mxPoint", {"x": str(points[0][0]), "y": str(points[0][1]), "as": "sourcePoint"})
            ET.SubElement(geo, "mxPoint", {"x": str(points[-1][0]), "y": str(points[-1][1]), "as": "targetPoint"})
        arr = ET.SubElement(geo, "Array", {"as": "points"})
        for x, y in points[1:-1]:
            ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})
        return rid

    def arrow(self, x1, y1, x2, y2, color="#333333", dashed=False, width=2, parent="1", rid=None):
        style = f"endArrow=block;endFill=1;strokeColor={color};strokeWidth={width};"
        if dashed:
            style += "dashed=1;dashPattern=4 4;"
        rid = rid or self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": rid,
                "value": "",
                "style": "edgeStyle=none;html=1;rounded=0;" + style,
                "edge": "1",
                "parent": parent,
            },
        )
        geo = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geo, "mxPoint", {"x": str(x1), "y": str(y1), "as": "sourcePoint"})
        ET.SubElement(geo, "mxPoint", {"x": str(x2), "y": str(y2), "as": "targetPoint"})
        return rid

    def save(self, path: Path):
        raw = ET.tostring(self.mxfile, encoding="utf-8")
        parsed = minidom.parseString(raw)
        path.write_text(parsed.toprettyxml(indent="  "), encoding="utf-8")


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def style_panel(stroke, fill):
    return (
        f"arcSize=8;fillColor={fill};strokeColor={stroke};strokeWidth=1.2;"
        "shadow=0;fontFamily=Arial;fontColor=#111111;"
    )


def style_card(stroke, fill="#FFFFFF"):
    return (
        f"arcSize=8;fillColor={fill};strokeColor={stroke};strokeWidth=1;"
        "shadow=0;fontFamily=Arial;fontColor=#111111;"
    )


def header(value, color):
    return (
        f"<font style='font-size:20px;color:{color};'><b>{value}</b></font>"
    )


def add_camera(g: Drawio, x, y, s=1.0):
    g.rect(x, y + 8 * s, 26 * s, 17 * s, "", "rounded=1;arcSize=15;fillColor=#5d6773;strokeColor=#222222;strokeWidth=1;", rounded=True)
    g.rect(x + 5 * s, y + 2 * s, 10 * s, 8 * s, "", "rounded=1;arcSize=15;fillColor=#7d8792;strokeColor=#222222;strokeWidth=1;", rounded=True)
    g.ellipse(x + 6 * s, y + 8 * s, 14 * s, 14 * s, "", "fillColor=#dce3ea;strokeColor=#222222;strokeWidth=1;")
    g.ellipse(x + 10 * s, y + 12 * s, 6 * s, 6 * s, "", "fillColor=#2b333b;strokeColor=#111111;strokeWidth=1;")


def add_photo(g: Drawio, x, y, w=108, h=118):
    g.rect(x, y, w, h, "", "arcSize=8;fillColor=#FFFFFF;strokeColor=#d1d5db;strokeWidth=1;", rounded=True)
    g.rect(x + 7, y + 8, w - 14, h - 16, "", "rounded=0;fillColor=#f3f4f6;strokeColor=none;", rounded=False)
    add_plant(g, x + w / 2, y + h * 0.57, 0.78, pot=True, mesh=False)
    # tiny red SfM-like floor marks
    for i in range(12):
        px = x + 18 + (i * 7) % (w - 34)
        py = y + h - 20 - ((i * 11) % 20)
        g.ellipse(px, py, 3, 3, "", "fillColor=#d86b57;strokeColor=none;")


def add_plant(g: Drawio, cx, cy, scale=1.0, pot=True, mesh=False, gray=False):
    green = "#6ab04c" if not gray else "#b8b8b8"
    dark = "#2f6f20" if not gray else "#777777"
    fill = "#88c96a" if not gray else "#d0d0d0"
    if pot:
        g.rect(cx - 22 * scale, cy + 38 * scale, 44 * scale, 9 * scale, "", f"rounded=0;fillColor={'#8bb36a' if mesh else '#7b5e45'};strokeColor={'#53883f' if mesh else '#4d3a2b'};strokeWidth=1;", rounded=False)
        g.rect(cx - 18 * scale, cy + 45 * scale, 36 * scale, 35 * scale, "", f"rounded=0;fillColor={'#b7df91' if mesh else '#9a7659'};strokeColor={'#53883f' if mesh else '#4d3a2b'};strokeWidth=1;", rounded=False)
    for dx, dy, rot, ww, hh in [
        (-24, -10, -25, 48, 18),
        (24, -12, 25, 48, 18),
        (-8, -42, -75, 46, 16),
        (12, -42, 75, 44, 16),
        (0, -24, 0, 48, 18),
        (-20, 8, 20, 38, 15),
        (20, 8, -20, 38, 15),
    ]:
        g.ellipse(
            cx + dx * scale - ww * scale / 2,
            cy + dy * scale - hh * scale / 2,
            ww * scale,
            hh * scale,
            "",
            f"fillColor={fill};strokeColor={dark};strokeWidth=1;rotation={rot};",
        )
    for dx in [-8, 0, 8]:
        g.line([(cx, cy + 38 * scale), (cx + dx * scale, cy - 38 * scale)], f"endArrow=none;strokeColor={dark};strokeWidth=1;")
    if mesh:
        for i in range(4):
            g.line([(cx - 24 * scale + i * 14 * scale, cy + 45 * scale), (cx - 18 * scale + i * 10 * scale, cy + 80 * scale)], "endArrow=none;strokeColor=#4f8f34;strokeWidth=0.7;")
        for j in range(4):
            g.line([(cx - 18 * scale, cy + (48 + j * 8) * scale), (cx + 18 * scale, cy + (48 + j * 8) * scale)], "endArrow=none;strokeColor=#4f8f34;strokeWidth=0.7;")


def add_mask(g: Drawio, x, y, w=110, h=92):
    g.rect(x, y, w, h, "", "rounded=0;fillColor=#000000;strokeColor=none;", rounded=False)
    # white plant silhouette approximation
    cx = x + w / 2
    cy = y + h / 2 + 5
    for dx, dy, rot, ww, hh in [
        (-25, -10, -25, 48, 18),
        (25, -11, 25, 48, 18),
        (-7, -37, -74, 42, 15),
        (12, -34, 70, 40, 15),
        (0, -21, 0, 44, 17),
        (-18, 10, 20, 34, 14),
        (18, 10, -20, 34, 14),
    ]:
        g.ellipse(cx + dx - ww / 2, cy + dy - hh / 2, ww, hh, "", f"fillColor=#FFFFFF;strokeColor=none;rotation={rot};")
    g.rect(cx - 15, cy + 22, 30, 28, "", "rounded=0;fillColor=#FFFFFF;strokeColor=none;", rounded=False)


def add_points(g: Drawio, cx, cy, color, n=70, rx=70, ry=45, size=4):
    for i in range(n):
        a = (i * 137.508) % 360
        r = math.sqrt(((i * 53) % 100) / 100)
        px = cx + math.cos(math.radians(a)) * rx * r
        py = cy + math.sin(math.radians(a)) * ry * r
        g.ellipse(round(px, 1), round(py, 1), size, size, "", f"fillColor={color};strokeColor=none;opacity=80;")


def add_bar_chart(g: Drawio, x, y):
    g.line([(x, y + 70), (x + 82, y + 70)], "endArrow=none;strokeColor=#111111;strokeWidth=3;")
    g.line([(x, y + 70), (x, y + 10)], "endArrow=none;strokeColor=#111111;strokeWidth=3;")
    for i, h in enumerate([22, 45, 34]):
        g.rect(x + 16 + i * 20, y + 70 - h, 10, h, "", "rounded=0;fillColor=#f6f6f6;strokeColor=#111111;strokeWidth=1;", rounded=False)
    g.ellipse(x + 83, y + 16, 18, 18, "✓", "fillColor=#b7e0a3;strokeColor=#2a7d2e;fontColor=#2a7d2e;fontStyle=1;fontSize=13;")
    g.ellipse(x + 83, y + 44, 18, 18, "×", "fillColor=#f7c9c4;strokeColor=#b64034;fontColor=#b64034;fontStyle=1;fontSize=13;")


def add_quality_stars(g: Drawio, x, y, stars):
    add_camera(g, x, y, 0.75)
    for i in range(5):
        fill = "#f2c230" if i < stars else "#efefef"
        g.rect(x + 42 + i * 20, y + 5, 14, 14, "★", f"rounded=0;fillColor=none;strokeColor=none;fontColor={fill};fontSize=17;fontStyle=1;", rounded=False)


def add_weight_bar(g: Drawio, x, y, label, frac):
    add_camera(g, x, y - 3, 0.65)
    g.rect(x + 36, y, 84 * frac, 20, "", "rounded=0;fillColor=#e5ae2f;strokeColor=#bf8f1f;strokeWidth=1;", rounded=False)
    g.rect(x + 36 + 84 * frac, y, 84 * (1 - frac), 20, "", "rounded=0;fillColor=#f6ead1;strokeColor=#e0cfaa;strokeWidth=1;", rounded=False)
    g.text(x + 132, y, 35, 20, label, "fontSize=14;align=left;verticalAlign=middle;")


def add_grid_cube(g: Drawio, x, y, w=120, h=145):
    # isometric-ish wire cube
    g.rect(x + 20, y + 20, w - 40, h - 50, "", "rounded=0;fillColor=none;strokeColor=#8c8c8c;strokeWidth=1;", rounded=False)
    for i in range(1, 5):
        xx = x + 20 + i * (w - 40) / 5
        g.line([(xx, y + 20), (xx, y + h - 30)], "endArrow=none;strokeColor=#b3b3b3;strokeWidth=0.7;")
        yy = y + 20 + i * (h - 50) / 5
        g.line([(x + 20, yy), (x + w - 20, yy)], "endArrow=none;strokeColor=#b3b3b3;strokeWidth=0.7;")
    add_plant(g, x + w / 2, y + 73, 0.55, pot=True, mesh=False)


def add_measure_icons(g: Drawio, x, y):
    add_plant(g, x + 65, y + 63, 0.72, pot=True, mesh=True)
    # separate trait icons
    g.line([(x + 24, y + 210), (x + 24, y + 250)], "endArrow=block;startArrow=block;startFill=1;endFill=1;strokeColor=#6ab04c;strokeWidth=2;")
    g.text(x + 72, y + 218, 90, 24, "Plant Height", "fontSize=13;align=left;")
    g.line([(x + 20, y + 286), (x + 62, y + 286)], "endArrow=block;startArrow=block;startFill=1;endFill=1;strokeColor=#6ab04c;strokeWidth=2;")
    g.text(x + 72, y + 274, 100, 24, "Canopy Width", "fontSize=13;align=left;")
    g.ellipse(x + 30, y + 325, 32, 16, "", "fillColor=#8bcf63;strokeColor=#4e8f36;strokeWidth=1;rotation=-35;")
    g.text(x + 72, y + 318, 80, 24, "Leaf Size", "fontSize=13;align=left;")
    g.text(x + 74, y + 362, 80, 26, "…", "fontSize=22;align=left;")


def add_legend_item(g: Drawio, x, y, color, label, stroke=None):
    g.rect(x, y, 30, 30, "", f"arcSize=8;fillColor={color};strokeColor={stroke or color};strokeWidth=1;", rounded=True)
    g.text(x + 46, y - 2, 130, 36, label, "fontSize=13;align=left;verticalAlign=middle;")


def main():
    g = Drawio()

    colors = {
        "input": ("#8b8b8b", "#f7f7f7"),
        "fsam": ("#9b6fc7", "#f5effb"),
        "colmap": ("#4f8ccc", "#eef6ff"),
        "dgs": ("#78a85d", "#f2f9ed"),
        "view": ("#d5a322", "#fff8e6"),
        "prune": ("#ed8b42", "#fff2e8"),
        "mesh": ("#c77839", "#fff4ec"),
        "pheno": ("#a5a5a5", "#f7f7f7"),
    }

    # white canvas
    g.rect(0, 0, 1680, 930, "", "rounded=0;fillColor=#FFFFFF;strokeColor=none;", rounded=False)

    y0, h = 32, 650
    xs = [4, 163, 370, 577, 868, 1069, 1307, 1484]
    ws = [136, 182, 183, 268, 176, 215, 152, 192]
    panel_keys = ["input", "fsam", "colmap", "dgs", "view", "prune", "mesh", "pheno"]
    panels = {}
    for x, w, k in zip(xs, ws, panel_keys):
        stroke, fill = colors[k]
        panels[k] = g.rect(x, y0, w, h, "", style_panel(stroke, fill), rounded=True)

    # headings
    g.text(30, 48, 85, 70, "<font style='font-size:20px'><b>Input</b></font><br><font style='font-size:14px'><b>Multi-view<br>RGB Images</b></font>", "align=center;verticalAlign=top;")
    g.text(198, 52, 110, 30, header("1. FSAM3", "#5b1aa0"), "align=center;")
    g.text(405, 52, 115, 30, header("2. COLMAP", "#1764b0"), "align=center;")
    g.text(612, 52, 210, 30, header("3. Plant-aware 2DGS", "#176d18"), "align=center;")
    g.text(888, 48, 140, 58, header("4. View Quality<br>Soft Weighting", "#a87500"), "align=center;")
    g.text(1104, 48, 150, 58, header("5. Mask-guided<br>Gaussian Pruning", "#e24b0c"), "align=center;")
    g.text(1330, 48, 110, 58, header("6. TSDF<br>Meshing", "#b85313"), "align=center;")
    g.text(1510, 48, 145, 58, header("7. Phenotype<br>Measurement", "#666666"), "align=center;")

    # Input
    add_photo(g, 18, 132, 108, 118)
    add_photo(g, 18, 270, 108, 118)
    g.text(54, 406, 35, 36, "<b>⋮</b>", "fontSize=24;align=center;")
    add_photo(g, 18, 458, 108, 118)
    g.text(45, 568, 55, 34, "<i>N</i> views", "fontSize=16;fontStyle=2;align=center;")

    # FSAM3 cards
    g.rect(174, 94, 160, 152, "", style_card("#b98ddf", "#fbf7ff"), rounded=True)
    g.text(200, 116, 120, 28, "<b>Quality Filtering</b>", "fontSize=14;fontColor=#5b1aa0;align=center;")
    add_bar_chart(g, 210, 146)
    g.arrow(254, 246, 254, 258, "#8d5abd", width=2)
    g.rect(174, 258, 160, 164, "", style_card("#b98ddf", "#fbf7ff"), rounded=True)
    g.text(188, 280, 130, 24, "<b>Plant Segmentation</b>", "fontSize=14;fontColor=#5b1aa0;align=center;")
    add_mask(g, 200, 316, 108, 92)
    g.arrow(254, 422, 254, 438, "#8d5abd", width=2)
    g.rect(174, 438, 160, 180, "", style_card("#b98ddf", "#fbf7ff"), rounded=True)
    g.text(188, 456, 130, 46, "<b>Main Component<br>Refinement</b>", "fontSize=14;fontColor=#5b1aa0;align=center;")
    add_mask(g, 200, 512, 108, 92)
    g.text(186, 622, 135, 28, "<b>Aligned Foreground<br>Masks</b>", "fontSize=14;fontColor=#5b1aa0;align=center;")

    # COLMAP
    g.rect(382, 94, 160, 174, "", style_card("#91bce9", "#f7fbff"), rounded=True)
    g.text(412, 116, 105, 46, "<b>Camera Pose<br>Estimation</b>", "fontSize=14;align=center;")
    add_points(g, 462, 194, "#e96b5f", n=55, rx=50, ry=38, size=3)
    for cx, cy in [(410, 190), (443, 168), (490, 168), (522, 190), (410, 230), (522, 230)]:
        add_camera(g, cx, cy, 0.55)
    g.arrow(462, 268, 462, 294, "#1764b0", width=2)
    g.rect(382, 294, 160, 164, "", style_card("#91bce9", "#f7fbff"), rounded=True)
    g.text(414, 318, 100, 46, "<b>Sparse Point<br>Triangulation</b>", "fontSize=14;align=center;")
    add_points(g, 462, 398, "#b0b0b0", n=65, rx=60, ry=40, size=4)
    g.arrow(462, 458, 462, 482, "#1764b0", width=2)
    g.rect(382, 482, 160, 166, "", style_card("#91bce9", "#f7fbff"), rounded=True)
    g.text(410, 492, 108, 48, "<b>Foreground Point<br>Filtering</b>", "fontSize=14;align=center;")
    add_points(g, 462, 574, "#68b947", n=70, rx=62, ry=44, size=4)

    # Plant-aware 2DGS
    g.rect(588, 94, 248, 152, "", style_card("#a4c58e", "#fbfff8"), rounded=True)
    g.text(646, 110, 132, 26, "<b>Differentiable Rendering</b>", "fontSize=14;align=center;")
    add_camera(g, 606, 172, 0.58)
    add_camera(g, 800, 172, 0.58)
    add_photo(g, 630, 164, 46, 54)
    add_photo(g, 758, 164, 46, 54)
    add_points(g, 714, 178, "#6bb744", n=90, rx=50, ry=28, size=3)
    g.line([(642, 178), (714, 178), (780, 178)], "endArrow=none;strokeColor=#777777;strokeWidth=1;dashed=1;")
    card_y = [254, 328, 402]
    labels = ["RGB Loss<br>(in Foreground)", "Alpha Mask Loss", "Background Opacity Loss"]
    for yy, lab in zip(card_y, labels):
        g.rect(588, yy, 248, 62, "", style_card("#a4c58e", "#ffffff"), rounded=True)
        g.text(600, yy + 15, 108, 34, lab, "fontSize=15;align=center;")
    add_plant(g, 734, 276, 0.28, pot=False)
    add_mask(g, 778, 270, 42, 34)
    add_mask(g, 724, 344, 38, 32)
    g.arrow(764, 360, 784, 360, "#111111", width=1)
    add_mask(g, 786, 344, 38, 32)
    add_points(g, 650, 430, "#67b842", n=45, rx=34, ry=30, size=3)
    g.arrow(696, 434, 742, 434, "#111111", width=1)
    g.rect(760, 414, 58, 42, "", "rounded=0;fillColor=#eeeeee;strokeColor=#999999;strokeWidth=1;", rounded=False)
    add_plant(g, 790, 428, 0.25, pot=False)
    g.arrow(714, 464, 714, 522, "#3f9b2f", width=2)
    g.rect(588, 522, 248, 148, "", style_card("#a4c58e", "#ffffff"), rounded=True)
    g.text(638, 532, 148, 24, "<b>Optimized Gaussians</b>", "fontSize=14;align=center;")
    add_points(g, 714, 600, "#6ab947", n=150, rx=95, ry=42, size=4)
    add_points(g, 714, 600, "#e8b96e", n=70, rx=86, ry=36, size=5)

    # View weighting
    g.rect(878, 120, 156, 234, "", style_card("#e1b13b", "#fffdf7"), rounded=True)
    g.text(910, 132, 92, 42, "<b>View Quality<br>Assessment</b>", "fontSize=14;fontColor=#a87500;align=center;")
    for i, s in enumerate([5, 3, 2, 1]):
        add_quality_stars(g, 900, 190 + i * 34, s)
    g.text(942, 322, 40, 24, "…", "fontSize=20;align=center;")
    g.arrow(956, 354, 956, 372, "#d5a322", width=2)
    g.rect(878, 372, 156, 186, "", style_card("#e1b13b", "#fffdf7"), rounded=True)
    g.text(922, 386, 72, 42, "<b>Soft Weights<br>(0 ~ 1)</b>", "fontSize=14;fontColor=#a87500;align=center;")
    for yy, frac, lab in [(440, 1.0, "1.00"), (474, 0.73, "0.73"), (508, 0.45, "0.45"), (542, 0.21, "0.21")]:
        add_weight_bar(g, 896, yy, lab, frac)
    g.text(942, 560, 40, 20, "…", "fontSize=18;align=center;")
    g.arrow(956, 558, 956, 576, "#d5a322", width=2)
    g.rect(878, 576, 156, 94, "", style_card("#e1b13b", "#fffdf7"), rounded=True)
    g.text(908, 590, 96, 34, "<b>Weighted Loss<br>Aggregation</b>", "fontSize=14;fontColor=#a87500;align=center;")
    g.text(910, 630, 92, 34, "∑ᵢ wᵢ Lᵢ", "fontSize=26;fontFamily=Times New Roman;align=center;")

    # Pruning
    g.rect(1080, 120, 194, 246, "", style_card("#f2a66d", "#fffaf6"), rounded=True)
    g.text(1122, 144, 112, 42, "<b>Mask-guided<br>Multi-cue Pruning</b>", "fontSize=14;fontColor=#e24b0c;align=center;")
    add_mask(g, 1098, 210, 74, 108)
    add_points(g, 1209, 262, "#73bf4c", n=110, rx=44, ry=62, size=3)
    add_points(g, 1209, 262, "#e58d53", n=45, rx=36, ry=52, size=3)
    g.arrow(1145, 330, 1180, 350, "#ef7b22", width=3)
    g.arrow(1177, 366, 1177, 408, "#ef7b22", width=2)
    g.rect(1080, 408, 194, 256, "", style_card("#f2a66d", "#fffaf6"), rounded=True)
    g.text(1120, 426, 112, 24, "<b>Pruned Gaussians</b>", "fontSize=14;fontColor=#e24b0c;align=center;")
    add_points(g, 1177, 548, "#6ab947", n=190, rx=80, ry=75, size=4)

    # Meshing
    g.rect(1314, 112, 136, 260, "", style_card("#d09060", "#fffaf7"), rounded=True)
    g.text(1342, 132, 76, 24, "<b>TSDF Fusion</b>", "fontSize=14;align=center;")
    add_grid_cube(g, 1320, 166, 124, 158)
    g.arrow(1382, 372, 1382, 394, "#b85313", width=2)
    g.rect(1314, 394, 136, 270, "", style_card("#d09060", "#fffaf7"), rounded=True)
    g.text(1328, 412, 108, 24, "<b>Mesh Extraction</b>", "fontSize=14;align=center;")
    add_plant(g, 1382, 516, 0.8, pot=True, mesh=True, gray=True)

    # Phenotyping
    g.rect(1492, 110, 176, 278, "", style_card("#c4c4c4", "#ffffff"), rounded=True)
    g.text(1518, 130, 120, 38, "<b>Plant Mesh<br>(For Measurement)</b>", "fontSize=14;align=center;")
    add_plant(g, 1580, 246, 1.2, pot=True, mesh=True)
    g.rect(1492, 408, 176, 240, "", style_card("#c4c4c4", "#ffffff"), rounded=True)
    g.text(1522, 426, 120, 24, "<b>Virtual Phenotypes</b>", "fontSize=14;align=center;")
    add_measure_icons(g, 1514, 244)

    # Data flow arrows between columns
    g.arrow(140, 340, 163, 340, "#7a5ba8", width=2)
    g.arrow(345, 340, 370, 340, "#1764b0", width=2)
    g.arrow(553, 340, 577, 340, "#1764b0", width=2)
    g.arrow(845, 340, 868, 340, "#2e9d2a", width=2)
    g.arrow(1044, 340, 1069, 340, "#d5a322", width=2)
    g.arrow(1285, 340, 1307, 340, "#e24b0c", width=2)
    g.arrow(1460, 340, 1484, 340, "#b85313", width=2)

    # Feedback/guidance dashed line
    bottom_y = 712
    g.line([(70, 688), (70, bottom_y), (1382, bottom_y), (1382, 682)], "endArrow=block;endFill=1;strokeColor=#9b4adb;strokeWidth=1.5;dashed=1;dashPattern=4 4;")
    for x, color in [(254, "#9b4adb"), (462, "#1764b0"), (714, "#2e9d2a"), (956, "#d5a322"), (1177, "#e24b0c")]:
        g.arrow(x, bottom_y, x, 680, color, dashed=True, width=1.5)

    # Legend box
    g.rect(12, 744, 1664, 90, "", "arcSize=3;fillColor=#FFFFFF;strokeColor=#9e9e9e;strokeWidth=1;dashed=1;dashPattern=6 4;", rounded=True)
    add_camera(g, 43, 779, 0.9)
    g.text(82, 776, 86, 30, "Camera", "fontSize=13;align=left;verticalAlign=middle;")
    add_legend_item(g, 170, 772, "#f5effb", "FSAM3<br>(Preprocessing)", "#9b6fc7")
    add_legend_item(g, 348, 772, "#eef6ff", "COLMAP<br>(SfM)", "#4f8ccc")
    add_legend_item(g, 497, 772, "#f2f9ed", "2DGS Optimization<br>(Plant-aware)", "#78a85d")
    add_legend_item(g, 698, 772, "#fff8e6", "View Weighting<br>(Training Strategy)", "#d5a322")
    add_legend_item(g, 898, 772, "#fff2e8", "Gaussian Pruning<br>(Model Refinement)", "#ed8b42")
    add_legend_item(g, 1124, 772, "#fff4ec", "Meshing<br>(Geometry)", "#c77839")
    add_legend_item(g, 1270, 772, "#f7f7f7", "Phenotyping<br>(Measurement)", "#a5a5a5")
    g.arrow(1460, 776, 1504, 776, "#111111", width=2)
    g.text(1518, 764, 92, 24, "Data Flow", "fontSize=13;align=left;")
    g.arrow(1460, 808, 1504, 808, "#111111", dashed=True, width=2)
    g.text(1518, 796, 140, 24, "Feedback / Guidance", "fontSize=13;align=left;")

    # Caption
    g.text(
        420,
        878,
        850,
        36,
        "<font style='font-size:22px'><b>ForeSplat Overview.</b></font>"
        "<font style='font-size:22px'> From multi-view images to plant-only mesh and virtual phenotypes.</font>",
        "align=left;verticalAlign=middle;",
    )

    g.save(OUT_FILE)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
