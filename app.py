"""
TolSlope - Analisis Stabilitas Lereng Timbunan Jalan Tol (metode Fellenius dan Bishop disederhanakan)
Jalankan lokal : streamlit run app.py
Deploy         : upload app.py dan requirements.txt ke GitHub, lalu deploy di share.streamlit.io
"""
import io
import json
from datetime import datetime

import ezdxf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.properties import PageSetupProperties

st.set_page_config(page_title="TolSlope", page_icon="⛰️", layout="wide")


def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# Jenis tanah dasar (nilai awal contoh, bisa diubah di sidebar).
# Lunak/Lempung: kuat geser tak terdrainase (c = cu, phi = 0). Keras: parameter efektif.
SOILS = {
    "Lunak": dict(t=6.0, g=15.5, c=20.0, ph=0.0, col="#c9b79c"),
    "Lempung": dict(t=5.0, g=17.0, c=50.0, ph=0.0, col="#a98f6c"),
    "Keras": dict(t=5.0, g=19.0, c=25.0, ph=28.0, col="#7d6a55"),
}


def prep(p, names=None):
    """Susun lapisan tanah dasar aktif (tebal > 0) menjadi array untuk perhitungan."""
    p = dict(p)
    act = [i for i in (1, 2, 3) if p[f"t{i}"] > 0]
    p["Lt"] = np.array([p[f"t{i}"] for i in act], float)
    p["Lg"] = np.array([p[f"g{i}"] for i in act], float)
    p["Lc"] = np.array([p[f"c{i}"] for i in act], float)
    p["Lph"] = np.array([p[f"ph{i}"] for i in act], float)
    p["Ln"] = [(names[i - 1] if names else f"Lapisan {i}") for i in act]
    p["Lz"] = np.cumsum(p["Lt"])  # kedalaman dasar tiap lapisan
    p["D"] = float(p["Lz"][-1])
    return p


# ----------------------------------------------------------------------------
# Model geometri dan tanah
# Sumbu x: dari sumbu jalan ke kanan. y = 0 adalah muka tanah asli (kaki lereng).
# Timbunan: puncak datar sampai x = B pada y = H, lalu lereng 1V : m H turun ke y = 0.
# Tanah dasar: 1 sampai 3 lapisan (lunak, lempung, keras) di bawah y = 0 dengan total tebal D;
# di bawahnya dianggap batuan/tanah sangat keras (tidak dianalisis).
# ----------------------------------------------------------------------------
def surface_y(x, p):
    xp = [-1000.0, p["B"], p["B"] + p["m"] * p["H"], 1000.0]
    yp = [p["H"], p["H"], 0.0, 0.0]
    return np.interp(x, xp, yp)


def circle_fs(p, xc, yc, R, nsl=50, detail=False):
    """FS Bishop (dan Fellenius) untuk satu lingkaran. Mengembalikan None bila lingkaran tidak valid."""
    xs = np.linspace(xc - R, xc + R, 401)
    yarc = yc - np.sqrt(np.maximum(R * R - (xs - xc) ** 2, 0.0))
    d = surface_y(xs, p) - yarc
    idx = np.nonzero(d > 0)[0]
    if len(idx) < 2:
        return None
    i0, i1 = idx[0], idx[-1]
    if i0 == 0 or i1 == len(xs) - 1 or not np.all(d[i0 : i1 + 1] > 0):
        return None
    xa = xs[i0 - 1] + (0 - d[i0 - 1]) * (xs[i0] - xs[i0 - 1]) / (d[i0] - d[i0 - 1])
    xb = xs[i1] + (0 - d[i1]) * (xs[i1 + 1] - xs[i1]) / (d[i1 + 1] - d[i1])
    if xb <= p["B"] + 1e-6 or xb - xa < 1.0:
        return None  # bidang longsor harus keluar di lereng atau di kaki
    edges = np.linspace(xa, xb, nsl + 1)
    xm = (edges[:-1] + edges[1:]) / 2
    b = (xb - xa) / nsl
    yb = yc - np.sqrt(R * R - (xm - xc) ** 2)
    ys = surface_y(xm, p)
    hE = np.maximum(0.0, ys - np.maximum(yb, 0.0))
    depth = np.maximum(0.0, -yb)
    top = p["Lz"] - p["Lt"]
    wF = np.sum(p["Lg"] * np.clip(depth[:, None] - top, 0.0, p["Lt"]), axis=1)  # tegangan overburden tanah dasar
    Wsoil = b * (p["gE"] * hE + wF)
    Wq = p["q"] * b * (xm <= p["B"])
    W = Wsoil + Wq
    sinA = (xc - xm) / R
    cosA = np.sqrt(np.maximum(1 - sinA**2, 1e-6))
    inE = yb >= 0
    idx = np.minimum(np.searchsorted(p["Lz"], depth, side="left"), len(p["Lz"]) - 1)
    c = np.where(inE, p["cE"], p["Lc"][idx])
    tphi = np.tan(np.radians(np.where(inE, p["phE"], p["Lph"][idx])))
    drive = float(np.sum(W * sinA))
    if drive <= 1e-9:
        return None
    ub = p["ru"] * Wsoil  # u * b
    fel = float(np.sum(c * b / cosA + (W * cosA - ub / cosA) * tphi) / drive)
    if not np.isfinite(fel):
        return None
    fs = fel
    if fel > 1e-9:  # bila kuat geser nol, FS = 0 dan iterasi Bishop dilewati (hindari bagi nol)
        for _ in range(60):
            mA = cosA + sinA * tphi / fs
            if np.any(mA <= 0.05):
                return None
            new = float(np.sum((c * b + (W - ub) * tphi) / mA) / drive)
            if not np.isfinite(new) or new <= 0:
                return None
            if abs(new - fs) < 1e-5:
                fs = new
                break
            fs = new
    out = dict(fs=fs, fel=fel, xa=xa, xb=xb, ya=float(surface_y(xa, p)), yb_=float(surface_y(xb, p)))
    if detail:
        out["slices"] = pd.DataFrame(
            {
                "x tengah (m)": xm,
                "lebar b (m)": b,
                "berat W (kN/m)": W,
                "α (°)": np.degrees(np.arcsin(sinA)),
                "c (kPa)": c,
                "φ (°)": np.degrees(np.arctan(tphi)),
                "W sin α": W * sinA,
            }
        ).round(3)
    return out


def search(p, nx=15, ny=11, nr=20, refine=True):
    """Cari lingkaran kritis (FS Bishop minimum) dengan grid pusat lingkaran."""
    H, m, B, D = p["H"], p["m"], p["B"], p["D"]
    toe = B + m * H
    xcs = np.linspace(B - 1.5 * H, toe + 0.5 * H, nx)
    ycs = np.linspace(0.9 * H, 2.5 * H + 0.8 * D, ny)
    grid = np.full((ny, nx), np.nan)
    best = None

    def scan(xc, yc, n):
        res = None
        for ybot in np.linspace(-D, 0.8 * H, n):
            R = yc - ybot
            if R < 1.0:
                continue
            r = circle_fs(p, xc, yc, R)
            if r is not None and (res is None or r["fs"] < res[0]):
                res = (r["fs"], R)
        return res

    for j, yc in enumerate(ycs):
        for i, xc in enumerate(xcs):
            r = scan(xc, yc, nr)
            if r is not None:
                grid[j, i] = r[0]
                if best is None or r[0] < best[0]:
                    best = (r[0], xc, yc, r[1])
    if best is None:
        return dict(best=None, xcs=xcs, ycs=ycs, grid=grid)
    if refine:
        dx, dy = xcs[1] - xcs[0], ycs[1] - ycs[0]
        for xc in np.linspace(best[1] - dx, best[1] + dx, 7):
            for yc in np.linspace(best[2] - dy, best[2] + dy, 7):
                r = scan(xc, yc, 36)
                if r is not None and r[0] < best[0]:
                    best = (r[0], xc, yc, r[1])
    return dict(best=best, xcs=xcs, ycs=ycs, grid=grid)


@st.cache_data(show_spinner="Mencari bidang longsor kritis...")
def cached_search(pt, nx, ny, nr, refine):
    p = prep(dict(zip(PKEYS, pt)))
    return search(p, nx, ny, nr, refine)


PKEYS = ["H", "B", "m", "gE", "cE", "phE", "q", "ru"] + [f"{k}{i}" for i in (1, 2, 3) for k in ("t", "g", "c", "ph")]


# ----------------------------------------------------------------------------
# Gambar
# ----------------------------------------------------------------------------
def arc_points(xc, yc, R, xa, xb, n=120):
    xs = np.linspace(xa, xb, n)
    return xs, yc - np.sqrt(np.maximum(R * R - (xs - xc) ** 2, 0.0))


def plot_result(p, res, target):
    fs, xc, yc, R = res["best"]
    r = circle_fs(p, xc, yc, R)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    toe = p["B"] + p["m"] * p["H"]
    xmin, xmax = min(r["xa"], p["B"] - 6) - 4, toe + 14
    z0, seen = 0.0, set()
    for k, (t_, nm) in enumerate(zip(p["Lt"], p["Ln"])):
        z1 = z0 + float(t_)
        ax.fill([xmin, xmax, xmax, xmin], [-z1, -z1, -z0, -z0], color=SOILS.get(nm, {}).get("col", "#c9b79c"), label=None if nm in seen else f"Tanah dasar {nm.lower()}")
        seen.add(nm)
        if k > 0:
            ax.plot([xmin, xmax], [-z0, -z0], color="#6b4f2a", lw=0.6, ls="--")
        ax.text(xmin + 0.5, -(z0 + z1) / 2, f"{nm} ({fmt(t_, 1)} m)", va="center", fontsize=8, color="#222")
        z0 = z1
    ax.fill([xmin, xmin, p["B"], toe, xmin], [0, p["H"], p["H"], 0, 0], color="#d9d3c3", ec="#6b4f2a", lw=1.2, label="Timbunan")
    ax.axhline(-p["D"], color="#555", lw=1.0)
    ax.text(xmax - 1, -p["D"] - 0.2, "tanah keras", ha="right", va="top", fontsize=8, color="#555")
    ax.plot([xmin, xmax], [0, 0], color="#6b4f2a", lw=1)
    if p["q"] > 0:
        ax.annotate("", xy=(xmin + 2, p["H"] + 0.05), xytext=(p["B"], p["H"] + 0.05), arrowprops=dict(arrowstyle="-", color="#c0392b"))
        ax.text((xmin + 2 + p["B"]) / 2, p["H"] + 0.3, f"q = {fmt(p['q'], 0)} kPa (lalu lintas)", ha="center", fontsize=8, color="#c0392b")
    xs, ys = arc_points(xc, yc, R, r["xa"], r["xb"])
    ax.plot(xs, ys, color="#d62728", lw=2.2, label=f"Bidang longsor kritis, FS = {fmt(fs)}")
    for xv in np.linspace(r["xa"], r["xb"], 14):
        yb = yc - np.sqrt(max(R * R - (xv - xc) ** 2, 0))
        ax.plot([xv, xv], [yb, float(surface_y(xv, p))], color="#d62728", lw=0.5, alpha=0.5)
    ax.plot([xc], [yc], marker="+", color="#d62728", ms=11)
    ax.plot([xc, r["xa"]], [yc, r["ya"]], color="#d62728", lw=0.6, ls=":")
    ax.plot([xc, r["xb"]], [yc, r["yb_"]], color="#d62728", lw=0.6, ls=":")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-p["D"] - 1.5, max(yc + 1.0, p["H"] + 4))
    ax.set_aspect("equal")
    ax.set_xlabel("Jarak dari sumbu jalan (m)")
    ax.set_ylabel("Elevasi relatif terhadap tanah asli (m)")
    ax.set_title("Penampang dan bidang longsor kritis", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=0.2)
    X, Y = np.meshgrid(res["xcs"], res["ycs"])
    top = max(2.2 * fs, target * 1.3)
    G = np.ma.masked_invalid(np.minimum(res["grid"], top))
    levels = np.linspace(min(fs, target) * 0.98, top, 14)
    cs = ax2.contourf(X, Y, G, levels=levels, cmap="RdYlGn", vmin=levels[0], vmax=levels[-1])
    fig.colorbar(cs, ax=ax2, label="FS minimum per titik pusat")
    if np.nanmin(res["grid"]) < target < np.nanmax(G):
        ax2.contour(X, Y, G, levels=[target], colors="black", linewidths=1.2, linestyles="--")
    ax2.plot([xc], [yc], marker="*", color="black", ms=14)
    ax2.set_xlabel("Pusat lingkaran x (m)")
    ax2.set_ylabel("Pusat lingkaran y (m)")
    ax2.set_title(f"Peta FS pusat lingkaran (garis putus: FS = {fmt(target)})", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig, r


# ----------------------------------------------------------------------------
# Ekspor DXF
# ----------------------------------------------------------------------------
def dxf_bytes(p, res, r):
    fs, xc, yc, R = res["best"]
    doc = ezdxf.new("R2010", setup=True)
    doc.units = 6
    for name, color in [("TOL_TIMBUNAN", 8), ("TOL_TANAH_DASAR", 30), ("TOL_BIDANG_LONGSOR", 1), ("TOL_TEKS", 7)]:
        doc.layers.add(name, color=color)
    msp = doc.modelspace()
    toe = p["B"] + p["m"] * p["H"]
    xl = r["xa"] - 5
    xr = toe + 15
    msp.add_lwpolyline([(xl, 0), (xl, p["H"]), (p["B"], p["H"]), (toe, 0)], close=True, dxfattribs={"layer": "TOL_TIMBUNAN"})
    z0 = 0.0
    for t_, nm in zip(p["Lt"], p["Ln"]):
        z1 = z0 + float(t_)
        msp.add_lwpolyline([(xl, -z0), (xr, -z0), (xr, -z1), (xl, -z1)], close=True, dxfattribs={"layer": "TOL_TANAH_DASAR"})
        tx = msp.add_text(f"{nm} ({t_:.1f} m)", height=0.5, dxfattribs={"layer": "TOL_TEKS"})
        tx.set_placement((xl + 0.5, -(z0 + z1) / 2))
        z0 = z1
    a0 = np.degrees(np.arctan2(r["ya"] - yc, r["xa"] - xc)) % 360
    a1 = np.degrees(np.arctan2(r["yb_"] - yc, r["xb"] - xc)) % 360
    msp.add_arc((float(xc), float(yc)), float(R), float(a0), float(a1), dxfattribs={"layer": "TOL_BIDANG_LONGSOR"})
    msp.add_circle((float(xc), float(yc)), 0.3, dxfattribs={"layer": "TOL_BIDANG_LONGSOR"})
    t = msp.add_text(f"FS (Bishop) = {fs:.2f}   pusat ({xc:.2f}; {yc:.2f})   R = {R:.2f} m", height=0.6, dxfattribs={"layer": "TOL_TEKS"})
    t.set_placement((xl, p["H"] + 2.5))
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


# ----------------------------------------------------------------------------
# Ekspor Excel (laporan rapi: Ringkasan, Irisan, Sensitivitas, Penampang)
# ----------------------------------------------------------------------------
def excel_bytes(P0, P, res, r, det, sens, target, png):
    fs, xc, yc, R = res["best"]
    A = "Arial"
    fn, fb = Font(name=A, size=10), Font(name=A, size=10, bold=True)
    fin = Font(name=A, size=10, color="0000FF")  # biru = data masukan
    fh = Font(name=A, size=10, bold=True, color="FFFFFF")
    fill_h, fill_s = PatternFill("solid", fgColor="1F3A5F"), PatternFill("solid", fgColor="DCE6F1")
    thin = Side(style="thin", color="BFBFBF")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    cen = Alignment(horizontal="center", vertical="center", wrap_text=True)
    wb = Workbook()

    def put(ws, row, col, val, font=fn, fmt=None, al=None, border=True):
        c = ws.cell(row=row, column=col, value=val)
        c.font = font
        if fmt:
            c.number_format = fmt
        if al:
            c.alignment = al
        if border:
            c.border = box
        return c

    def title(ws, text, sub):
        ws["A1"], ws["A2"] = text, sub
        ws["A1"].font = Font(name=A, size=15, bold=True, color="1F3A5F")
        ws["A2"].font = Font(name=A, size=9, italic=True, color="595959")
        ws.sheet_view.showGridLines = False
        ws.page_setup.orientation, ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = "landscape", 1, 0
        ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)

    def section(ws, row, text, n):
        for j in range(1, n + 1):
            ws.cell(row=row, column=j).fill = fill_s
        ws.cell(row=row, column=1, value=text).font = fb

    def header(ws, row, labels):
        for j, t in enumerate(labels, 1):
            c = put(ws, row, j, t, font=fh, al=cen)
            c.fill = fill_h
        ws.row_dimensions[row].height = 30

    def widths(ws, ws_w):
        for col, w in zip("ABCDEFGHIJ", ws_w):
            ws.column_dimensions[col].width = w

    ok_style = dict(fill=PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"), font=Font(name=A, bold=True, color="006100"))
    bad_style = dict(fill=PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"), font=Font(name=A, bold=True, color="9C0006"))
    stamp = f"TolSlope · metode keseimbangan batas (Fellenius dan Bishop disederhanakan) · dibuat {datetime.now():%d-%m-%Y %H:%M}"

    # ---------------- Ringkasan ----------------
    ws = wb.active
    ws.title = "Ringkasan"
    title(ws, "Laporan Analisis Stabilitas Lereng Timbunan Jalan Tol", stamp)
    widths(ws, [38, 22, 18, 18, 14, 16, 12])
    row = 4
    section(ws, row, "HASIL ANALISIS", 7)
    header(ws, row + 1, ["Parameter", "Nilai", "Satuan"])
    r_fs, r_fel, r_t, r_st, r_d = row + 2, row + 3, row + 4, row + 5, row + 6
    put(ws, r_fs, 1, "FS Bishop (lingkaran kritis)", fb); put(ws, r_fs, 2, float(fs), fb, "0.00", cen); put(ws, r_fs, 3, "–", al=cen)
    put(ws, r_fel, 1, "FS Fellenius (lingkaran yang sama)"); put(ws, r_fel, 2, float(r["fel"]), fn, "0.00", cen); put(ws, r_fel, 3, "–", al=cen)
    put(ws, r_t, 1, "FS yang disyaratkan"); put(ws, r_t, 2, float(target), fin, "0.00", cen); put(ws, r_t, 3, "–", al=cen)
    put(ws, r_st, 1, "Status", fb); put(ws, r_st, 2, f'=IF(B{r_fs}>=B{r_t},"MEMENUHI","TIDAK MEMENUHI")', fb, None, cen); put(ws, r_st, 3, "–", al=cen)
    put(ws, r_d, 1, "Selisih FS terhadap persyaratan"); put(ws, r_d, 2, f"=B{r_fs}-B{r_t}", fn, "+0.00;-0.00;0.00", cen); put(ws, r_d, 3, "–", al=cen)
    ws.conditional_formatting.add(f"B{r_st}", FormulaRule(formula=[f'$B${r_st}="MEMENUHI"'], **ok_style))
    ws.conditional_formatting.add(f"B{r_st}", FormulaRule(formula=[f'$B${r_st}="TIDAK MEMENUHI"'], **bad_style))
    ws.conditional_formatting.add(f"B{r_fs}", FormulaRule(formula=[f"$B${r_fs}>=$B${r_t}"], **ok_style))
    ws.conditional_formatting.add(f"B{r_fs}", FormulaRule(formula=[f"$B${r_fs}<$B${r_t}"], **bad_style))

    row = r_d + 2
    section(ws, row, "TIMBUNAN, BEBAN, DAN AIR PORI", 7)
    header(ws, row + 1, ["Parameter", "Nilai", "Satuan"])
    items = [("Tinggi timbunan H", P0["H"], "m", "0.0"), ("Lebar puncak dari sumbu ke tepi B", P0["B"], "m", "0.0"),
             ("Lereng timbunan 1 : m (V : H)", P0["m"], "–", "0.00"), ("Berat isi timbunan γ", P0["gE"], "kN/m³", "0.0"),
             ("Kohesi timbunan c'", P0["cE"], "kPa", "0.0"), ("Sudut geser timbunan φ'", P0["phE"], "°", "0.0"),
             ("Beban lalu lintas q", P0["q"], "kPa", "0.0"), ("Rasio tekanan air pori ru", P0["ru"], "–", "0.00")]
    for k, (nm, val, un, fmt_) in enumerate(items):
        rr = row + 2 + k
        put(ws, rr, 1, nm); put(ws, rr, 2, float(val), fin, fmt_, cen); put(ws, rr, 3, un, al=cen)

    row = row + 2 + len(items) + 1
    section(ws, row, "TANAH DASAR (dari lapisan paling atas)", 7)
    header(ws, row + 1, ["Lapisan", "Tebal (m)", "Kedalaman atas (m)", "Kedalaman bawah (m)", "γ (kN/m³)", "c atau cu (kPa)", "φ (°)"])
    first = row + 2
    for k, (nm, t_, g_, c_, p_) in enumerate(zip(P["Ln"], P["Lt"], P["Lg"], P["Lc"], P["Lph"])):
        rr = first + k
        put(ws, rr, 1, f"Lapisan {k + 1}: {nm}")
        put(ws, rr, 2, float(t_), fin, "0.0", cen)
        put(ws, rr, 3, 0 if k == 0 else f"=D{rr - 1}", fn, "0.0", cen)
        put(ws, rr, 4, f"=C{rr}+B{rr}", fn, "0.0", cen)
        put(ws, rr, 5, float(g_), fin, "0.0", cen); put(ws, rr, 6, float(c_), fin, "0.0", cen); put(ws, rr, 7, float(p_), fin, "0.0", cen)
    last = first + len(P["Lt"]) - 1
    put(ws, last + 1, 1, "Total tebal dianalisis", fb); put(ws, last + 1, 2, f"=SUM(B{first}:B{last})", fb, "0.0", cen)
    for j in range(3, 8):
        put(ws, last + 1, j, None)

    row = last + 3
    section(ws, row, "BIDANG LONGSOR KRITIS (LINGKARAN)", 7)
    header(ws, row + 1, ["Parameter", "Nilai", "Satuan"])
    for k, (nm, val) in enumerate([("Pusat lingkaran x", xc), ("Pusat lingkaran y", yc), ("Jari-jari R", R), ("Titik masuk (x)", r["xa"]), ("Titik keluar (x)", r["xb"])]):
        put(ws, row + 2 + k, 1, nm); put(ws, row + 2 + k, 2, float(val), fn, "0.00", cen); put(ws, row + 2 + k, 3, "m", al=cen)

    row = row + 2 + 5 + 1
    section(ws, row, "CATATAN", 7)
    notes = ["Huruf biru = data masukan, huruf hitam = hasil atau rumus. Ubah angka biru dan rumus di sheet ini ikut menghitung ulang (FS Bishop sendiri berasal dari aplikasi).",
             "Parameter tanah dan beban adalah data yang dimasukkan pengguna di aplikasi. Ganti dengan data penyelidikan tanah proyek. Hasil bukan pengganti perangkat lunak geoteknik.",
             "Nilai FS yang disyaratkan diisi pengguna. Cek SNI 8460:2017 untuk kriteria pada kasus proyekmu."]
    for k, t in enumerate(notes):
        ws.merge_cells(start_row=row + 1 + k, start_column=1, end_row=row + 1 + k, end_column=7)
        c = ws.cell(row=row + 1 + k, column=1, value=t)
        c.font, c.alignment = Font(name=A, size=9, color="404040"), Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[row + 1 + k].height = 28

    # ---------------- Irisan ----------------
    wi = wb.create_sheet("Irisan")
    title(wi, "Tabel Irisan Lingkaran Kritis", f"{len(det)} irisan · FS Bishop = {fs:.2f} · pusat ({xc:.2f}; {yc:.2f}) · R = {R:.2f} m")
    widths(wi, [8, 16, 14, 16, 12, 12, 12, 16])
    header(wi, 4, ["No", "x tengah (m)", "Lebar b (m)", "Berat W (kN/m)", "α (°)", "c (kPa)", "φ (°)", "W sin α (kN/m)"])
    fm = ["0.00", "0.00", "0.0", "0.0", "0.0", "0.0"]
    for k, rowv in enumerate(det.itertuples(index=False)):
        rr = 5 + k
        put(wi, rr, 1, k + 1, al=cen)
        for j in range(6):
            put(wi, rr, 2 + j, float(rowv[j]), fn, fm[j], cen)
        put(wi, rr, 8, f"=D{rr}*SIN(RADIANS(E{rr}))", fn, "0.0", cen)
    e = 4 + len(det)
    put(wi, e + 1, 1, "Σ", fb, al=cen)
    for j in range(2, 9):
        put(wi, e + 1, j, None)
    put(wi, e + 1, 4, f"=SUM(D5:D{e})", fb, "0.0", cen); put(wi, e + 1, 8, f"=SUM(H5:H{e})", fb, "0.0", cen)
    wi.freeze_panes = "A5"
    wi.print_title_rows = "4:4"

    # ---------------- Sensitivitas ----------------
    wsn = wb.create_sheet("Sensitivitas")
    title(wsn, "Pengaruh Kemiringan Lereng terhadap FS", "Pencarian lebih kasar daripada analisis utama. Parameter lain sama dengan sheet Ringkasan.")
    widths(wsn, [16, 14, 14, 14, 20])
    header(wsn, 4, ["Lereng 1 : m", "FS Bishop", "FS syarat", "Selisih", "Status"])
    n = len(sens)
    for k, (mm, fv) in enumerate(zip(sens["m (1 : m)"], sens["FS Bishop"])):
        rr = 5 + k
        put(wsn, rr, 1, float(mm), fn, "0.0", cen)
        put(wsn, rr, 2, None if pd.isna(fv) else float(fv), fn, "0.00", cen)
        put(wsn, rr, 3, f"=Ringkasan!$B${r_t}", fn, "0.00", cen)
        put(wsn, rr, 4, f'=IF(B{rr}="","",B{rr}-C{rr})', fn, "+0.00;-0.00;0.00", cen)
        put(wsn, rr, 5, f'=IF(B{rr}="","-",IF(B{rr}>=C{rr},"Memenuhi","Tidak memenuhi"))', fn, None, cen)
    rng = f"E5:E{4 + n}"
    wsn.conditional_formatting.add(rng, FormulaRule(formula=['$E5="Memenuhi"'], **ok_style))
    wsn.conditional_formatting.add(rng, FormulaRule(formula=['$E5="Tidak memenuhi"'], **bad_style))
    put(wsn, 6 + n, 1, "Lereng paling curam yang memenuhi", fb, border=False)
    put(wsn, 6 + n, 5, f'=IFERROR(INDEX(A5:A{4 + n},MATCH("Memenuhi",E5:E{4 + n},0)),"tidak ada")', fb, "0.0", cen)
    ch = LineChart()
    ch.title, ch.height, ch.width = "FS Bishop vs kemiringan lereng", 8, 16
    ch.y_axis.title, ch.x_axis.title = "FS", "Lereng 1 : m"
    ch.add_data(Reference(wsn, min_col=2, max_col=3, min_row=4, max_row=4 + n), titles_from_data=True)
    ch.set_categories(Reference(wsn, min_col=1, min_row=5, max_row=4 + n))
    ch.x_axis.delete = ch.y_axis.delete = False
    lo = min([float(target)] + [float(v) for v in sens["FS Bishop"] if pd.notna(v)])
    ch.y_axis.scaling.min = max(0.0, np.floor(lo * 10) / 10 - 0.1)
    wsn.add_chart(ch, "G4")

    # ---------------- Penampang ----------------
    wp = wb.create_sheet("Penampang")
    title(wp, "Penampang dan Bidang Longsor Kritis", f"FS Bishop = {fs:.2f}")
    img = XLImage(io.BytesIO(png))
    k_ = 1000 / img.width
    img.width, img.height = int(img.width * k_), int(img.height * k_)
    wp.add_image(img, "A4")

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


# ----------------------------------------------------------------------------
# Animasi (kanvas HTML/JS mandiri): pencarian lingkaran kritis dan simulasi longsor
# ----------------------------------------------------------------------------
ANIM_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{margin:0;font:13px Arial,sans-serif;color:#1d1d1b;background:#fff}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:6px 2px}
button,select{font:inherit;padding:5px 10px;border:1px solid #cfcfc8;border-radius:6px;background:#fff;color:#1d1d1b;cursor:pointer}
button.on{background:#1f3a5f;color:#fff;border-color:#1f3a5f}
canvas{width:100%;display:block;border:1px solid #dcdad2;border-radius:8px;background:#eef4fa}
input[type=range]{flex:1;min-width:120px}.note{color:#6b6a64;font-size:12px;padding:2px}
</style></head><body>
<div class="bar"><button id="m1" class="on">1 · Pencarian lingkaran kritis</button><button id="m2">2 · Simulasi longsor</button><span style="flex:1"></span>
<button id="pp">⏸ Jeda</button><button id="rs">⟲ Ulang</button><select id="sp"><option value="0.5">0,5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></div>
<canvas id="cv"></canvas>
<div class="bar"><input type="range" id="sl" min="0" max="1000" value="0"></div>
<div class="note" id="nt"></div>
<script>
const D=__DATA__;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
let mode=1,playing=true,speed=1,t=0,last=null,W=0,Hh=0,S=1,ox=0,oy=0,drag=false;
const T1=14,T2=9,TR=0.35,N=D.trials.length;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v)),lerp=(a,b,u)=>a+(b-a)*u,ease=u=>u*u*(3-2*u);
const f2=x=>x.toFixed(2).replace('.',',');
const order=D.trials.map((c,i)=>i).sort((a,b)=>D.trials[a].fs-D.trials[b].fs);
const rank=new Array(N);order.forEach((idx,k)=>rank[idx]=k/Math.max(1,N-1));
const thMax=clamp(0.18+0.8*(1-D.fs),0.2,0.5);
function rot(p,th){const dx=p[0]-D.xc,dy=p[1]-D.yc,c=Math.cos(th),s=Math.sin(th);return [D.xc+dx*c-dy*s,D.yc+dx*s+dy*c]}
let BD=[D.xmin,D.xmax,-D.D-1.2,Math.max(D.yc+2,D.H+3)];
if(D.fs<1){D.mass.forEach(p=>{const q=rot(p,thMax);BD[1]=Math.max(BD[1],q[0]+2);BD[0]=Math.min(BD[0],q[0]-2);BD[3]=Math.max(BD[3],q[1]+1)})}
const sx=x=>ox+S*x,sy=y=>oy-S*y;
function layout(){const dpr=window.devicePixelRatio||1,w=cv.parentElement.clientWidth,wr=BD[1]-BD[0],hr=BD[3]-BD[2];
 const h=Math.round(Math.min(w*hr/wr,500));cv.style.height=h+'px';cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 W=w;Hh=h;S=Math.min(w/wr,h/hr);ox=(w-S*wr)/2-S*BD[0];oy=(h+S*hr)/2+S*BD[2]}
function addPoly(pts){pts.forEach((p,i)=>i?ctx.lineTo(sx(p[0]),sy(p[1])):ctx.moveTo(sx(p[0]),sy(p[1])));ctx.closePath()}
function strokeArc(pts,prog,col,lw){const n=Math.max(2,Math.round(pts.length*prog));ctx.beginPath();pts.slice(0,n).forEach((p,i)=>i?ctx.lineTo(sx(p[0]),sy(p[1])):ctx.moveTo(sx(p[0]),sy(p[1])));ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.lineJoin='round';ctx.stroke()}
function terrain(){const x0=sx(BD[0]-6),ww=sx(BD[1]+6)-x0;
 ctx.fillStyle='#a9a49a';ctx.fillRect(x0,sy(-D.D),ww,sy(BD[2]-2)-sy(-D.D));
 D.layers.forEach(L=>{ctx.fillStyle=L.col;ctx.fillRect(x0,sy(-L.z0),ww,sy(-L.z1)-sy(-L.z0))});
 ctx.beginPath();addPoly(D.emb);ctx.fillStyle='#d9d3c3';ctx.fill();ctx.strokeStyle='#6b4f2a';ctx.lineWidth=1.2;ctx.stroke();
 ctx.beginPath();ctx.moveTo(x0,sy(0));ctx.lineTo(x0+ww,sy(0));ctx.strokeStyle='#6b4f2a';ctx.lineWidth=1;ctx.stroke()}
function labels(){ctx.font='11px Arial';ctx.fillStyle='#222';ctx.textBaseline='middle';
 D.layers.forEach(L=>ctx.fillText(L.name+' ('+L.t.toFixed(1).replace('.',',')+' m)',sx(BD[0])+8,sy(-(L.z0+L.z1)/2)));
 ctx.fillStyle='#fff';ctx.fillText('tanah keras (tidak dianalisis)',sx(BD[0])+8,sy(-D.D-0.6))}
function fsCol(fs,a){const h=120*clamp((fs-1)/(Math.max(D.target*1.3,1.3)-1),0,1);return 'hsla('+h+',75%,42%,'+a+')'}
function chip(txt,bg,y){ctx.font='bold 13px Arial';const w=ctx.measureText(txt).width+18;ctx.fillStyle=bg;ctx.beginPath();ctx.roundRect(10,y,w,26,6);ctx.fill();ctx.fillStyle='#fff';ctx.textBaseline='middle';ctx.fillText(txt,19,y+13)}
function info(txt,y){ctx.font='12px Arial';const w=ctx.measureText(txt).width+16;ctx.fillStyle='rgba(255,255,255,.88)';ctx.beginPath();ctx.roundRect(10,y,w,22,5);ctx.fill();ctx.fillStyle='#1d1d1b';ctx.textBaseline='middle';ctx.fillText(txt,18,y+11)}
function legend(){const x=W-170,y=Hh-26;for(let i=0;i<120;i++){ctx.fillStyle=fsCol(1+i/120*(Math.max(D.target*1.3,1.3)-1),1);ctx.fillRect(x+i,y,1.5,8)}
 ctx.font='10px Arial';ctx.fillStyle='#333';ctx.textBaseline='alphabetic';ctx.fillText('FS ≤ 1',x,y-3);ctx.textAlign='right';ctx.fillText('FS ≥ '+f2(Math.max(D.target*1.3,1.3)),x+122,y-3);ctx.textAlign='left'}
function drawM1(){
 ctx.fillStyle='#eef4fa';ctx.fillRect(0,0,W,Hh);terrain();labels();
 const ph2=clamp((t-7)/3,0,1),ph3=clamp((t-10)/3,0,1),keep=lerp(1,0.15,ease(ph2))-ph3*0.5;let shown=0,mn=1e9;
 for(let i=0;i<N;i++){const ta=0.5+i/N*6.2;if(t<ta)continue;shown++;mn=Math.min(mn,D.trials[i].fs);
  const a=0.6*clamp(1-(rank[i]-keep)*12,0,1);if(a<0.02)continue;strokeArc(D.trials[i].pts,clamp((t-ta)/TR,0,1),fsCol(D.trials[i].fs,a),1.6)}
 const u=clamp((t-9.5)/1.5,0,1);
 if(u>0){strokeArc(D.crit.pts,1,'rgba(214,39,40,'+u+')',lerp(2,5,u));
  ctx.save();ctx.globalAlpha=u;ctx.setLineDash([5,4]);ctx.strokeStyle='#d62728';ctx.lineWidth=1;ctx.beginPath();
  const a=D.crit.pts[0],b=D.crit.pts[D.crit.pts.length-1];ctx.moveTo(sx(a[0]),sy(a[1]));ctx.lineTo(sx(D.xc),sy(D.yc));ctx.lineTo(sx(b[0]),sy(b[1]));ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle='#d62728';ctx.beginPath();ctx.arc(sx(D.xc),sy(D.yc),4+2*Math.sin(t*6),0,7);ctx.fill();ctx.restore()}
 legend();
 if(t<7)info('Mencoba lingkaran ke-'+shown+' dari '+N+' · FS terendah sejauh ini: '+(shown?f2(mn):'–'),10);
 else if(t<10)info('Menyaring: hanya lingkaran dengan FS terendah yang dipertahankan',10);
 else chip('Lingkaran kritis ditemukan · FS = '+f2(D.fs),D.fs>=D.target?'#1e7d46':'#c0392b',10)}
function drawM2(){
 let th=0;
 if(D.fs<1)th=thMax*ease(clamp((t-1)/5,0,1));
 else if(D.fs<D.target)th=0.018*Math.sin(t*9)*(0.6+0.4*Math.sin(t*1.3));
 ctx.fillStyle='#eef4fa';ctx.fillRect(0,0,W,Hh);
 ctx.save();ctx.beginPath();ctx.rect(0,0,W,Hh);addPoly(D.mass);ctx.clip('evenodd');terrain();ctx.restore();
 ctx.beginPath();addPoly(D.mass);ctx.fillStyle='#8f8a7e';ctx.fill();
 ctx.save();ctx.translate(sx(D.xc),sy(D.yc));ctx.rotate(-th);ctx.translate(-sx(D.xc),-sy(D.yc));
 ctx.beginPath();addPoly(D.mass);ctx.save();ctx.clip();terrain();ctx.restore();ctx.beginPath();addPoly(D.mass);ctx.strokeStyle='#6b4f2a';ctx.lineWidth=1.2;ctx.stroke();ctx.restore();
 labels();strokeArc(D.crit.pts,1,'#d62728',3);
 ctx.fillStyle='#d62728';ctx.beginPath();ctx.arc(sx(D.xc),sy(D.yc),3.5,0,7);ctx.fill();
 if(D.fs<1)chip('LONGSOR · FS = '+f2(D.fs)+' < 1,00','#c0392b',10);
 else if(D.fs<D.target)chip('RAWAN · FS = '+f2(D.fs)+' di bawah syarat '+f2(D.target),'#d68910',10);
 else chip('STABIL · FS = '+f2(D.fs)+' ≥ '+f2(D.target),'#1e7d46',10);
 info('Ilustrasi gerak rotasi massa di atas bidang longsor kritis',42)}
function render(){mode===1?drawM1():drawM2()}
function loop(ts){if(last===null)last=ts;const dt=Math.min(0.05,(ts-last)/1000);last=ts;
 const T=mode===1?T1:T2;if(playing&&!drag){t+=dt*speed;if(t>=T)t-=T}
 render();if(!drag)document.getElementById('sl').value=Math.round(t/T*1000);requestAnimationFrame(loop)}
const NOTE={1:'Setiap garis adalah satu lingkaran percobaan. Warna menunjukkan FS (merah rawan, hijau aman). Aplikasi menyaring sampai tersisa lingkaran dengan FS terendah.',
 2:'Ilustrasi saja: massa di atas bidang longsor diputar terhadap pusat lingkaran. Ini bukan prediksi kecepatan atau besar pergeseran longsor sebenarnya.'};
function setMode(m){mode=m;t=0;document.getElementById('m1').className=m===1?'on':'';document.getElementById('m2').className=m===2?'on':'';document.getElementById('nt').textContent=NOTE[m]}
document.getElementById('m1').onclick=()=>setMode(1);document.getElementById('m2').onclick=()=>setMode(2);
document.getElementById('pp').onclick=e=>{playing=!playing;e.target.textContent=playing?'⏸ Jeda':'▶ Putar'};
document.getElementById('rs').onclick=()=>{t=0};
document.getElementById('sp').onchange=e=>{speed=parseFloat(e.target.value)};
const sl=document.getElementById('sl');sl.oninput=()=>{drag=true;t=sl.value/1000*(mode===1?T1:T2)};sl.onchange=()=>{drag=false};
window.addEventListener('resize',layout);
window.anim={set:(m,tt,pause)=>{setMode(m);t=tt;if(pause){playing=false}}};
layout();setMode(1);requestAnimationFrame(loop);
</script></body></html>
"""


def trial_circles(p, best, n_far=40, n_near=14, seed=7):
    """Lingkaran percobaan untuk animasi: sebagian acak di seluruh area, sebagian dekat lingkaran kritis."""
    fs0, xc0, yc0, R0 = best
    H, m, B, D = p["H"], p["m"], p["B"], p["D"]
    toe = B + m * H
    rng = np.random.default_rng(seed)
    out, far, near, tries = [], 0, 0, 0

    def add(xc, yc, ybot):
        R = yc - ybot
        if R < 1.0:
            return False
        r = circle_fs(p, xc, yc, R)
        if r is None:
            return False
        xs, ys = arc_points(xc, yc, R, r["xa"], r["xb"], 36)
        out.append(dict(fs=round(float(r["fs"]), 3), pts=[[round(float(a), 2), round(float(b), 2)] for a, b in zip(xs, ys)]))
        return True

    while (far < n_far or near < n_near) and tries < 3000:
        tries += 1
        if far < n_far:
            far += add(rng.uniform(B - 1.5 * H, toe + 0.5 * H), rng.uniform(0.9 * H, 2.5 * H + 0.8 * D), rng.uniform(-D, 0.8 * H))
        if near < n_near:
            sc = max(1.0, 0.6 * H)
            ybot = float(np.clip((yc0 - R0) + rng.normal(0, 0.8), -D, 0.8 * H))
            near += add(xc0 + rng.normal(0, 0.6 * sc), yc0 + rng.normal(0, 0.6 * sc), ybot)
    order = rng.permutation(len(out))
    return [out[i] for i in order]


@st.cache_data(show_spinner="Menyiapkan animasi...")
def cached_trials(pt, best):
    return trial_circles(prep(dict(zip(PKEYS, pt))), best)


def anim_html(p, res, r, target, trials):
    fs, xc, yc, R = res["best"]
    toe = p["B"] + p["m"] * p["H"]
    rd = lambda a: [[round(float(x), 2), round(float(y), 2)] for x, y in a]
    xa, ya = arc_points(xc, yc, R, r["xa"], r["xb"], 60)
    sxs = np.linspace(r["xb"], r["xa"], 60)
    layers, z0 = [], 0.0
    for t_, nm in zip(p["Lt"], p["Ln"]):
        layers.append(dict(z0=z0, z1=z0 + float(t_), t=float(t_), name=nm, col=SOILS.get(nm, {}).get("col", "#c9b79c")))
        z0 += float(t_)
    xmin = min(r["xa"], p["B"] - 6) - 4
    data = dict(
        H=float(p["H"]), D=float(p["D"]), xmin=float(xmin), xmax=float(toe + 14), xc=float(xc), yc=float(yc), fs=float(fs), target=float(target),
        layers=layers, emb=rd([(xmin, 0), (xmin, p["H"]), (p["B"], p["H"]), (toe, 0)]),
        crit=dict(pts=rd(zip(xa, ya))), mass=rd(list(zip(xa, ya)) + list(zip(sxs, surface_y(sxs, p)))), trials=trials,
    )
    return ANIM_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))


# ----------------------------------------------------------------------------
# Antarmuka
# ----------------------------------------------------------------------------
st.title("⛰️ TolSlope")
st.caption("Analisis stabilitas lereng timbunan jalan tol di atas tanah dasar berlapis (lunak, lempung, keras) dengan metode keseimbangan batas (Fellenius dan Bishop disederhanakan). Atur parameter, tekan Hitung, lalu lihat bidang longsor kritis dan faktor keamanannya.")

st.sidebar.header("Tanah dasar")
nl = st.sidebar.radio("Jumlah lapisan tanah dasar", [1, 2, 3], horizontal=True)
types = [
    st.sidebar.selectbox(f"Jenis lapisan {i + 1}" + (" (paling atas)" if i == 0 else ""), list(SOILS), index=i, key=f"jenis{i}")
    for i in range(nl)
]

with st.sidebar.form("param"):
    st.header("Timbunan")
    H = st.number_input("Tinggi timbunan H (m)", 1.0, 15.0, 5.0, 0.5)
    B = st.number_input("Lebar puncak dari sumbu ke tepi (m)", 3.0, 40.0, 16.8, 0.1)
    m = st.number_input("Lereng timbunan 1 : m (V : H)", 1.0, 4.0, 2.0, 0.25)
    gE = st.number_input("Berat isi γ (kN/m³)", 14.0, 22.0, 18.0, 0.5, key="gE")
    cE = st.number_input("Kohesi c' (kPa)", 0.0, 50.0, 5.0, 1.0, key="cE")
    phE = st.number_input("Sudut geser φ' (°)", 0.0, 45.0, 30.0, 1.0, key="phE")
    P0 = dict(H=H, B=B, m=m, gE=gE, cE=cE, phE=phE)
    for i in (1, 2, 3):
        if i <= nl:
            tp = types[i - 1]
            sd = SOILS[tp]
            st.header(f"Lapisan {i}: {tp}")
            P0[f"t{i}"] = st.number_input("Tebal (m)", 0.5, 30.0, sd["t"], 0.5, key=f"t{i}_{tp}")
            P0[f"g{i}"] = st.number_input("Berat isi γ (kN/m³)", 12.0, 22.0, sd["g"], 0.5, key=f"g{i}_{tp}")
            P0[f"c{i}"] = st.number_input("Kohesi c atau cu (kPa)", 0.0, 300.0, sd["c"], 1.0, key=f"c{i}_{tp}")
            P0[f"ph{i}"] = st.number_input("Sudut geser φ (°)", 0.0, 45.0, sd["ph"], 1.0, key=f"ph{i}_{tp}")
        else:
            P0.update({f"t{i}": 0.0, f"g{i}": 16.0, f"c{i}": 0.0, f"ph{i}": 0.0})
    st.header("Beban dan air")
    P0["q"] = st.number_input("Beban lalu lintas q (kPa)", 0.0, 40.0, 10.0, 1.0)
    P0["ru"] = st.slider("Rasio tekanan air pori ru", 0.0, 0.6, 0.0, 0.05)
    st.header("Kriteria")
    target = st.number_input("FS minimum yang disyaratkan", 1.0, 2.5, 1.5, 0.05)
    fine = st.checkbox("Pencarian lebih teliti (lebih lambat)", value=False)
    go = st.form_submit_button("🔍 Hitung")

PT = tuple(float(P0[k]) for k in PKEYS)
P = prep(P0, types)
grid = (22, 16, 28, True) if fine else (15, 11, 20, True)

def sensitivity(P0):
    rows = []
    for mm in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        pt = tuple(float(mm) if k == "m" else float(P0[k]) for k in PKEYS)
        rr = cached_search(pt, 11, 8, 14, False)
        rows.append((mm, rr["best"][0] if rr["best"] else np.nan))
    return pd.DataFrame(rows, columns=["m (1 : m)", "FS Bishop"])


sens = sensitivity(P0)
tab1, tab2, tab3 = st.tabs(["📉 Stabilitas lereng", "📈 Pengaruh kemiringan lereng", "🎬 Animasi"])

with tab1:
    res = cached_search(PT, *grid)
    if res["best"] is None:
        st.error("Tidak ada bidang longsor yang valid pada rentang pencarian. Coba ubah geometri atau parameter.")
    else:
        fs, xc, yc, R = res["best"]
        fig, r = plot_result(P, res, target)
        ok = fs >= target
        st.caption("Tanah dasar: " + " → ".join(f"{n} {fmt(t, 1)} m" for n, t in zip(P["Ln"], P["Lt"])) + f" (total {fmt(P['D'], 1)} m, di bawahnya dianggap keras dan tidak dianalisis)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FS Bishop (kritis)", fmt(fs))
        c2.metric("FS Fellenius (lingkaran yang sama)", fmt(r["fel"]))
        c3.metric("FS yang disyaratkan", fmt(target))
        c4.metric("Status", "Memenuhi" if ok else "Tidak memenuhi")
        if ok:
            st.success(f"FS = {fmt(fs)} ≥ {fmt(target)}. Lereng memenuhi kriteria pada kondisi yang dimasukkan.")
        else:
            st.error(f"FS = {fmt(fs)} < {fmt(target)}. Landaikan lereng, kurangi tinggi timbunan, atau perkuat tanah dasar.")
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Lingkaran kritis: pusat (x = {fmt(xc)} m; y = {fmt(yc)} m), jari-jari {fmt(R)} m, masuk di x = {fmt(r['xa'])} m, keluar di x = {fmt(r['xb'])} m."
        )
        det = (circle_fs(P, xc, yc, R, nsl=30, detail=True) or circle_fs(P, xc, yc, R, detail=True))["slices"]
        d0, d1, d2 = st.columns(3)
        d0.download_button("📊 Unduh laporan Excel (.xlsx)", excel_bytes(P0, P, res, r, det, sens, target, png.getvalue()), file_name="laporan_stabilitas_lereng.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        d1.download_button("⬇️ Unduh DXF", dxf_bytes(P, res, r), file_name="stabilitas_lereng_tol.dxf", mime="application/dxf")
        d2.download_button("⬇️ Unduh irisan (CSV)", det.to_csv(index=False).encode("utf-8"), file_name="irisan_lingkaran_kritis.csv", mime="text/csv")
        with st.expander(f"Tabel irisan lingkaran kritis ({len(det)} irisan)"):
            st.dataframe(det, hide_index=True)

with tab2:
    st.markdown("FS minimum untuk beberapa kemiringan lereng, dengan parameter lain sama seperti di sidebar. Dipakai untuk menentukan lereng paling curam yang masih memenuhi FS yang disyaratkan.")
    fig2, ax3 = plt.subplots(figsize=(8, 3.6))
    ax3.plot(sens["m (1 : m)"], sens["FS Bishop"], marker="o", color="#2b6cb0")
    ax3.axhline(target, color="#c0392b", ls="--", lw=1, label=f"FS minimum = {fmt(target)}")
    ax3.set_xlabel("Kemiringan lereng 1 : m (V : H)")
    ax3.set_ylabel("FS Bishop minimum")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    okm = sens[sens["FS Bishop"] >= target]
    if len(okm):
        st.info(f"Lereng paling curam yang memenuhi pada daftar ini: 1 : {fmt(okm['m (1 : m)'].min(), 1)}. Pencarian di tab ini lebih kasar, jadi periksa lagi di tab Stabilitas untuk nilai yang dipilih.")
    else:
        st.warning("Tidak ada kemiringan pada daftar ini yang memenuhi FS yang disyaratkan. Perlu perbaikan tanah dasar atau perkuatan.")
    st.dataframe(sens.round(3), hide_index=True)

with st.expander("Metode, nilai awal, dan batasan"):
    st.markdown(
        "- **Metode:** keseimbangan batas irisan pada bidang longsor lingkaran. Fellenius: FS = Σ[c·l + (W cos α − u·l) tan φ] / Σ W sin α. "
        "Bishop disederhanakan: FS = Σ[(c·b + (W − u·b) tan φ) / mα] / Σ W sin α, dengan mα = cos α + sin α tan φ / FS, dihitung iterasi.\n"
        "- **Pencarian:** grid titik pusat lingkaran, jari-jari divariasikan sampai dasar lapisan tanah paling bawah, lalu dihaluskan di sekitar minimum.\n"
        "- **Kriteria FS:** nilai FS minimum diisi sendiri. Studi yang mengutip SNI 8460:2017 memakai antara lain 1,5 (kondisi jangka panjang statis), 1,3 (jangka pendek), dan 1,1 (gempa). "
        "Cek tabel faktor keamanan pada naskah SNI 8460:2017 untuk kasusmu.\n"
        "- **Parameter tanah, tebal tiap lapisan, beban lalu lintas:** jenis Lunak, Lempung, dan Keras hanya memberi nilai awal contoh; kamu bisa mengubah semua angkanya.  nilai awal hanya contoh. Ganti dengan data penyelidikan tanah. "
        "Jangka pendek di tanah lunak memakai kuat geser tak terdrainase (cu, φ = 0). Jangka panjang memakai parameter efektif (c', φ').\n"
        "- **Belum dimodelkan:** beban gempa, muka air tanah rinci, perkuatan (geotekstil, PVD, bored pile), konsolidasi bertahap, dan bidang longsor non-lingkaran. "
        "Hasil bukan pengganti analisis perangkat lunak geoteknik (misalnya Slide, GeoStudio, atau Plaxis)."
    )

with tab3:
    if res["best"] is None:
        st.info("Belum ada bidang longsor yang valid untuk dianimasikan. Ubah parameter di sidebar.")
    else:
        fs_a, xc_a, yc_a, R_a = res["best"]
        r_a = circle_fs(P, xc_a, yc_a, R_a)
        trials = cached_trials(PT, (float(fs_a), float(xc_a), float(yc_a), float(R_a)))
        components.html(anim_html(P, res, r_a, target, trials), height=700, scrolling=False)
        st.caption("Animasi bersifat ilustrasi. Data lingkaran dan FS diambil dari hasil analisis di tab pertama dan ikut berubah saat parameter diubah.")
