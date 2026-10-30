"""
TolSlope - Analisis Stabilitas Lereng Timbunan Jalan Tol (metode Fellenius dan Bishop disederhanakan)
Jalankan lokal : streamlit run app.py
Deploy         : upload app.py dan requirements.txt ke GitHub, lalu deploy di share.streamlit.io
"""
import io

import ezdxf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TolSlope", page_icon="⛰️", layout="wide")


def fmt(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# ----------------------------------------------------------------------------
# Model geometri dan tanah
# Sumbu x: dari sumbu jalan ke kanan. y = 0 adalah muka tanah asli (kaki lereng).
# Timbunan: puncak datar sampai x = B pada y = H, lalu lereng 1V : m H turun ke y = 0.
# Tanah dasar lunak: lapisan setebal D di bawah y = 0, di bawahnya tanah keras (tidak dianalisis).
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
    hF = np.maximum(0.0, -yb)
    Wsoil = b * (p["gE"] * hE + p["gF"] * hF)
    Wq = p["q"] * b * (xm <= p["B"])
    W = Wsoil + Wq
    sinA = (xc - xm) / R
    cosA = np.sqrt(np.maximum(1 - sinA**2, 1e-6))
    inE = yb >= 0
    c = np.where(inE, p["cE"], p["cF"])
    tphi = np.tan(np.radians(np.where(inE, p["phE"], p["phF"])))
    drive = float(np.sum(W * sinA))
    if drive <= 1e-9:
        return None
    ub = p["ru"] * Wsoil  # u * b
    fel = float(np.sum(c * b / cosA + (W * cosA - ub / cosA) * tphi) / drive)
    fs = fel
    for _ in range(60):
        mA = cosA + sinA * tphi / fs
        if np.any(mA <= 0.05):
            return None
        new = float(np.sum((c * b + (W - ub) * tphi) / mA) / drive)
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
    p = dict(zip(PKEYS, pt))
    return search(p, nx, ny, nr, refine)


PKEYS = ["H", "B", "m", "gE", "cE", "phE", "D", "gF", "cF", "phF", "q", "ru"]


# ----------------------------------------------------------------------------
# Gambar
# ----------------------------------------------------------------------------
def arc_points(xc, yc, R, xa, xb, n=120):
    xs = np.linspace(xa, xb, n)
    return xs, yc - np.sqrt(np.maximum(R * R - (xs - xc) ** 2, 0.0))


def plot_result(p, res, target):
    fs, xc, yc, R = res["best"]
    r = circle_fs(p, xc, yc, R, nsl=12)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    toe = p["B"] + p["m"] * p["H"]
    xmin, xmax = min(r["xa"], p["B"] - 6) - 4, toe + 14
    ax.fill([xmin, xmax, xmax, xmin], [-p["D"], -p["D"], 0, 0], color="#c9b79c", label="Tanah dasar lunak")
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
    msp.add_lwpolyline([(xl, 0), (xr, 0), (xr, -p["D"]), (xl, -p["D"])], close=True, dxfattribs={"layer": "TOL_TANAH_DASAR"})
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
# Antarmuka
# ----------------------------------------------------------------------------
st.title("⛰️ TolSlope")
st.caption("Analisis stabilitas lereng timbunan jalan tol di atas tanah lunak dengan metode keseimbangan batas (Fellenius dan Bishop disederhanakan). Atur parameter, tekan Hitung, lalu lihat bidang longsor kritis dan faktor keamanannya.")

with st.sidebar.form("param"):
    st.header("Timbunan")
    H = st.number_input("Tinggi timbunan H (m)", 1.0, 15.0, 5.0, 0.5)
    B = st.number_input("Lebar puncak dari sumbu ke tepi (m)", 3.0, 40.0, 16.8, 0.1)
    m = st.number_input("Lereng timbunan 1 : m (V : H)", 1.0, 4.0, 2.0, 0.25)
    gE = st.number_input("Berat isi γ (kN/m³)", 14.0, 22.0, 18.0, 0.5, key="gE")
    cE = st.number_input("Kohesi c' (kPa)", 0.0, 50.0, 5.0, 1.0, key="cE")
    phE = st.number_input("Sudut geser φ' (°)", 0.0, 45.0, 30.0, 1.0, key="phE")
    st.header("Tanah dasar lunak")
    D = st.number_input("Tebal lapisan lunak D (m)", 1.0, 30.0, 6.0, 0.5)
    gF = st.number_input("Berat isi γ (kN/m³)", 12.0, 20.0, 16.0, 0.5, key="gF")
    cF = st.number_input("Kohesi c atau cu (kPa)", 0.0, 100.0, 30.0, 1.0, key="cF")
    phF = st.number_input("Sudut geser φ (°)", 0.0, 35.0, 0.0, 1.0, key="phF")
    st.header("Beban dan air")
    q = st.number_input("Beban lalu lintas q (kPa)", 0.0, 40.0, 10.0, 1.0)
    ru = st.slider("Rasio tekanan air pori ru", 0.0, 0.6, 0.0, 0.05)
    st.header("Kriteria")
    target = st.number_input("FS minimum yang disyaratkan", 1.0, 2.5, 1.5, 0.05)
    fine = st.checkbox("Pencarian lebih teliti (lebih lambat)", value=False)
    go = st.form_submit_button("🔍 Hitung", use_container_width=True)

P = dict(H=H, B=B, m=m, gE=gE, cE=cE, phE=phE, D=D, gF=gF, cF=cF, phF=phF, q=q, ru=ru)
PT = tuple(float(P[k]) for k in PKEYS)
grid = (22, 16, 28, True) if fine else (15, 11, 20, True)

tab1, tab2 = st.tabs(["📉 Stabilitas lereng", "📈 Pengaruh kemiringan lereng"])

with tab1:
    res = cached_search(PT, *grid)
    if res["best"] is None:
        st.error("Tidak ada bidang longsor yang valid pada rentang pencarian. Coba ubah geometri atau parameter.")
    else:
        fs, xc, yc, R = res["best"]
        fig, r = plot_result(P, res, target)
        ok = fs >= target
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FS Bishop (kritis)", fmt(fs))
        c2.metric("FS Fellenius (lingkaran yang sama)", fmt(r["fel"]))
        c3.metric("FS yang disyaratkan", fmt(target))
        c4.metric("Status", "Memenuhi" if ok else "Tidak memenuhi")
        if ok:
            st.success(f"FS = {fmt(fs)} ≥ {fmt(target)}. Lereng memenuhi kriteria pada kondisi yang dimasukkan.")
        else:
            st.error(f"FS = {fmt(fs)} < {fmt(target)}. Landaikan lereng, kurangi tinggi timbunan, atau perkuat tanah dasar.")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Lingkaran kritis: pusat (x = {fmt(xc)} m; y = {fmt(yc)} m), jari-jari {fmt(R)} m, masuk di x = {fmt(r['xa'])} m, keluar di x = {fmt(r['xb'])} m."
        )
        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Unduh DXF (penampang + bidang longsor)", dxf_bytes(P, res, r), file_name="stabilitas_lereng_tol.dxf", mime="application/dxf")
        det = circle_fs(P, xc, yc, R, nsl=30, detail=True)["slices"]
        d2.download_button("⬇️ Unduh tabel irisan (CSV)", det.to_csv(index=False).encode("utf-8"), file_name="irisan_lingkaran_kritis.csv", mime="text/csv")
        with st.expander("Tabel irisan lingkaran kritis (30 irisan)"):
            st.dataframe(det, hide_index=True)

with tab2:
    st.markdown("FS minimum untuk beberapa kemiringan lereng, dengan parameter lain sama seperti di sidebar. Dipakai untuk menentukan lereng paling curam yang masih memenuhi FS yang disyaratkan.")
    rows = []
    for mm in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        pt = tuple(float(mm) if k == "m" else float(P[k]) for k in PKEYS)
        rr = cached_search(pt, 11, 8, 14, False)
        rows.append((mm, rr["best"][0] if rr["best"] else np.nan))
    sens = pd.DataFrame(rows, columns=["m (1 : m)", "FS Bishop"])
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
        "- **Pencarian:** grid titik pusat lingkaran, jari-jari divariasikan sampai batas lapisan lunak, lalu dihaluskan di sekitar minimum.\n"
        "- **Kriteria FS:** nilai FS minimum diisi sendiri. Studi yang mengutip SNI 8460:2017 memakai antara lain 1,5 (kondisi jangka panjang statis), 1,3 (jangka pendek), dan 1,1 (gempa). "
        "Cek tabel faktor keamanan pada naskah SNI 8460:2017 untuk kasusmu.\n"
        "- **Parameter tanah, tebal lapisan lunak, beban lalu lintas:** nilai awal hanya contoh. Ganti dengan data penyelidikan tanah. "
        "Jangka pendek di tanah lunak memakai kuat geser tak terdrainase (cu, φ = 0). Jangka panjang memakai parameter efektif (c', φ').\n"
        "- **Belum dimodelkan:** beban gempa, muka air tanah rinci, perkuatan (geotekstil, PVD, bored pile), konsolidasi bertahap, dan bidang longsor non-lingkaran. "
        "Hasil bukan pengganti analisis perangkat lunak geoteknik (misalnya Slide, GeoStudio, atau Plaxis)."
    )
