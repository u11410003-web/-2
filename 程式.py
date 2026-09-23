#火山
import base64
from datetime import datetime
from IPython.display import HTML, Image, display
import ipywidgets as widgets
import numpy as np
import pandas as pd
import pygmt
import xarray as xr

# ==== 1. 設定區域與基本參數 ====
REGION = [135, 150, 10, 25]  # 馬里亞納海溝範圍
CUT_LAT = 16.5
START, MINMAG = "1900-01-01", 5.0
Z_MIN_3D = -200  # 3D 深度下限 (km)
Z_MAX_2D = 700  # 2D 剖面深度 (km)


# ==== 火山資料抓取函數 (支援線上 API 與備用清單) ====
def get_mariana_volcanoes():
  try:
    # 嘗試從 NOAA Hazel API 擷取全區火山資料
    volc_url = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/volcanoes"
    v_df = pd.read_json(volc_url)
    if "items" in v_df:
      v_df = pd.DataFrame(v_df["items"])
    mask = (
        (v_df["longitude"] >= REGION[0])
        & (v_df["longitude"] <= REGION[1])
        & (v_df["latitude"] >= REGION[2])
        & (v_df["latitude"] <= REGION[3])
    )
    m_volc = v_df[mask].copy()
    if len(m_volc) > 0:
      return m_volc[["name", "latitude", "longitude", "elevation"]]
  except Exception:
    pass

  # 備用清單：馬里亞納火山弧主要活火山/海山
  mariana_volcanoes = [
      {
          "name": "Farallon de Pajaros",
          "latitude": 20.54,
          "longitude": 144.89,
          "elevation": 360,
      },
      {
          "name": "Supply Reef",
          "latitude": 20.14,
          "longitude": 145.10,
          "elevation": -8,
      },
      {
          "name": "Maug Islands",
          "latitude": 20.02,
          "longitude": 145.22,
          "elevation": 227,
      },
      {
          "name": "Asuncion",
          "latitude": 19.67,
          "longitude": 145.40,
          "elevation": 857,
      },
      {
          "name": "Agrihan",
          "latitude": 18.77,
          "longitude": 145.67,
          "elevation": 965,
      },
      {
          "name": "Pagan",
          "latitude": 18.13,
          "longitude": 145.80,
          "elevation": 570,
      },
      {
          "name": "Alamagan",
          "latitude": 17.60,
          "longitude": 145.83,
          "elevation": 744,
      },
      {
          "name": "Guguan",
          "latitude": 17.32,
          "longitude": 145.83,
          "elevation": 301,
      },
      {
          "name": "Sarigan",
          "latitude": 16.71,
          "longitude": 145.78,
          "elevation": 538,
      },
      {
          "name": "Anatahan",
          "latitude": 16.35,
          "longitude": 145.67,
          "elevation": 787,
      },
      {
          "name": "Esmeralda Bank",
          "latitude": 15.00,
          "longitude": 145.25,
          "elevation": -43,
      },
      {
          "name": "NW Rota-1",
          "latitude": 14.60,
          "longitude": 144.77,
          "elevation": -517,
      },
  ]
  return pd.DataFrame(mariana_volcanoes)


volcanoes_df = get_mariana_volcanoes()

# 載入 USGS 地震資料
url = (
    f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
    f"&starttime={START}&minmagnitude={MINMAG}"
    f"&minlongitude={REGION[0]}&maxlongitude={REGION[1]}"
    f"&minlatitude={REGION[2]}&maxlatitude={REGION[3]}"
)
quakes = pd.read_csv(url).dropna(subset=["longitude", "latitude", "depth", "mag"])

global_default_count = quakes["depth"].isin([10.0, 33.0]).sum()
global_ratio = (global_default_count / len(quakes)) * 100

print(
    f"成功載入馬里亞納地震: {len(quakes)} 筆\n"
    f"成功載入火山資料: {len(volcanoes_df)} 座\n"
    f"時間範圍: {START} 至今 ({datetime.now().year})\n"
    f"分析深度: 0 至 {Z_MAX_2D} km\n"
    f"USGS 預設深度 (10/33 km) 總筆數: {global_default_count} 筆 (佔全區資料 {global_ratio:.1f}%)\n"
)

grid_m = pygmt.datasets.load_earth_relief(resolution="01m", region=REGION)
grid_km = grid_m / 1000.0


def mag_size(m):
  return 0.1 * 2 ** (m - 5.0)


km_per_deg_lon = 111.32 * np.cos(np.radians(CUT_LAT))


# ==== 2. 繪圖核心函數 ====
def plot_mariana_dashboard(
    lon_A,
    lat_A,
    lon_B,
    lat_B,
    half_width_deg,
    azimuth,
    elevation,
    z_scale,
    depth_exag,
    show_trend=False,
    show_volcanoes=True,
    show_inclined_plane=False,
):
  fig = pygmt.Figure()

  pygmt.config(
      MAP_FRAME_TYPE="fancy",
      FORMAT_GEO_MAP="dddF",
      FONT_TITLE="14p,Helvetica-Bold,black",
      FONT_LABEL="10p,Helvetica-Bold,black",
      FONT_ANNOT_PRIMARY="9p,Helvetica,black",
  )

  WIDTH_CM_3D = 14
  WIDTH_CM_2D = 14
  HEIGHT_CM_2D = 12

  width_km = (REGION[1] - REGION[0]) * km_per_deg_lon
  zsize_cm = 6.0 * z_scale
  effective_z_min = Z_MIN_3D / depth_exag
  ve_3d = (zsize_cm / abs(effective_z_min)) / (WIDTH_CM_3D / width_km)

  # 1. 3D 地形
  pygmt.makecpt(cmap="oleron", series=[-8, 4, 0.5])
  fig.grdview(
      grid=grid_km * depth_exag,
      region=REGION + [Z_MIN_3D, 4 * depth_exag],
      projection=f"M{WIDTH_CM_3D}c",
      zsize=f"{zsize_cm}c",
      perspective=[azimuth, elevation],
      surftype="i",
      cmap=True,
      shading="+a315+nt0.9",
      frame=[
          f"WSnEZ+tMariana Trench (VE ~{ve_3d:.1f}x)",
          "xa5f1",
          "ya5f1",
          f"za{50 * depth_exag:.0f}f{25 * depth_exag:.0f}+lz (km)",
      ],
  )

  # 2. 繪製 3D 火山標記
  if show_volcanoes and len(volcanoes_df) > 0:
    fig.plot3d(
        x=volcanoes_df["longitude"],
        y=volcanoes_df["latitude"],
        z=np.zeros(len(volcanoes_df)),  # 放置於海平面頂部
        style="kvolcano/0.4c",  # GMT 火山標誌
        fill="red",
        pen="0.5p,black",
        perspective=[azimuth, elevation],
        region=REGION + [Z_MIN_3D, 4 * depth_exag],
        projection=f"M{WIDTH_CM_3D}c",
        zsize=f"{zsize_cm}c",
    )

  # 3. 繪製 3D 震源趨勢曲面
  if show_trend and len(quakes) >= 6:
    x_q = quakes["longitude"].values
    y_q = quakes["latitude"].values
    z_q = -quakes["depth"].values * depth_exag

    A_mat = np.column_stack(
        [x_q**2, y_q**2, x_q * y_q, x_q, y_q, np.ones_like(x_q)]
    )
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, z_q, rcond=None)

    lons_g = np.linspace(REGION[0], REGION[1], 35)
    lats_g = np.linspace(REGION[2], REGION[3], 35)
    lon_grid, lat_grid = np.meshgrid(lons_g, lats_g)

    z_grid = (
        coeffs[0] * lon_grid**2
        + coeffs[1] * lat_grid**2
        + coeffs[2] * lon_grid * lat_grid
        + coeffs[3] * lon_grid
        + coeffs[4] * lat_grid
        + coeffs[5]
    )
    z_grid = np.clip(z_grid, Z_MIN_3D * depth_exag, 0)

    trend_grid = xr.DataArray(
        z_grid,
        coords={"latitude": lats_g, "longitude": lons_g},
        dims=["latitude", "longitude"],
    )

    pygmt.makecpt(
        cmap="plasma",
        series=[Z_MIN_3D * depth_exag, 0, 10],
        output="trend_surface.cpt",
    )

    mesh_kwargs = {
        "grid": trend_grid,
        "region": REGION + [Z_MIN_3D, 4 * depth_exag],
        "projection": f"M{WIDTH_CM_3D}c",
        "zsize": f"{zsize_cm}c",
        "perspective": [azimuth, elevation],
        "surftype": "s",
        "cmap": "trend_surface.cpt",
        "transparency": 60,
    }

    try:
      fig.grdview(**mesh_kwargs, meshpen="0.5p,white@30")
    except pygmt.exceptions.GMTInvalidInput:
      fig.grdview(**mesh_kwargs, mesh_pen="0.5p,white@30")

  # 4. 3D 地震點
  pygmt.makecpt(cmap="inferno", series=[MINMAG, 8.0, 0.2])
  fig.plot3d(
      x=quakes.longitude,
      y=quakes.latitude,
      z=-quakes.depth * depth_exag,
      size=quakes.mag.apply(mag_size),
      fill=quakes.mag,
      cmap=True,
      style="u",
      pen="0.5p,white",
      transparency=20,
      perspective=[azimuth, elevation],
      region=REGION + [Z_MIN_3D, 4 * depth_exag],
      projection=f"M{WIDTH_CM_3D}c",
      zsize=f"{zsize_cm}c",
  )

  # 5. 剖面線與標記
  fig.plot3d(
      x=[lon_A, lon_B],
      y=[lat_A, lat_B],
      z=[0, 0],
      pen="2.5p,red",
      perspective=[azimuth, elevation],
      region=REGION + [Z_MIN_3D, 4 * depth_exag],
      projection=f"M{WIDTH_CM_3D}c",
      zsize=f"{zsize_cm}c",
  )

  fig.text(
      x=[lon_A, lon_B],
      y=[lat_A, lat_B],
      text=["A", "B"],
      perspective=[azimuth, elevation],
      font="14p,Helvetica-Bold,red",
      fill="white@20",
      pen="0.8p,red",
      offset="0.5c/0.5c",
      region=REGION + [Z_MIN_3D, 4 * depth_exag],
      projection=f"M{WIDTH_CM_3D}c",
  )

  fig.colorbar(
      position=f"jBR+o-1c/{-0.5 + 0.5 * z_scale:.1f}c+w6c/0.3c+h",
      frame=["xa1f0.5+lMagnitude (M)"],
      perspective=[180, 90],
  )

  # 6. 地震與火山資料投影至 A-B 走廊
  half_w_km = half_width_deg * 111.13
  try:
    data_in = quakes[["longitude", "latitude", "depth", "mag"]].values
    proj_df = pygmt.project(
        data=data_in,
        center=[lon_A, lat_A],
        endpoint=[lon_B, lat_B],
        width=f"{-half_w_km}/{half_w_km}",
        unit=True,
    )
    proj_df = proj_df.iloc[:, :6]
    proj_df.columns = ["lon", "lat", "depth", "mag", "p", "q"]
    num_quakes = len(proj_df)
  except Exception:
    proj_df = pd.DataFrame(columns=["lon", "lat", "depth", "mag", "p", "q"])
    num_quakes = 0

  max_dist = max(proj_df["p"].max() if num_quakes > 0 else 1000, 100)

  # 7. 右側面板：2D 剖面圖
  fig.shift_origin(xshift=f"{WIDTH_CM_3D + 6}c")

  fig.basemap(
      region=[0, max_dist * 1.05, 0, Z_MAX_2D],
      projection=f"X{WIDTH_CM_2D}c/-{HEIGHT_CM_2D}c",
      frame=[
          f"WSne+tProfile A-B (Width: {half_width_deg} deg)",
          "xa200f100+lDistance (km)",
          "ya100f50+lDepth (km)",
      ],
  )

  # 8. 繪製 AB 剖面疊加俯衝板塊傾斜面 (Slab Interface Plane Overlay)
  if show_inclined_plane and num_quakes >= 3:
    subduction_quakes = proj_df[
        (proj_df["depth"] > 30) | (proj_df["p"] > 150)
    ]
    if len(subduction_quakes) >= 3:
      p_vals = subduction_quakes["p"].values
      d_vals = subduction_quakes["depth"].values
    else:
      p_vals = proj_df["p"].values
      d_vals = proj_df["depth"].values

    poly_coef = np.polyfit(d_vals, p_vals, deg=2)
    d_grid = np.linspace(0, Z_MAX_2D, 150)
    p_top = np.polyval(poly_coef, d_grid)
    p_bot = p_top - 40.0  # 模擬約 40 km 厚度的俯衝板塊體

    # 繪製半透明板塊傾斜面 (Cyan 傾斜帶)
    polygon_p = np.concatenate([p_top, p_bot[::-1]])
    polygon_d = np.concatenate([d_grid, d_grid[::-1]])
    fig.plot(x=polygon_p, y=polygon_d, fill="cyan@80", pen="1p,cyan,dashed")

  # 9. 繪製 2D 地震點
  if num_quakes > 0:
    pygmt.makecpt(cmap="batlow", series=[0, Z_MAX_2D, 25], reverse=True)
    fig.plot(
        x=proj_df["p"],
        y=proj_df["depth"],
        style="c0.25c",
        fill=proj_df["depth"],
        cmap=True,
        pen="0.2p,black",
    )

    # 繪製 2D 震源趨勢擬合線
    if show_trend and num_quakes >= 3:
      subduction_quakes = proj_df[
          (proj_df["depth"] > 30) | (proj_df["p"] > 150)
      ]
      p_vals = (
          subduction_quakes["p"].values
          if len(subduction_quakes) >= 3
          else proj_df["p"].values
      )
      d_vals = (
          subduction_quakes["depth"].values
          if len(subduction_quakes) >= 3
          else proj_df["depth"].values
      )

      poly_coef = np.polyfit(d_vals, p_vals, deg=2)
      d_line = np.linspace(d_vals.min(), d_vals.max(), 100)
      p_line = np.polyval(poly_coef, d_line)
      fig.plot(x=p_line, y=d_line, pen="2p,magenta,--")

  # 10. 投影並繪製 2D 剖面火山圖示 (置於深度 0 km)
  if show_volcanoes and len(volcanoes_df) > 0:
    try:
      v_data = volcanoes_df[["longitude", "latitude"]].values
      v_proj = pygmt.project(
          data=v_data,
          center=[lon_A, lat_A],
          endpoint=[lon_B, lat_B],
          width=f"{-half_w_km}/{half_w_km}",
          unit=True,
      )
      if len(v_proj) > 0:
        v_p = v_proj.iloc[:, 4]
        fig.plot(
            x=v_p,
            y=np.zeros(len(v_p)),
            style="kvolcano/0.45c",
            fill="red",
            pen="0.5p,black",
        )
    except Exception:
      pass

  fig.colorbar(position="JMR+o1.5c/0c+w10c/0.3c", frame=["x+lDepth (km)"])

  # 存檔與顯式渲染顯示
  fig.savefig("Mariana_Dashboard.png", dpi=200)
  display(Image(filename="Mariana_Dashboard.png"))


# ==== 3. 建立 JavaScript 彈出視窗功能與按鈕 ====
def open_popup_window(b):
  try:
    with open("Mariana_Dashboard.png", "rb") as f:
      encoded_img = base64.b64encode(f.read()).decode("utf-8")

    js_code = f"""
        <script>
            var win = window.open("", "_blank");
            if (win) {{
                win.document.write('<title>Mariana Dashboard 放大檢視</title>');
                win.document.write('<body style="background:#111;margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;"><img src="data:image/png;base64,{encoded_img}" style="max-width:100%;height:auto;"/></body>');
                win.focus();
            }} else {{
                alert("彈出視窗被瀏覽器阻擋！請允許來自此頁面的快顯視窗。");
            }}
        </script>
        """
    display(HTML(js_code))
  except FileNotFoundError:
    print("請先調整拉桿生成圖表。")


btn_popup = widgets.Button(
    description="彈出視窗放大檢視",
    button_style="info",
    icon="external-link-alt",
    layout=widgets.Layout(width="320px", height="40px", margin="10px 0px"),
)

btn_popup.on_click(open_popup_window)

# ==== 4. 綁定控制項與顯示介面 ====
controls = widgets.interactive(
    plot_mariana_dashboard,
    lon_A=widgets.FloatSlider(
        value=136.0,
        min=REGION[0],
        max=REGION[1],
        step=0.5,
        description="A點經度",
        continuous_update=False,
    ),
    lat_A=widgets.FloatSlider(
        value=22.0,
        min=REGION[2],
        max=REGION[3],
        step=0.5,
        description="A點緯度",
        continuous_update=False,
    ),
    lon_B=widgets.FloatSlider(
        value=146.0,
        min=REGION[0],
        max=REGION[1],
        step=0.5,
        description="B點經度",
        continuous_update=False,
    ),
    lat_B=widgets.FloatSlider(
        value=12.0,
        min=REGION[2],
        max=REGION[3],
        step=0.5,
        description="B點緯度",
        continuous_update=False,
    ),
    half_width_deg=widgets.FloatSlider(
        value=1.5,
        min=0.1,
        max=5.0,
        step=0.1,
        description="走廊半寬",
        continuous_update=False,
    ),
    azimuth=widgets.IntSlider(
        value=165,
        min=0,
        max=360,
        step=5,
        description="3D方位角",
        continuous_update=False,
    ),
    elevation=widgets.IntSlider(
        value=30,
        min=10,
        max=85,
        step=5,
        description="3D仰角",
        continuous_update=False,
    ),
    z_scale=widgets.FloatSlider(
        value=1.2,
        min=0.2,
        max=2.5,
        step=0.1,
        description="3D高度比",
        continuous_update=False,
    ),
    depth_exag=widgets.FloatSlider(
        value=1.0,
        min=0.5,
        max=3.0,
        step=0.1,
        description="3D深度拉伸",
        continuous_update=False,
    ),
    show_volcanoes=widgets.Checkbox(value=True, description="標記火山位置"),
    show_inclined_plane=widgets.Checkbox(
        value=False, description="AB剖面疊加傾斜面"
    ),
    show_trend=widgets.Checkbox(
        value=False, description="顯示震源趨勢(曲面/線)"
    ),
)

display(btn_popup)
display(controls)
