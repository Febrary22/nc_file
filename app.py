"""
NC Viewer — 해양 예측 NetCDF(.nc) 파일을 클릭 몇 번으로 확인하는 웹앱
초보자용: 파일을 올리면 -> 변수/정보 자동 표시 -> 지도로 시각화

실행 방법:
    streamlit run app.py
"""

import glob
import os
import re
import tempfile

import koreanize_matplotlib  # noqa: F401  (matplotlib에 한글 폰트를 자동 등록해줌)
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

st.set_page_config(page_title="NC Viewer", page_icon="🌊", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# 변수명이 축약어라 초보자에게 뜻이 안 와닿는 경우가 많아서 한글 설명을 붙여줌
VAR_KOR_NAME = {
    "so": "염분 (Salinity)",
    "thetao": "수온 (Temperature)",
    "uo": "동서방향 해류속도 (Eastward current)",
    "vo": "남북방향 해류속도 (Northward current)",
    "zos": "해수면 높이 이상 (Sea Surface Height)",
    "zoo": "동물플랑크톤 (Zooplankton)",
}


# ---------------------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_local_files():
    """data/ 폴더 안의 nc 파일 목록"""
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.nc")))


@st.cache_resource(show_spinner="파일을 여는 중...")
def open_dataset(path: str):
    return xr.open_dataset(path)


def data_var_names(ds: xr.Dataset):
    """좌표(coord) 변수를 뺀 실제 데이터 변수만 추출"""
    coord_like = {"time", "depth", "lat", "lon", "latitude", "longitude"}
    return [v for v in ds.data_vars if v not in coord_like]


_MONTH_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])")


def month_label(filename: str) -> str:
    """파일명에서 YYYYMM 패턴을 찾아 'YYYY년 MM월' 형태로 바꿔준다.
    (예: zos_predict_202608.nc -> 2026년 08월). 못 찾으면 파일명 그대로 반환."""
    m = _MONTH_RE.search(filename)
    if not m:
        return filename
    return f"{m.group(1)}년 {m.group(2)}월"


# ---------------------------------------------------------------------------
# 사이드바: 파일 선택
# ---------------------------------------------------------------------------

st.sidebar.title("🌊 NC Viewer")
st.sidebar.caption("해류·수송 예측 NetCDF 파일을 쉽게 열어보는 도구")

local_files = list_local_files()
uploaded = st.sidebar.file_uploader(
    "nc 파일 업로드 (여러 개 선택 가능)",
    type=["nc"],
    accept_multiple_files=True,
    help="data/ 폴더에 파일을 미리 넣어두면 업로드 없이 바로 선택할 수 있어요.",
)

# 업로드된 파일은 임시 폴더에 저장해서 경로로 다룬다
file_options = {}
for p in local_files:
    file_options[os.path.basename(p)] = p

if uploaded:
    tmp_dir = tempfile.mkdtemp()
    for uf in uploaded:
        tmp_path = os.path.join(tmp_dir, uf.name)
        with open(tmp_path, "wb") as f:
            f.write(uf.getbuffer())
        file_options[uf.name] = tmp_path

if not file_options:
    st.info(
        "👈 왼쪽에서 nc 파일을 업로드하거나, `data/` 폴더에 nc 파일을 넣고 새로고침하세요."
    )
    st.stop()

def _file_select_label(name: str) -> str:
    label = month_label(name)
    return f"{label} ({name})" if label != name else name


selected_name = st.sidebar.selectbox(
    "파일 선택", sorted(file_options.keys()), format_func=_file_select_label
)
selected_path = file_options[selected_name]

ds = open_dataset(selected_path)

# ---------------------------------------------------------------------------
# 상단: 파일 기본 정보 (초보자가 "이 파일에 뭐가 담겨있는지" 바로 보는 부분)
# ---------------------------------------------------------------------------

st.title("🌊 NC Viewer")
st.caption(f"현재 파일: **{selected_name}**")

with st.expander("📋 이 파일에 담긴 정보 보기", expanded=True):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**차원 (Dimensions)**")
        dim_rows = [{"이름": k, "크기": v} for k, v in ds.sizes.items()]
        st.dataframe(dim_rows, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**전역 정보**")
        info_keys = ["title", "institution", "product_version"]
        info_rows = [
            {"항목": k, "값": ds.attrs.get(k, "-")}
            for k in info_keys
            if k in ds.attrs
        ]
        st.dataframe(info_rows, hide_index=True, use_container_width=True)

    st.markdown("**변수 목록 (Variables)**")
    var_rows = []
    for vname in data_var_names(ds):
        v = ds[vname]
        var_rows.append(
            {
                "변수": vname,
                "설명": VAR_KOR_NAME.get(vname, v.attrs.get("long_name", "-")),
                "단위": v.attrs.get("units", "-"),
                "차원": " x ".join(v.dims),
                "크기": " x ".join(str(s) for s in v.shape),
            }
        )
    st.dataframe(var_rows, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# 시각화 설정
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🗺️ 지도로 그려보기")

available_vars = data_var_names(ds)
var_labels = [f"{v} — {VAR_KOR_NAME.get(v, v)}" for v in available_vars]
label_to_var = dict(zip(var_labels, available_vars))

selected_label = st.selectbox("그릴 변수 선택", var_labels)
varname = label_to_var[selected_label]
da = ds[varname]

# time / depth 슬라이더 (있는 경우에만)
sel = {}
ctrl_cols = st.columns(3)

col_i = 0
if "time" in da.dims and da.sizes["time"] > 1:
    with ctrl_cols[col_i]:
        t_idx = st.slider("시간(time) 인덱스", 0, da.sizes["time"] - 1, 0)
    sel["time"] = t_idx
    col_i += 1
elif "time" in da.dims:
    sel["time"] = 0

if "depth" in da.dims:
    depth_vals = ds["depth"].values
    with ctrl_cols[col_i]:
        d_idx = st.select_slider(
            "깊이(depth, m)",
            options=list(range(len(depth_vals))),
            value=0,
            format_func=lambda i: f"{depth_vals[i]:.0f} m",
        )
    sel["depth"] = d_idx
    col_i += 1

overlay_current = False
if varname in ("uo", "vo"):
    with ctrl_cols[col_i]:
        overlay_current = st.checkbox("uo/vo 화살표(해류 방향) 같이 보기", value=True)

# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------

slice_da = da.isel(**sel)
lon = ds["lon"].values
lat = ds["lat"].values
values = slice_da.values

fig, ax = plt.subplots(figsize=(9, 7))
mesh = ax.pcolormesh(lon, lat, values, cmap="turbo", shading="auto")
cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
cbar.set_label(f"{varname} ({da.attrs.get('units', '')})")

title_bits = [VAR_KOR_NAME.get(varname, varname)]
if "depth" in sel:
    title_bits.append(f"깊이 {depth_vals[sel['depth']]:.0f} m")
ax.set_title(" · ".join(title_bits))
ax.set_xlabel("경도 (Longitude)")
ax.set_ylabel("위도 (Latitude)")
ax.set_aspect("equal")

# 해류 벡터 오버레이: uo/vo 둘 다 있어야 하므로 짝이 되는 파일을 같은 폴더에서 찾는다
if overlay_current:
    other_var = "vo" if varname == "uo" else "uo"
    other_fname = selected_name.replace(varname, other_var)
    other_path = file_options.get(other_fname)
    if other_path:
        other_ds = open_dataset(other_path)
        other_da = other_ds[other_var].isel(**sel)
        step = 20  # 격자 전체를 다 그리면 너무 빽빽해서 일부만 샘플링
        u = (ds["uo"].isel(**sel) if varname == "uo" else other_da).values
        v = (other_da if varname == "uo" else ds["vo"].isel(**sel)).values
        ax.quiver(
            lon[::step, ::step],
            lat[::step, ::step],
            u[::step, ::step],
            v[::step, ::step],
            color="white",
            scale=10,
            width=0.002,
        )
    else:
        st.warning(
            f"짝이 되는 파일 '{other_fname}' 을(를) 찾지 못해 화살표는 생략했어요. "
            "같은 폴더/업로드 목록에 두 파일을 함께 넣어주세요."
        )

st.pyplot(fig, use_container_width=True)

# 간단 통계
with st.expander("📊 값 통계 (최소/최대/평균)"):
    valid = values[~np.isnan(values)]
    if valid.size:
        s1, s2, s3 = st.columns(3)
        s1.metric("최소", f"{valid.min():.3f}")
        s2.metric("최대", f"{valid.max():.3f}")
        s3.metric("평균", f"{valid.mean():.3f}")
    else:
        st.write("이 슬라이스에는 유효한 값이 없어요 (전부 육지/결측일 수 있어요).")

# 이미지 다운로드
buf_path = os.path.join(tempfile.gettempdir(), "nc_viewer_plot.png")
fig.savefig(buf_path, dpi=150, bbox_inches="tight")
with open(buf_path, "rb") as f:
    st.download_button("🖼️ 그림 PNG로 다운로드", f, file_name=f"{varname}_{selected_name}.png")

# ---------------------------------------------------------------------------
# 두 시점 비교 (예: 이번 달 예측 vs 지난 달 예측)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📉 두 시점 비교 (차이 지도)")

# 같은 변수를 담고 있을 것으로 보이는 파일들(파일명에 변수명이 포함된 것)만 후보로 삼는다
compare_candidates = sorted(
    name for name in file_options if varname in name
)

if len(compare_candidates) < 2:
    st.info(
        f"'{varname}' 변수가 들어있는 파일이 2개 이상 있어야 비교할 수 있어요. "
        "같은 변수의 다른 시점(달) 파일을 함께 업로드해보세요."
    )
else:
    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        name_a = st.selectbox(
            "기준 시점 (A)",
            compare_candidates,
            index=0,
            format_func=month_label,
            key="compare_a",
        )
    with cmp_col2:
        name_b = st.selectbox(
            "비교 시점 (B)",
            compare_candidates,
            index=len(compare_candidates) - 1,
            format_func=month_label,
            key="compare_b",
        )

    if name_a == name_b:
        st.warning("서로 다른 두 시점을 선택해주세요.")
    else:
        ds_a = open_dataset(file_options[name_a])
        ds_b = open_dataset(file_options[name_b])
        da_a = ds_a[varname].isel(**{k: v for k, v in sel.items() if k in ds_a[varname].dims})
        da_b = ds_b[varname].isel(**{k: v for k, v in sel.items() if k in ds_b[varname].dims})
        values_a = da_a.values
        values_b = da_b.values

        if values_a.shape != values_b.shape:
            st.error("두 파일의 격자 크기가 달라서 비교할 수 없어요.")
        else:
            diff = values_b - values_a
            valid_diff = diff[~np.isnan(diff)]

            fig_diff, ax_diff = plt.subplots(figsize=(9, 7))
            vmax = np.abs(valid_diff).max() if valid_diff.size else 1
            mesh_diff = ax_diff.pcolormesh(
                lon, lat, diff, cmap="coolwarm", shading="auto", vmin=-vmax, vmax=vmax
            )
            cbar_diff = fig_diff.colorbar(mesh_diff, ax=ax_diff, shrink=0.8)
            cbar_diff.set_label(f"{varname} 차이 (B - A, {da.attrs.get('units', '')})")
            ax_diff.set_title(
                f"{month_label(name_b)} - {month_label(name_a)} "
                f"({VAR_KOR_NAME.get(varname, varname)})"
            )
            ax_diff.set_xlabel("경도 (Longitude)")
            ax_diff.set_ylabel("위도 (Latitude)")
            ax_diff.set_aspect("equal")
            st.pyplot(fig_diff, use_container_width=True)

            if valid_diff.size:
                d1, d2, d3 = st.columns(3)
                d1.metric("최소 차이", f"{valid_diff.min():.3f}")
                d2.metric("최대 차이", f"{valid_diff.max():.3f}")
                d3.metric("평균 차이", f"{valid_diff.mean():.3f}")
            else:
                st.write("비교할 유효한 값이 없어요.")
