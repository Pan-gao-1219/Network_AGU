"""
CycleGAN 地震波阻抗反演 · 数据预处理平台
Seismic Impedance Inversion Preprocessing Platform
"""

import streamlit as st
import subprocess
import sys
import os
import time
from pathlib import Path
import numpy as np

# ── 页面基础配置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="地震阻抗反演 · 预处理平台",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

PYTHON  = sys.executable
APP_DIR = Path(__file__).parent

# ── CSS 样式 ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* 侧边栏 */
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #c9d1d9; }

/* 卡片 */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 10px 0;
}
.card-green { border-left: 4px solid #2ea043; }
.card-blue  { border-left: 4px solid #388bfd; }
.card-orange{ border-left: 4px solid #d29922; }

/* 流程步骤框 */
.step-box {
    text-align: center;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 10px;
    transition: border-color 0.2s;
}
.step-box:hover { border-color: #2ea043; }

/* 状态徽章 */
.badge-ok  { background:#1a4731; color:#2ea043; padding:2px 10px; border-radius:12px; font-size:12px; }
.badge-err { background:#4a1e1e; color:#f85149; padding:2px 10px; border-radius:12px; font-size:12px; }
.badge-na  { background:#2d333b; color:#8b949e; padding:2px 10px; border-radius:12px; font-size:12px; }

/* 大标题渐变 */
.hero-title {
    background: linear-gradient(90deg, #2ea043, #388bfd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# Session state 默认值
# ══════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    original_dir   = r'C:\PythonProject\cyclegan\model pre+imp',
    output_root    = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output',
    seismic_segy   = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\synthetic\0度处理.segy',
    impedance_segy = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\resampled\MODEL_AI_1.25m_resampled_6800x1400.segy',
    trainA_dir     = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\trainA',
    trainB_dir     = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\trainB',
    merge_input    = r'',
    merge_output   = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\merged_impedance.npy',
    patch_size     = 256,
    stride         = 128,
    scale_factor   = 0.5,
    angles         = '0,5,10,15,20',
    max_workers    = 4,
    aug_count      = 3,
    merge_mode     = 'gaussian',
    norm_stats     = '',
    blocks_index   = '',
    original_segy  = '',
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════

def path_badge(p: str) -> str:
    p = Path(p)
    if not p.exists():
        return '<span class="badge-err">不存在</span>'
    if p.is_file():
        sz = p.stat().st_size / 1024**2
        return f'<span class="badge-ok">✓ {sz:.1f} MB</span>'
    n = sum(1 for _ in p.rglob('*') if _.is_file())
    return f'<span class="badge-ok">✓ {n} 文件</span>'


def count_blocks(directory: str) -> int:
    d = Path(directory)
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob('block_*.npy'))


def show_pngs(directory: str, max_n: int = 6, caption_prefix: str = ""):
    """在 3 列网格中展示目录下的 PNG 文件"""
    d = Path(directory)
    if not d.exists():
        return
    pngs = sorted(d.glob('**/*.png'))[:max_n]
    if not pngs:
        st.caption("暂无图像输出")
        return
    cols = st.columns(min(len(pngs), 3))
    for i, p in enumerate(pngs):
        cols[i % 3].image(str(p), caption=caption_prefix + p.name, use_container_width=True)


def run_and_stream(cmd: list, placeholder=None):
    """
    运行子进程并实时流式输出到 placeholder（st.empty()）。
    返回 (returncode, full_output_str)。
    """
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
        cwd=str(APP_DIR), bufsize=1, env=env,
    )
    lines = []
    for line in iter(proc.stdout.readline, ''):
        lines.append(line)
        if placeholder is not None:
            # 只显示最后 120 行，避免过长
            placeholder.code(''.join(lines[-120:]), language='text')
    proc.wait()
    return proc.returncode, ''.join(lines)


def ss(key):
    """Shortcut for st.session_state[key]"""
    return st.session_state[key]


def build_run(*parts):
    """构建 python run.py <parts> 命令"""
    return [PYTHON, str(APP_DIR / 'run.py')] + [str(x) for x in parts]


def result_section(output_dir: str, title: str = "输出结果"):
    """展示输出目录中的统计和图像"""
    d = Path(output_dir)
    if not d.exists():
        return
    blocks = count_blocks(output_dir)
    pngs   = list(d.glob('**/*.npy'))
    st.markdown(f"**{title}** — `{output_dir}`")
    c1, c2 = st.columns(2)
    c1.metric("NPY块数量", blocks)
    c2.metric("NPY文件数", len(pngs))
    show_pngs(output_dir)


# ══════════════════════════════════════════════════════════════════
# 侧边栏导航
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🌋 地震阻抗反演\n##### 数据预处理平台")
    st.markdown("---")

    page = st.radio(
        "页面导航",
        options=[
            "📖 项目介绍",
            "🔧 全局配置",
            "─────────────",
            "1️⃣  重采样",
            "2️⃣  正演模拟",
            "3️⃣  数据增强",
            "4️⃣  切割 trainA",
            "5️⃣  切割 trainB",
            "─────────────",
            "🚀  一键全流程",
            "🔄  合并结果",
            "✅  验证复原",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # 快速路径状态
    st.markdown("**路径状态**")
    for label, key in [
        ("原始SEGY", "original_dir"),
        ("合成地震", "seismic_segy"),
        ("阻抗SEGY", "impedance_segy"),
        ("trainA",   "trainA_dir"),
        ("trainB",   "trainB_dir"),
    ]:
        p = Path(ss(key))
        icon = "🟢" if p.exists() else "🔴"
        st.caption(f"{icon} {label}: `{p.name}`")

    st.markdown("---")
    st.caption(f"Python: `{Path(PYTHON).name}`")
    st.caption(f"工作目录: `{APP_DIR.name}`")


# ══════════════════════════════════════════════════════════════════
# 分隔符页（直接跳过）
# ══════════════════════════════════════════════════════════════════
if page == "─────────────":
    st.info("请从左侧导航选择功能页面。")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# 页面：项目介绍
# ══════════════════════════════════════════════════════════════════
elif page == "📖 项目介绍":

    st.markdown('<div class="hero-title">基于 CycleGAN 的地震波阻抗反演<br>数据预处理平台</div>', unsafe_allow_html=True)
    st.markdown("**Seismic Impedance Inversion Preprocessing Platform via CycleGAN**")
    st.markdown("---")

    # 项目简介
    st.markdown("""
    <div class="card card-green">
    <h4>📌 项目简介</h4>
    <p style="color:#c9d1d9; line-height:1.9; font-size:15px">
    本平台实现了一套完整的地震→波阻抗无监督域转换数据预处理流水线。针对地震勘探中波阻抗反演问题，
    利用 <b>CycleGAN</b> 非配对图像翻译框架，将合成地震记录（A域）映射到波阻抗剖面（B域），
    无需严格配对样本即可完成训练。本平台提供从原始模型文件到可直接训练的 NPY 数据集的完整处理能力，
    支持多角度正演、四种数据增强策略、高斯加权/中心保留两种合并方式及质量验证。
    </p>
    </div>
    """, unsafe_allow_html=True)

    # 作者介绍区域（可自定义）
    st.markdown("""
    <div class="card card-blue">
    <h4>👤 作者介绍</h4>
    <p style="color:#c9d1d9; line-height:1.8; font-size:15px">
    <!-- 请在此处填写您的个人介绍 -->
    <i style="color:#8b949e">请在 app.py 的"作者介绍"区域填写您的个人/团队介绍。</i>
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 处理流程
    st.subheader("🔄 完整处理流程")
    cols = st.columns(6)
    steps = [
        ("🗜️", "Step 1", "重采样", "原始SEGY\n缩放0.5倍\n13568→6800道"),
        ("📡", "Step 2", "正演模拟", "VP/VS/密度\n→合成地震\n5个入射角"),
        ("✨", "Step 3", "数据增强", "噪声·时移\n振幅·带通\n各×3份"),
        ("✂️", "Step 4a", "切割trainA", "256×256\nstride=128\n3通道编码"),
        ("🎯", "Step 4b", "切割trainB", "256×256\nstride=128\n单通道归一化"),
        ("🔀", "CycleGAN", "模型训练", "无监督域转换\n地震→阻抗"),
    ]
    for col, (icon, step, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="step-box">
            <div style="font-size:1.8rem">{icon}</div>
            <div style="color:#8b949e; font-size:11px; margin:2px 0">{step}</div>
            <b style="color:#2ea043; font-size:13px">{title}</b>
            <p style="color:#8b949e; font-size:11px; margin:4px 0; white-space:pre-line">{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 技术亮点
    st.subheader("⚙️ 技术亮点")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card card-green">
        <b>🎨 三通道地震编码</b><br>
        <span style="color:#8b949e; font-size:13px">
        基于希尔伯特变换将单道地震转为三通道：<br>
        · Ch0: 原始振幅（z-score）<br>
        · Ch1: 瞬时振幅（能量包络）<br>
        · Ch2: 瞬时相位 [-π,π]→[0,1]
        </span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card card-blue">
        <b>🔀 双模式无缝合并</b><br>
        <span style="color:#8b949e; font-size:13px">
        · 高斯加权：σ=P/6，边界平滑过渡<br>
        · 中心保留：取中心区域硬边界拼接<br>
        · 自动网格推算（无需padding_info）<br>
        · 反归一化还原原始值域
        </span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card card-orange">
        <b>⚡ 并行高效处理</b><br>
        <span style="color:#8b949e; font-size:13px">
        · 多进程并行切割SEGY文件<br>
        · 自动内存检测与限制<br>
        · 断点续跑（交互式询问）<br>
        · 支持10GB+大文件处理
        </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # 数据规格表
    st.subheader("📊 数据规格")
    st.table({
        "参数":   ["原始数据尺寸",   "重采样后尺寸", "切块大小",  "滑动步长", "重叠率", "trainA格式",      "trainB格式"],
        "数值":   ["13568×2688道",   "6800×1400道",  "256×256",   "128",      "50%",    "(H,W,3) float32", "(H,W) float32"],
        "说明":   ["1.25m间距原始模型","0.5倍重采样", "像素切块",  "半块重叠","相邻块有50%重叠","三通道地震","归一化[0,1]阻抗"],
    })

    # 快速开始
    st.markdown("---")
    st.subheader("🚀 快速开始")
    st.code("""# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动平台
streamlit run app.py

# 3. 也可直接命令行运行完整流程
python run.py all""", language="bash")


# ══════════════════════════════════════════════════════════════════
# 页面：全局配置
# ══════════════════════════════════════════════════════════════════
elif page == "🔧 全局配置":

    st.title("🔧 全局路径与参数配置")
    st.info('配置保存在当前会话中。修改后点击"保存配置"生效，所有步骤将读取此处的设置。')

    with st.form("global_config"):
        st.subheader("📁 路径配置")
        c1, c2 = st.columns(2)
        original_dir   = c1.text_input("原始SEGY目录（resample输入）",    ss("original_dir"))
        output_root    = c2.text_input("输出根目录",                       ss("output_root"))
        seismic_segy   = c1.text_input("合成地震SEGY（forward输出）",      ss("seismic_segy"))
        impedance_segy = c2.text_input("阻抗SEGY（cut_impedance输入）",    ss("impedance_segy"))
        trainA_dir     = c1.text_input("trainA 输出目录",                  ss("trainA_dir"))
        trainB_dir     = c2.text_input("trainB 输出目录",                  ss("trainB_dir"))
        merge_input    = c1.text_input("fake_B 目录（CycleGAN输出）",      ss("merge_input"))
        merge_output   = c2.text_input("合并结果输出路径（.npy）",         ss("merge_output"))

        st.subheader("⚙️ 处理参数")
        c1, c2, c3, c4 = st.columns(4)
        patch_size  = c1.number_input("patch_size（块大小）",  64, 512, ss("patch_size"), 64)
        stride      = c2.number_input("stride（步长）",        16, 256, ss("stride"), 16)
        max_workers = c3.number_input("max_workers（并行数）",  1,  32,  ss("max_workers"), 1)
        aug_count   = c4.number_input("aug_count（增强份数）",  1,  20,  ss("aug_count"), 1)

        c1, c2, c3 = st.columns(3)
        scale_factor = c1.number_input("scale_factor（重采样比例）", 0.1, 1.0, ss("scale_factor"), 0.05)
        angles       = c2.text_input("正演角度（逗号分隔）", ss("angles"))
        merge_mode   = c3.selectbox("合并模式", ["gaussian", "center"],
                                    index=0 if ss("merge_mode") == "gaussian" else 1)

        submitted = st.form_submit_button("💾 保存配置", type="primary", use_container_width=True)
        if submitted:
            updates = dict(
                original_dir=original_dir, output_root=output_root,
                seismic_segy=seismic_segy, impedance_segy=impedance_segy,
                trainA_dir=trainA_dir, trainB_dir=trainB_dir,
                merge_input=merge_input, merge_output=merge_output,
                patch_size=int(patch_size), stride=int(stride),
                max_workers=int(max_workers), aug_count=int(aug_count),
                scale_factor=float(scale_factor), angles=angles,
                merge_mode=merge_mode,
            )
            st.session_state.update(updates)
            st.success("✅ 配置已保存！")

    # 路径状态总览
    st.markdown("---")
    st.subheader("🔍 路径状态总览")
    checks = [
        ("原始SEGY目录",   "original_dir"),
        ("输出根目录",     "output_root"),
        ("合成地震SEGY",   "seismic_segy"),
        ("阻抗SEGY",       "impedance_segy"),
        ("trainA目录",     "trainA_dir"),
        ("trainB目录",     "trainB_dir"),
        ("fake_B目录",     "merge_input"),
    ]
    rows = []
    for label, key in checks:
        p = Path(ss(key))
        if not ss(key):
            status, detail = "⬜ 未设置", ""
        elif not p.exists():
            status, detail = "🔴 不存在", ""
        elif p.is_file():
            sz = p.stat().st_size / 1024**2
            status, detail = "🟢 存在", f"{sz:.1f} MB"
        else:
            n = count_blocks(ss(key))
            nf = sum(1 for _ in p.rglob('*') if _.is_file())
            status, detail = "🟢 存在", f"{nf} 文件 / {n} blocks"
        rows.append({"路径名称": label, "状态": status, "详情": detail, "路径": ss(key)})
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# 步骤1: 重采样
# ══════════════════════════════════════════════════════════════════
elif page == "1️⃣  重采样":

    st.title("步骤1 · SEGY 重采样")
    st.markdown("将原始高分辨率地震/弹性参数模型文件（VP/VS/密度/阻抗）重采样到目标分辨率，生成重采样SEGY文件及对比图。")

    resampled_dir = Path(ss("output_root")) / "resampled"
    existing = list(resampled_dir.glob("*.segy")) if resampled_dir.exists() else []

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"**输入目录**: `{ss('original_dir')}`")
        st.markdown(f"**输出目录**: `{resampled_dir}`")
    with c2:
        if existing:
            st.success(f"已有 {len(existing)} 个SEGY")
        else:
            st.warning("尚未生成")

    with st.expander("⚙️ 参数设置", expanded=not bool(existing)):
        c1, c2, c3, c4 = st.columns(4)
        scale  = c1.number_input("缩放比例", 0.1, 1.0, ss("scale_factor"), 0.05, key="r_scale")
        dt     = c2.number_input("时间采样间隔 dt (s)", 0.0001, 0.01, 0.001, 0.0001, key="r_dt", format="%.4f")
        dx     = c3.number_input("空间采样间隔 dx (m)", 1.0, 100.0, 12.5, 0.5, key="r_dx")
        st.markdown(f"**原始形状**: {ss('original_dir')}")

    log_box = st.empty()
    if st.button("▶ 开始重采样", type="primary", use_container_width=True, key="run_resample"):
        if not Path(ss("original_dir")).exists():
            st.error(f"❌ 原始目录不存在: {ss('original_dir')}，请先在[全局配置]中设置正确路径。")
        else:
            cmd = build_run(
                "resample",
                "--input_dir",    ss("original_dir"),
                "--output_root",  ss("output_root"),
                "--scale_factor", str(scale),
                "--dt",           str(dt),
                "--dx",           str(dx),
            )
            with st.spinner("重采样中..."):
                ret, out = run_and_stream(cmd, log_box)
            if ret == 0:
                st.success("✅ 重采样完成！")
            else:
                st.error("❌ 重采样失败，请查看日志。")

    if resampled_dir.exists():
        st.markdown("---")
        st.subheader("📊 输出文件")
        segy_files = sorted(resampled_dir.glob("*.segy"))
        for f in segy_files:
            sz = f.stat().st_size / 1024**2
            st.markdown(f"- `{f.name}` &nbsp; **{sz:.1f} MB**")
        show_pngs(str(resampled_dir))


# ══════════════════════════════════════════════════════════════════
# 步骤2: 正演模拟
# ══════════════════════════════════════════════════════════════════
elif page == "2️⃣  正演模拟":

    st.title("步骤2 · 正演模拟")
    st.markdown("读取重采样后的 VP、VS、密度文件，进行多角度 AVO 正演模拟，输出合成地震 SEGY 文件。")

    synthetic_dir = Path(ss("output_root")) / "synthetic"
    existing = list(synthetic_dir.glob("*.segy")) if synthetic_dir.exists() else []
    resampled_dir = Path(ss("output_root")) / "resampled"

    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**输入目录**: `{resampled_dir}`")
    c1.markdown(f"**输出目录**: `{synthetic_dir}`")
    if existing:
        c2.success(f"已有 {len(existing)} 个SEGY")
    else:
        c2.warning("尚未生成")

    with st.expander("⚙️ 参数设置", expanded=not bool(existing)):
        c1, c2 = st.columns(2)
        angles     = c1.text_input("入射角度（逗号分隔，度）", ss("angles"), key="f_angles")
        max_workers = c2.number_input("并行进程数", 1, 16, ss("max_workers"), key="f_workers")
        c1.markdown("**VP 文件名**: `MODEL_P-WAVE_VELOCITY_1.25m_resampled_*.segy`")
        c2.markdown("**VS 文件名**: `MODEL_S-WAVE_VELOCITY_1.25m_resampled_*.segy`")

    log_box = st.empty()
    if st.button("▶ 开始正演", type="primary", use_container_width=True, key="run_forward"):
        if not resampled_dir.exists() or not any(resampled_dir.glob("*.segy")):
            st.error("❌ 未找到重采样SEGY文件，请先完成步骤1（重采样）。")
        else:
            cmd = build_run(
                "forward",
                "--resampled_dir", str(resampled_dir),
                "--output_root",   ss("output_root"),
                "--angles",        angles,
                "--max_workers",   str(int(max_workers)),
            )
            with st.spinner("正演计算中（可能需要数分钟）..."):
                ret, out = run_and_stream(cmd, log_box)
            if ret == 0:
                st.success("✅ 正演完成！")
            else:
                st.error("❌ 正演失败，请查看日志。")

    if synthetic_dir.exists():
        st.markdown("---")
        st.subheader("📊 输出文件")
        for f in sorted(synthetic_dir.glob("*.segy")):
            sz = f.stat().st_size / 1024**2
            st.markdown(f"- `{f.name}` &nbsp; **{sz:.1f} MB**")
        show_pngs(str(synthetic_dir))


# ══════════════════════════════════════════════════════════════════
# 步骤3: 数据增强
# ══════════════════════════════════════════════════════════════════
elif page == "3️⃣  数据增强":

    st.title("步骤3 · 数据增强")
    st.markdown("对合成地震记录应用四种增强策略，扩充训练数据多样性。")

    aug_root = Path(ss("output_root")) / "augmented"
    synthetic_dir = Path(ss("output_root")) / "synthetic"

    # 各增强类型状态
    aug_types = {"noise": "高斯带限噪声", "time_shift": "时移增强",
                 "amplitude": "振幅缩放", "bandpass": "带通滤波"}
    st.subheader("当前增强状态")
    cols = st.columns(4)
    for col, (key, name) in zip(cols, aug_types.items()):
        d = aug_root / key
        n = len(list(d.glob("*.segy"))) if d.exists() else 0
        status = f"✅ {n} 个SEGY" if n > 0 else "⬜ 未生成"
        col.metric(name, status)

    with st.expander("⚙️ 参数设置", expanded=True):
        c1, c2 = st.columns(2)
        aug_count = c1.number_input("每种增强份数", 1, 20, ss("aug_count"), key="a_count")
        c1.markdown("**地震SEGY**: " + ss("seismic_segy"))
        c2.markdown("**阻抗SEGY**: " + ss("impedance_segy"))

        st.markdown("**启用的增强类型**")
        cc = st.columns(4)
        do_noise     = cc[0].checkbox("噪声增强",   True)
        do_timeshift = cc[1].checkbox("时移增强",   True)
        do_amplitude = cc[2].checkbox("振幅缩放",   True)
        do_bandpass  = cc[3].checkbox("带通滤波",   True)

    log_box = st.empty()
    if st.button("▶ 开始数据增强", type="primary", use_container_width=True, key="run_augment"):
        if not Path(ss("seismic_segy")).exists():
            st.error(f"❌ 合成地震SEGY不存在: {ss('seismic_segy')}，请先完成步骤2（正演）。")
        else:
            cmd = build_run(
                "augment",
                "--seismic",    ss("seismic_segy"),
                "--impedance",  ss("impedance_segy"),
                "--output_root", ss("output_root"),
                "--aug_count",  str(int(aug_count)),
            )
            if not do_noise:     cmd.append("--no_noise")
            if not do_timeshift: cmd.append("--no_time_shift")
            if not do_amplitude: cmd.append("--no_amplitude")
            if not do_bandpass:  cmd.append("--no_bandpass")

            with st.spinner("数据增强中..."):
                ret, out = run_and_stream(cmd, log_box)
            if ret == 0:
                st.success("✅ 数据增强完成！")
                st.rerun()
            else:
                st.error("❌ 增强失败，请查看日志。")

    if aug_root.exists():
        st.markdown("---")
        st.subheader("📊 增强结果预览")
        show_pngs(str(aug_root), max_n=6)


# ══════════════════════════════════════════════════════════════════
# 步骤4a: 切割 trainA
# ══════════════════════════════════════════════════════════════════
elif page == "4️⃣  切割 trainA":

    st.title("步骤4a · 切割地震数据 → trainA")
    st.markdown("将合成地震SEGY及增强数据切割为 256×256 NPY块，三通道编码（原始振幅+瞬时振幅+瞬时相位）。")

    trainA = Path(ss("trainA_dir"))
    synthetic_dir = Path(ss("output_root")) / "synthetic"
    aug_root = Path(ss("output_root")) / "augmented"
    n_blocks = count_blocks(ss("trainA_dir"))

    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**输出目录**: `{trainA}`")
    if n_blocks > 0:
        c2.success(f"已有 {n_blocks} 个块")
    else:
        c2.warning("尚未生成")

    with st.expander("⚙️ 参数设置"):
        c1, c2, c3 = st.columns(3)
        patch_size  = c1.number_input("patch_size", 64, 512, ss("patch_size"), 64, key="ca_ps")
        stride      = c2.number_input("stride",     16, 256, ss("stride"), 16, key="ca_st")
        max_workers = c3.number_input("max_workers", 1, 32, ss("max_workers"), key="ca_w")

        st.markdown("**切割来源**（依次切割以下数据）")
        do_synthetic = st.checkbox("✅ 原始正演数据（synthetic/）", True)
        do_aug_noise = st.checkbox("噪声增强（augmented/noise/）",     True)
        do_aug_ts    = st.checkbox("时移增强（augmented/time_shift/）",True)
        do_aug_amp   = st.checkbox("振幅缩放（augmented/amplitude/）", True)
        do_aug_bp    = st.checkbox("带通滤波（augmented/bandpass/）",  True)

    log_box = st.empty()
    if st.button("▶ 开始切割 trainA", type="primary", use_container_width=True, key="run_cut"):
        sources = []
        if do_synthetic: sources.append((str(synthetic_dir), str(trainA)))
        aug_map = [("noise", do_aug_noise), ("time_shift", do_aug_ts),
                   ("amplitude", do_aug_amp), ("bandpass", do_aug_bp)]
        for sub, enabled in aug_map:
            if enabled:
                d = aug_root / sub
                if d.exists() and any(d.glob("*.segy")):
                    sources.append((str(d), str(trainA / sub)))

        if not sources:
            st.error("❌ 没有可切割的数据源，请先完成步骤2/3。")
        else:
            all_ok = True
            for src, dst in sources:
                st.info(f"切割: `{src}` → `{dst}`")
                cmd = build_run(
                    "cut",
                    "--input",       src,
                    "--output",      dst,
                    "--patch_size",  str(int(patch_size)),
                    "--stride",      str(int(stride)),
                    "--max_workers", str(int(max_workers)),
                )
                ret, out = run_and_stream(cmd, log_box)
                if ret != 0:
                    all_ok = False
                    st.error(f"❌ 切割失败: `{src}`")
                    break
            if all_ok:
                st.success(f"✅ 切割完成！共 {count_blocks(ss('trainA_dir'))} 个块")
                st.rerun()

    if trainA.exists() and n_blocks > 0:
        st.markdown("---")
        st.subheader("📊 trainA 统计")
        subdirs = [d for d in trainA.iterdir() if d.is_dir()]
        if subdirs:
            rows = []
            for d in sorted(subdirs):
                nb = count_blocks(str(d))
                rows.append({"子目录": d.name, "块数量": nb})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        show_pngs(ss("trainA_dir"), max_n=6)


# ══════════════════════════════════════════════════════════════════
# 步骤5: 切割 trainB
# ══════════════════════════════════════════════════════════════════
elif page == "5️⃣  切割 trainB":

    st.title("步骤5 · 切割阻抗数据 → trainB")
    st.markdown("将波阻抗SEGY归一化到 [0,1] 并切割为 256×256 NPY块（单通道），同时保存归一化统计 `norm_stats.npy`。")

    trainB = Path(ss("trainB_dir"))
    n_blocks = count_blocks(ss("trainB_dir"))

    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**输入**: `{ss('impedance_segy')}`")
    c1.markdown(f"**输出**: `{trainB}`")
    if n_blocks > 0:
        c2.success(f"已有 {n_blocks} 个块")
    else:
        c2.warning("尚未生成")

    with st.expander("⚙️ 参数设置"):
        c1, c2 = st.columns(2)
        patch_size = c1.number_input("patch_size", 64, 512, ss("patch_size"), 64, key="cb_ps")
        stride     = c2.number_input("stride",     16, 256, ss("stride"), 16, key="cb_st")

    log_box = st.empty()
    if st.button("▶ 开始切割 trainB", type="primary", use_container_width=True, key="run_cut_b"):
        if not Path(ss("impedance_segy")).exists():
            st.error(f"❌ 阻抗SEGY不存在: {ss('impedance_segy')}")
        else:
            cmd = build_run(
                "cut_impedance",
                "--input",      ss("impedance_segy"),
                "--output",     str(trainB),
                "--patch_size", str(int(patch_size)),
                "--stride",     str(int(stride)),
            )
            with st.spinner("切割中..."):
                ret, out = run_and_stream(cmd, log_box)
            if ret == 0:
                st.success(f"✅ 切割完成！共 {count_blocks(ss('trainB_dir'))} 个块")
                st.rerun()
            else:
                st.error("❌ 切割失败，请查看日志。")

    # norm_stats 预览
    if trainB.exists():
        st.markdown("---")
        st.subheader("📊 归一化统计")
        for ns_path in trainB.rglob("norm_stats.npy"):
            stats = np.load(str(ns_path), allow_pickle=True).item()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("data_min", f"{stats.get('data_min', 0):.2f}")
            c2.metric("data_max", f"{stats.get('data_max', 0):.2f}")
            c3.metric("块总数",   f"{stats.get('n_rows',0)} × {stats.get('n_cols',0)}")
            c4.metric("原始形状", str(stats.get('original_shape', '?')))
            break
        show_pngs(ss("trainB_dir"), max_n=6)


# ══════════════════════════════════════════════════════════════════
# 一键全流程
# ══════════════════════════════════════════════════════════════════
elif page == "🚀  一键全流程":

    st.title("🚀 一键全流程")
    st.markdown("依次执行重采样→正演→增强→切割，自动询问是否跳过已完成步骤（命令行交互模式）。")

    st.warning("⚠️ 一键流程会在后台以非交互模式运行（自动跳过已有输出步骤）。如需交互控制，请分步执行。")

    with st.expander("⚙️ 参数确认", expanded=True):
        st.markdown(f"- **原始目录**: `{ss('original_dir')}`")
        st.markdown(f"- **输出根目录**: `{ss('output_root')}`")
        st.markdown(f"- **trainA**: `{ss('trainA_dir')}`")
        st.markdown(f"- **trainB**: `{ss('trainB_dir')}`")
        st.markdown(f"- patch_size={ss('patch_size')}, stride={ss('stride')}, aug_count={ss('aug_count')}")

    force = st.checkbox("⚡ --force 强制重新运行所有步骤", False)
    log_box = st.empty()

    if st.button("🚀 启动全流程", type="primary", use_container_width=True):
        if not Path(ss("original_dir")).exists():
            st.error(f"❌ 原始目录不存在: {ss('original_dir')}")
        else:
            # 逐步运行，自动 y 回答交互提示
            steps = [
                ("重采样", build_run("resample", "--input_dir", ss("original_dir"),
                                     "--output_root", ss("output_root"),
                                     "--scale_factor", str(ss("scale_factor")))),
                ("正演",   build_run("forward", "--resampled_dir",
                                     str(Path(ss("output_root"))/"resampled"),
                                     "--output_root", ss("output_root"),
                                     "--angles", ss("angles"),
                                     "--max_workers", str(ss("max_workers")))),
                ("增强",   build_run("augment", "--seismic", ss("seismic_segy"),
                                     "--impedance", ss("impedance_segy"),
                                     "--output_root", ss("output_root"),
                                     "--aug_count", str(ss("aug_count")))),
                ("切割A",  build_run("cut", "--input",
                                     str(Path(ss("output_root"))/"synthetic"),
                                     "--output", ss("trainA_dir"),
                                     "--patch_size", str(ss("patch_size")),
                                     "--stride", str(ss("stride")),
                                     "--max_workers", str(ss("max_workers")))),
                ("切割B",  build_run("cut_impedance", "--input", ss("impedance_segy"),
                                     "--output", ss("trainB_dir"),
                                     "--patch_size", str(ss("patch_size")),
                                     "--stride", str(ss("stride")))),
            ]
            if force:
                for step in steps:
                    step[1].append("--force") if "all" not in step[1] else None

            overall_ok = True
            progress = st.progress(0)
            for i, (name, cmd) in enumerate(steps):
                st.markdown(f"**▶ 步骤 {i+1}/{len(steps)}: {name}**")
                # 用 echo y | 自动回答交互提示
                env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    cwd=str(APP_DIR), bufsize=1, env=env,
                )
                # 自动输入 y 跳过已存在的步骤（配合 _step_done 中的 input() 询问）
                try:
                    proc.stdin.write('y\ny\ny\ny\ny\n')
                    proc.stdin.flush()
                    proc.stdin.close()
                except Exception:
                    pass

                lines = []
                for line in iter(proc.stdout.readline, ''):
                    lines.append(line)
                    log_box.code(''.join(lines[-80:]), language='text')
                proc.wait()

                if proc.returncode != 0:
                    st.error(f"❌ {name} 失败")
                    overall_ok = False
                    break
                progress.progress((i + 1) / len(steps))

            if overall_ok:
                st.success("🎉 全流程完成！")
                st.balloons()


# ══════════════════════════════════════════════════════════════════
# 合并结果
# ══════════════════════════════════════════════════════════════════
elif page == "🔄  合并结果":

    st.title("🔄 合并 CycleGAN 预测结果")
    st.markdown("将 CycleGAN 输出的 fake_B NPY块合并为完整波阻抗剖面，支持高斯加权或中心保留两种模式。")

    with st.form("merge_form"):
        st.subheader("📁 路径")
        c1, c2 = st.columns(2)
        merge_input  = c1.text_input("fake_B 目录（含 block_*.npy）", ss("merge_input"))
        merge_output = c2.text_input("输出路径（.npy）",               ss("merge_output"))

        c1, c2 = st.columns(2)
        norm_stats   = c1.text_input("norm_stats.npy 或 file_meta_info.npy（留空自动查找）", ss("norm_stats"))
        blocks_index = c2.text_input("blocks_index.npy（留空使用Strategy 3自动推算）",       ss("blocks_index"))
        original_segy = st.text_input("原始阻抗SEGY（用于MSE对比图，可选）", ss("original_segy"))

        st.subheader("⚙️ 参数")
        c1, c2, c3 = st.columns(3)
        patch_size = c1.number_input("patch_size", 64, 512, ss("patch_size"), 64, key="m_ps")
        stride     = c2.number_input("stride",     16, 256, ss("stride"), 16, key="m_st")
        merge_mode = c3.selectbox("合并模式",
                                  ["gaussian（高斯加权，平滑过渡）", "center（中心保留，硬边界）"],
                                  index=0 if ss("merge_mode") == "gaussian" else 1)

        submitted = st.form_submit_button("💾 确认参数", use_container_width=True)
        if submitted:
            st.session_state.update(merge_input=merge_input, merge_output=merge_output,
                                    norm_stats=norm_stats, blocks_index=blocks_index,
                                    original_segy=original_segy, patch_size=int(patch_size),
                                    stride=int(stride),
                                    merge_mode="gaussian" if "gaussian" in merge_mode else "center")
            st.success("参数已更新")

    log_box = st.empty()
    if st.button("▶ 开始合并", type="primary", use_container_width=True, key="run_merge"):
        if not Path(ss("merge_input")).exists():
            st.error(f"❌ fake_B 目录不存在: {ss('merge_input')}")
        else:
            cmd = build_run(
                "merge",
                "--input",      ss("merge_input"),
                "--output",     ss("merge_output"),
                "--patch_size", str(ss("patch_size")),
                "--stride",     str(ss("stride")),
                "--merge_mode", ss("merge_mode"),
            )
            if ss("norm_stats"):    cmd += ["--norm_stats",    ss("norm_stats")]
            if ss("blocks_index"):  cmd += ["--blocks_index",  ss("blocks_index")]
            if ss("original_segy"): cmd += ["--original",      ss("original_segy")]

            with st.spinner("合并中..."):
                ret, out = run_and_stream(cmd, log_box)
            if ret == 0:
                st.success("✅ 合并完成！")
                # 显示输出
                out_path = Path(ss("merge_output"))
                if out_path.exists():
                    data = np.load(str(out_path))
                    st.markdown(f"**输出形状**: `{data.shape}` &nbsp; 范围: `[{data.min():.2f}, {data.max():.2f}]`")
                show_pngs(str(out_path.parent), max_n=4)
            else:
                st.error("❌ 合并失败，请查看日志。")

    # 合并模式说明
    with st.expander("📖 合并模式说明"):
        c1, c2 = st.columns(2)
        c1.markdown("""
        **🌊 Gaussian（高斯加权）**
        - 重叠区域按高斯核加权平均
        - σ = patch_size / 6
        - 边界平滑过渡，无拼缝伪影
        - 适合大多数场景
        """)
        c2.markdown("""
        **🔲 Center（中心保留）**
        - 每块只取中心区域（各让出 overlap/2 像素）
        - P=256, S=128 → 取中心 128×128
        - 硬边界拼接，无模糊
        - 适合边界处信号精度要求高的场景
        """)


# ══════════════════════════════════════════════════════════════════
# 验证复原
# ══════════════════════════════════════════════════════════════════
elif page == "✅  验证复原":

    st.title("✅ 验证切割-合并复原精度")
    st.markdown("将 trainB 数据直接合并回原始形状，通过与原始阻抗SEGY对比验证切割→合并流程的保真度（理论MSE应极小）。")

    trainB = Path(ss("trainB_dir"))
    # 自动找第一个含 norm_stats.npy 的子目录
    auto_norm  = ""
    auto_input = ""
    if trainB.exists():
        for d in trainB.iterdir():
            if d.is_dir():
                ns = d / "norm_stats.npy"
                if ns.exists() and count_blocks(str(d)) > 0:
                    auto_norm  = str(ns)
                    auto_input = str(d)
                    break

    st.info(f"自动检测到:\n- **块目录**: `{auto_input or '未找到'}`\n- **norm_stats**: `{auto_norm or '未找到'}`")

    c1, c2 = st.columns(2)
    verify_input  = c1.text_input("trainB块目录",          auto_input)
    verify_output = c2.text_input("复原输出路径（.npy）",
                                  str(Path(ss("output_root")) / "verify_merge.npy"))
    norm_stats    = c1.text_input("norm_stats.npy 路径",    auto_norm)
    original_segy = c2.text_input("原始阻抗SEGY（用于MSE对比）", ss("impedance_segy"))

    c1, c2 = st.columns(2)
    patch_size = c1.number_input("patch_size", 64, 512, ss("patch_size"), 64, key="v_ps")
    stride     = c2.number_input("stride",     16, 256, ss("stride"), 16, key="v_st")
    merge_mode = st.selectbox("合并模式", ["gaussian", "center"], key="v_mode")

    log_box = st.empty()
    if st.button("▶ 开始验证", type="primary", use_container_width=True, key="run_verify"):
        if not verify_input or not Path(verify_input).exists():
            st.error("❌ 请指定有效的 trainB 块目录")
        else:
            cmd = build_run(
                "merge",
                "--input",      verify_input,
                "--output",     verify_output,
                "--patch_size", str(int(patch_size)),
                "--stride",     str(int(stride)),
                "--merge_mode", merge_mode,
            )
            if norm_stats:    cmd += ["--norm_stats", norm_stats]
            if original_segy: cmd += ["--original",   original_segy]

            with st.spinner("验证复原中..."):
                ret, out = run_and_stream(cmd, log_box)

            if ret == 0:
                st.success("✅ 验证完成！")
                vpath = Path(verify_output)
                if vpath.exists():
                    data = np.load(str(vpath))
                    st.markdown(f"**复原形状**: `{data.shape}` &nbsp; 值域: `[{data.min():.2f}, {data.max():.2f}]`")
                # 显示 MSE 对比图和剖面图
                show_pngs(str(vpath.parent), max_n=4)
                st.markdown("> 若 MSE 对比图中差异图接近零，说明切割→合并流程保真度良好。")
            else:
                st.error("❌ 验证失败，请查看日志。")
