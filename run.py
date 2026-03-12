"""
CycleGAN 地震数据流水线 - 统一命令行入口

完整处理流程:
  1. resample        重采样SEGY文件（生成SEGY + 对比图）
  2. forward         正演模拟（生成SEGY + 成像图）
  3. augment         数据增强（生成SEGY + 对比图）
  4. cut             地震数据归一化+切割为NPY（A域，生成样本图）
  4b.cut_impedance   阻抗归一化+切割（B域，生成样本图）
  5. merge           合并训练结果 + 与原始数据绘制MSE对比图

用法示例:
  # 一键完整预处理流程
  python run.py all --input_dir ./original --output_root ./output

  # 单步运行
  python run.py resample --input_dir ./original
  python run.py forward  --resampled_dir ./output/resampled
  python run.py augment  --seismic ./output/synthetic/0度处理.segy
  python run.py cut      --input ./output/synthetic --output ./trainA
  python run.py cut_impedance --input ./output/resampled/impedance.segy --output ./trainB
  python run.py merge    --input ./fake_B_npy --output ./merged.npy \\
                         --norm_stats ./trainB/norm_stats.npy \\
                         --original  ./output/resampled/impedance.segy

python run.py merge      --input <fake_B_npy> --output result.npy --merge_mode center
"""

import argparse
import gc
import sys
import time
import warnings
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── 中文字体设置 ──────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")


# ============================================================
# 默认参数（统一在这里修改）
# ============================================================
DEFAULTS = {
    # ── 原始未缩放文件目录（resample 步骤的输入）────────────────
    'original_dir':   r'C:\PythonProject\cyclegan\model pre+imp',
    # 原始各文件名（在 original_dir 下）
    'vp_file_orig':   'MODEL_P-WAVE_VELOCITY_1.25m.segy',
    'vs_file_orig':   'MODEL_S-WAVE_VELOCITY_1.25m.segy',
    'den_file_orig':  'MODEL_DENSITY_1.25m.segy',
    'imp_file_orig':  'MODEL_AI_1.25m.segy',

    # ── 输出根目录 ───────────────────────────────────────────────
    'output_root':    r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output',
    'seismic_segy':   r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\synthetic\0度处理.segy',
    'impedance_segy': r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\resampled\MODEL_AI_1.25m_resampled_6800x1400.segy',
    'trainA_dir':     r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\trainA',
    'trainB_dir':     r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\trainB',
    'merge_input':    r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\merge_input',
    'merge_output':   r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\merged_impedance.npy',

    # 切割参数
    'patch_size': 256,
    'stride':     128,

    # 重采样参数
    'original_shape': (13568, 2688),
    'scale_factor':   0.5,

    # 正演参数
    'angles':      [0, 5, 10, 15, 20],
    'max_workers': 4,

    # 增强参数
    'aug_count':   3,
    'noise':       True,
    'time_shift':  True,
    'amplitude':   True,
    'bandpass':    True,
}


# ============================================================
# 工具函数
# ============================================================

def _save_segy_2d(data: np.ndarray, output_path: str):
    """将2D numpy数组保存为SEGY文件"""
    import segyio
    data = data.astype(np.float32)
    n_traces, n_samples = data.shape
    spec = segyio.spec()
    spec.sorting = 1
    spec.format = 5   # 32位IEEE浮点
    spec.samples = np.arange(n_samples, dtype=np.float32)
    spec.tracecount = n_traces
    with segyio.create(output_path, spec) as f:
        for i in range(n_traces):
            f.trace[i] = data[i]
    print(f"  已保存SEGY: {output_path}")


def _check_mem_gb():
    """返回当前可用内存(GB)"""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**3
    except ImportError:
        return 999.0


def _show_sample_patches(patches_dir: Path, output_png: Path,
                         n_show: int = 9, title: str = "样本切片"):
    """从切割目录随机抽取n个patch显示"""
    files = sorted(patches_dir.glob('block_*.npy'))
    if not files:
        return
    step = max(1, len(files) // n_show)
    sample_files = files[::step][:n_show]
    cols = 3
    rows = (len(sample_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).flatten()
    for idx, f in enumerate(sample_files):
        block = np.load(f)
        if block.ndim == 3:
            block = block[:, :, 0]   # 取第0通道
        lo, hi = np.percentile(block, [2, 98])
        axes[idx].imshow(block.T, aspect='auto', cmap='seismic', vmin=lo, vmax=hi)
        axes[idx].set_title(f'块 #{int(f.stem.split("_")[-1])}', fontsize=9)
        axes[idx].axis('off')
    for ax in axes[len(sample_files):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  样本图已保存: {output_png}")


# ============================================================
# 步骤1: 重采样
# ============================================================

def run_resample(args):
    """步骤1: 重采样 - 生成SEGY + 对比图（中文）"""
    print("\n" + "=" * 60)
    print("步骤1: 重采样")
    print("=" * 60)

    from regetmodel import MarmousiResampler

    resampler = MarmousiResampler(
        original_shape=tuple(args.original_shape),
        dt=args.dt,
        dx=args.dx
    )

    output_dir = Path(args.output_root) / 'resampled'
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        list(Path(args.input_dir).glob('*.segy')) +
        list(Path(args.input_dir).glob('*.sgy'))
    )

    if not input_files:
        print(f"❌ 未找到SEGY文件: {args.input_dir}")
        return {}

    results = {}
    for f in input_files:
        print(f"\n处理: {f.name}")
        data, info = resampler.resample_file(
            input_path=str(f),
            output_dir=str(output_dir),
            scale_factor=args.scale_factor,
            method='zoom',
            order=1,
            show_plot=getattr(args, 'show_plot', False)
        )
        # 保存对比图（MarmousiResampler已生成，这里确保它有中文标题）
        results[f.stem] = {'path': str(output_dir / f"{f.stem}_resampled_{info['traces']}x{info['samples']}.segy"),
                           'shape': data.shape}
        print(f"  ✅ 完成: {data.shape}")
        gc.collect()

    return results


# ============================================================
# 步骤2: 正演模拟
# ============================================================

def run_forward(args):
    """步骤2: 正演模拟 - 生成SEGY + 成像图（中文）"""
    print("\n" + "=" * 60)
    print("步骤2: 正演模拟")
    print("=" * 60)

    resampled_dir = Path(args.resampled_dir)
    output_dir    = Path(args.output_root) / 'synthetic'
    output_dir.mkdir(parents=True, exist_ok=True)

    vp_path  = str(resampled_dir / args.vp_file)
    vs_path  = str(resampled_dir / args.vs_file)
    den_path = str(resampled_dir / args.den_file)

    for p in [vp_path, vs_path, den_path]:
        if not Path(p).exists():
            print(f"❌ 文件不存在: {p}")
            return {}

    angles = [int(a) for a in args.angles.split(',')]
    print(f"角度: {angles}")
    print(f"输出目录: {output_dir}")

    # ── 动态导入，避免模块级硬编码路径的副作用 ──────────────
    import importlib.util, types
    # 安全导入: 跳过模块级的 segyio.open 调用
    def _safe_import_forward():
        spec = importlib.util.spec_from_file_location(
            "forword_modeling",
            str(Path(__file__).parent / "forword_modeling.py")
        )
        mod = types.ModuleType("forword_modeling")
        # 先将全局变量设为dummy，防止模块级报错
        import segyio
        class _DummyFile:
            class bin:
                @staticmethod
                def __getitem__(k): return 2000
            def __enter__(self): return self
            def __exit__(self, *a): pass
        with open(str(Path(__file__).parent / "forword_modeling.py"), 'r', encoding='utf-8') as src:
            code = src.read()
        # 加载vp/vs/rho
        import segyio as _segyio
        vp_data  = _segyio.open(vp_path,  "r", ignore_geometry=True).trace.raw[:].astype(np.float32)
        vs_data  = _segyio.open(vs_path,  "r", ignore_geometry=True).trace.raw[:].astype(np.float32)
        rho_data = _segyio.open(den_path, "r", ignore_geometry=True).trace.raw[:].astype(np.float32)
        with _segyio.open(den_path, "r", ignore_geometry=True) as f:
            del_D = f.bin[_segyio.BinField.Interval]
        return vp_data, vs_data, rho_data, del_D

    vp_data, vs_data, rho_data, del_D = _safe_import_forward()

    # 导入核心函数
    sys.path.insert(0, str(Path(__file__).parent))
    # 使用进程池并行处理各角度
    from concurrent.futures import ProcessPoolExecutor
    # 由于forword_modeling有模块级副作用，改为串行调用
    from forword_modeling import process_angle

    results = {}
    for angle in angles:
        print(f"\n处理角度 {angle}°...")
        synthetic = process_angle(
            angle, vp_data, vs_data, rho_data, del_D,
            vp_data.shape[0], vp_data.shape[1],
            output_dir=str(output_dir)
        )

        # synthetic 可能是返回的数组，也可能已经保存到文件
        segy_file = output_dir / f"{angle}度处理.segy"
        if segy_file.exists():
            print(f"  ✅ 已生成: {segy_file.name}")
        elif isinstance(synthetic, np.ndarray):
            _save_segy_2d(synthetic, str(segy_file))

        # ── 生成可视化图 ─────────────────────────────────────
        if isinstance(synthetic, np.ndarray):
            _plot_forward_result(synthetic, angle, output_dir)
        elif segy_file.exists():
            import segyio
            with segyio.open(str(segy_file), "r", ignore_geometry=True) as f:
                data = f.trace.raw[:].astype(np.float32)
            _plot_forward_result(data, angle, output_dir)

        results[f"{angle}度"] = {'seismic': str(segy_file)}
        gc.collect()

    print(f"\n✅ 正演完成，生成 {len(results)} 个角度文件")
    return results


def _plot_forward_result(data: np.ndarray, angle: int, output_dir: Path):
    """生成正演结果可视化图（中文）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    lo, hi = np.percentile(data, [2, 98])

    # 完整剖面
    im = axes[0].imshow(data.T, aspect='auto', cmap='seismic', vmin=lo, vmax=hi)
    axes[0].set_title(f'{angle}° 合成地震剖面', fontsize=11)
    axes[0].set_xlabel('道号')
    axes[0].set_ylabel('采样点')
    plt.colorbar(im, ax=axes[0], fraction=0.03)

    # 中间道波形
    mid = data.shape[0] // 2
    axes[1].plot(data[mid], np.arange(data.shape[1]), 'b-', linewidth=0.8)
    axes[1].invert_yaxis()
    axes[1].set_title(f'中间道波形 (道#{mid})', fontsize=11)
    axes[1].set_xlabel('振幅')
    axes[1].set_ylabel('采样点')
    axes[1].grid(True, alpha=0.3)

    # 振幅直方图
    axes[2].hist(data.flatten(), bins=80, color='steelblue', alpha=0.75, density=True)
    axes[2].set_title('振幅分布', fontsize=11)
    axes[2].set_xlabel('振幅值')
    axes[2].set_ylabel('概率密度')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'正演模拟结果 - {angle}°入射角  ({data.shape[0]}道 × {data.shape[1]}采样点)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    png_path = output_dir / f"{angle}度处理_正演图.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  正演图已保存: {png_path.name}")


# ============================================================
# 步骤3: 数据增强
# ============================================================

def run_augment(args):
    """步骤3: 数据增强 - 生成SEGY + 中文对比图"""
    print("\n" + "=" * 60)
    print("步骤3: 数据增强")
    print("=" * 60)

    from data_pipeline.augmentation.noise     import GaussianBandlimitedNoise
    from data_pipeline.augmentation.timemove  import TimeShiftAugmentation
    from data_pipeline.augmentation.amplitude import AmplitudeScalingAugmentation
    from data_pipeline.augmentation.bandpass  import RandomBandpassFilterAugmentation

    seismic_path   = args.seismic
    impedance_path = getattr(args, 'impedance', None)
    aug_root       = Path(args.output_root) / 'augmented'
    n              = args.aug_count

    if not Path(seismic_path).exists():
        print(f"❌ 地震文件不存在: {seismic_path}")
        return

    if args.noise:
        print("应用噪声增强...")
        aug = GaussianBandlimitedNoise(seismic_path)
        aug_dir = aug_root / 'noise'
        aug.apply(output_dir=str(aug_dir), num_augmentations=n)
        _plot_augmentation_summary(seismic_path, aug_dir, '噪声增强', 'noise')
        gc.collect()

    if args.time_shift and impedance_path and Path(impedance_path).exists():
        print("应用时移增强...")
        aug = TimeShiftAugmentation(seismic_path, impedance_path)
        aug_dir = aug_root / 'time_shift'
        aug.apply(output_dir=str(aug_dir), num_augmentations=n, mode='roll')
        _plot_augmentation_summary(seismic_path, aug_dir, '时移增强', 'time_shift')
        gc.collect()

    if args.amplitude:
        print("应用振幅缩放...")
        aug = AmplitudeScalingAugmentation(seismic_path, impedance_path)
        aug_dir = aug_root / 'amplitude'
        aug.apply(output_dir=str(aug_dir), num_augmentations=n)
        _plot_augmentation_summary(seismic_path, aug_dir, '振幅缩放增强', 'amplitude')
        gc.collect()

    if args.bandpass:
        print("应用带通滤波...")
        aug = RandomBandpassFilterAugmentation(seismic_path, impedance_path)
        aug_dir = aug_root / 'bandpass'
        aug.apply(output_dir=str(aug_dir), num_augmentations=n)
        _plot_augmentation_summary(seismic_path, aug_dir, '带通滤波增强', 'bandpass')
        gc.collect()

    print("✅ 数据增强完成")


def _plot_augmentation_summary(original_segy: str, aug_dir: Path,
                                aug_name: str, aug_type: str):
    """生成增强汇总对比图（中文）"""
    import segyio
    aug_files = sorted(aug_dir.glob('*.segy'))
    if not aug_files:
        return

    with segyio.open(original_segy, "r", ignore_geometry=True) as f:
        orig = f.trace.raw[:].astype(np.float32)

    with segyio.open(str(aug_files[0]), "r", ignore_geometry=True) as f:
        aug_data = f.trace.raw[:].astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lo_o, hi_o = np.percentile(orig, [2, 98])
    lo_a, hi_a = np.percentile(aug_data, [2, 98])

    axes[0].imshow(orig.T, aspect='auto', cmap='seismic', vmin=lo_o, vmax=hi_o)
    axes[0].set_title('原始数据', fontsize=11)
    axes[0].set_xlabel('道号'); axes[0].set_ylabel('采样点')

    axes[1].imshow(aug_data.T, aspect='auto', cmap='seismic', vmin=lo_a, vmax=hi_a)
    axes[1].set_title(f'{aug_name}后（第1次）', fontsize=11)
    axes[1].set_xlabel('道号'); axes[1].set_ylabel('采样点')

    diff = aug_data - orig
    lo_d, hi_d = np.percentile(diff, [2, 98])
    axes[2].imshow(diff.T, aspect='auto', cmap='RdBu', vmin=lo_d, vmax=hi_d)
    axes[2].set_title('差异图', fontsize=11)
    axes[2].set_xlabel('道号'); axes[2].set_ylabel('采样点')

    fig.suptitle(f'{aug_name} 对比  ({aug_data.shape[0]}道 × {aug_data.shape[1]}采样点)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    png_path = aug_dir / f"{aug_type}_汇总对比图.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  增强汇总图已保存: {png_path.name}")


# ============================================================
# 步骤4a: 地震数据切割（A域）
# ============================================================

def run_cut(args):
    """步骤4a: 地震数据归一化 + 切割（A域，3通道编码），显示样本图"""
    print("\n" + "=" * 60)
    print("步骤4a: 地震数据切割（A域，3通道归一化NPY）")
    print("=" * 60)

    from parallel_programsegycutter import AutoParallelSEGYCutter

    cutter = AutoParallelSEGYCutter(
        input_dir=args.input,
        patch_size=args.patch_size,
        stride=args.stride,
        output_base_dir=args.output,
        norm_range=(0, 1),
        normalize=True,
        recursive=getattr(args, 'recursive', True),
        max_workers=args.max_workers,
        memory_limit_gb=getattr(args, 'memory_limit', 8.0)
    )
    results = cutter.run()
    total = results.get('total_blocks', '?')
    print(f"✅ 切割完成，共 {total} 块")
    print(f"   输出目录: {args.output}")

    # 显示样本图
    output_root = Path(args.output)
    for sub in output_root.iterdir():
        if sub.is_dir() and list(sub.glob('block_*.npy')):
            _show_sample_patches(
                sub,
                output_root / f"{sub.name}_样本图.png",
                n_show=9,
                title=f'地震切割样本 - {sub.name}'
            )
            break   # 只显示第一个子目录


# ============================================================
# 步骤4b: 阻抗数据切割（B域）
# ============================================================

def run_cut_impedance(args):
    """步骤4b: 阻抗归一化[0,1] + 切割为NPY（B域），保存归一化统计 + 样本图"""
    print("\n" + "=" * 60)
    print("步骤4b: 阻抗数据切割（B域，单通道，归一化到[0,1]）")
    print("=" * 60)

    import segyio

    input_path = Path(args.input)
    output_dir = Path(args.output)
    patch_size = args.patch_size
    stride     = args.stride

    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return

    # ── 加载 ─────────────────────────────────────────────────
    print(f"加载: {input_path.name}")
    with segyio.open(str(input_path), "r", ignore_geometry=True) as f:
        data = f.trace.raw[:].astype(np.float32)
    print(f"  形状: {data.shape}, 范围: [{data.min():.2f}, {data.max():.2f}]")

    # ── 归一化 [0,1] ─────────────────────────────────────────
    d_min, d_max = float(data.min()), float(data.max())
    data_norm = (data - d_min) / (d_max - d_min + 1e-10)
    print(f"  归一化后: [{data_norm.min():.4f}, {data_norm.max():.4f}]")

    # ── 填充（使切割整除）────────────────────────────────────
    n_traces, n_samples = data_norm.shape
    target_traces  = n_traces
    target_samples = n_samples
    if (target_traces  - patch_size) % stride != 0:
        target_traces  = ((target_traces  - patch_size) // stride + 1) * stride + patch_size
    if (target_samples - patch_size) % stride != 0:
        target_samples = ((target_samples - patch_size) // stride + 1) * stride + patch_size

    pad_top    = (target_traces  - n_traces)  // 2
    pad_bottom = target_traces   - n_traces   - pad_top
    pad_left   = (target_samples - n_samples) // 2
    pad_right  = target_samples  - n_samples  - pad_left

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        data_padded = np.pad(data_norm,
                             ((pad_top, pad_bottom), (pad_left, pad_right)),
                             mode='symmetric')
        print(f"  填充后: {data_padded.shape}  "
              f"(pad: top={pad_top}, bot={pad_bottom}, l={pad_left}, r={pad_right})")
    else:
        data_padded = data_norm

    # ── 切割 ─────────────────────────────────────────────────
    sub_dir = output_dir / input_path.stem
    sub_dir.mkdir(parents=True, exist_ok=True)

    n_i = (data_padded.shape[0] - patch_size) // stride + 1
    n_j = (data_padded.shape[1] - patch_size) // stride + 1
    total = n_i * n_j
    print(f"  切割网格: {n_i}行 × {n_j}列 = {total} 块")

    blocks_index = []
    block_id = 0
    for i in range(0, data_padded.shape[0] - patch_size + 1, stride):
        for j in range(0, data_padded.shape[1] - patch_size + 1, stride):
            block = data_padded[i:i + patch_size, j:j + patch_size].copy()
            np.save(sub_dir / f"block_{block_id:06d}.npy", block)
            blocks_index.append({
                'block_id': block_id,
                'file':     f"block_{block_id:06d}.npy",
                'position': (i, j),
                'shape':    block.shape
            })
            block_id += 1
            if block_id % 500 == 0:
                print(f"    进度: {block_id}/{total}", end='\r')

    # ── 保存元数据（不再保存padding_info，只保存norm_stats）──
    np.save(sub_dir / 'blocks_index.npy', blocks_index)
    np.save(sub_dir / 'norm_stats.npy', {
        'data_min':       d_min,
        'data_max':       d_max,
        'norm_range':     (0.0, 1.0),
        'patch_size':     patch_size,
        'stride':         stride,
        'original_shape': (n_traces, n_samples),
        'padded_shape':   data_padded.shape,
        'n_rows':         n_i,
        'n_cols':         n_j
    })

    print(f"\n✅ 完成: {block_id} 块 → {sub_dir}")
    print(f"   反归一化公式: x * {d_max - d_min:.4f} + {d_min:.4f}")

    # ── 样本图 ────────────────────────────────────────────────
    _show_sample_patches(
        sub_dir,
        output_dir / f"{input_path.stem}_样本切片图.png",
        n_show=9,
        title=f'阻抗切割样本 - {input_path.stem}'
    )

    # ── 归一化前后对比图 ──────────────────────────────────────
    _plot_normalization_comparison(data, data_norm, d_min, d_max,
                                   output_dir / f"{input_path.stem}_归一化对比图.png")


def _plot_normalization_comparison(orig: np.ndarray, norm: np.ndarray,
                                   d_min: float, d_max: float,
                                   png_path: Path):
    """归一化前后对比图（中文）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lo_o, hi_o = np.percentile(orig, [2, 98])
    axes[0, 0].imshow(orig.T, aspect='auto', cmap='seismic', vmin=lo_o, vmax=hi_o)
    axes[0, 0].set_title(f'原始数据  范围[{d_min:.1f}, {d_max:.1f}]', fontsize=10)
    axes[0, 0].set_xlabel('道号'); axes[0, 0].set_ylabel('采样点')

    axes[0, 1].imshow(norm.T, aspect='auto', cmap='seismic', vmin=0, vmax=1)
    axes[0, 1].set_title('归一化后 [0, 1]', fontsize=10)
    axes[0, 1].set_xlabel('道号'); axes[0, 1].set_ylabel('采样点')

    axes[1, 0].hist(orig.flatten(), bins=80, color='steelblue', alpha=0.7, density=True)
    axes[1, 0].set_title('原始数据振幅分布', fontsize=10)
    axes[1, 0].set_xlabel('振幅'); axes[1, 0].set_ylabel('概率密度')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(norm.flatten(), bins=80, color='coral', alpha=0.7, density=True)
    axes[1, 1].set_title('归一化后振幅分布', fontsize=10)
    axes[1, 1].set_xlabel('归一化值'); axes[1, 1].set_ylabel('概率密度')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('阻抗数据归一化对比', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  归一化对比图已保存: {png_path.name}")


# ============================================================
# 步骤5: 合并（不依赖padding_info，自动推算网格）
# ============================================================

def _infer_grid(pred_dir: Path, patch_size: int, stride: int,
                blocks_index_path: Path = None):
    """
    从块文件推算合并网格，不依赖 padding_info.npy。

    策略优先级:
      1. blocks_index.npy 中的位置信息（最可靠）
         - 先查 blocks_index_path（外部指定），再查 pred_dir 本目录
      2. norm_stats.npy / file_meta_info.npy 中记录的 n_rows/n_cols
      3. 从块文件数量枚举合理因子对
    返回: (n_rows, n_cols, padded_h, padded_w)
    """
    block_files = sorted(pred_dir.glob('block_*.npy'))
    n_blocks    = len(block_files)
    if n_blocks == 0:
        raise ValueError(f"目录中没有 block_*.npy 文件: {pred_dir}")

    # ── 策略1: blocks_index.npy ──────────────────────────────
    # 优先用外部指定路径，否则查本目录
    index_path = blocks_index_path if (blocks_index_path and Path(blocks_index_path).exists()) \
                 else pred_dir / 'blocks_index.npy'
    if index_path.exists():
        try:
            idx = np.load(str(index_path), allow_pickle=True)
            positions = [b['position'] for b in idx]
            max_i = max(int(p[0]) for p in positions)
            max_j = max(int(p[1]) for p in positions)
            padded_h = max_i + patch_size
            padded_w = max_j + patch_size
            n_rows = (padded_h - patch_size) // stride + 1
            n_cols = (padded_w - patch_size) // stride + 1
            print(f"  [网格推算] 来自 blocks_index.npy: "
                  f"{n_rows}行×{n_cols}列, 合并尺寸({padded_h},{padded_w})")
            return n_rows, n_cols, padded_h, padded_w
        except Exception as e:
            print(f"  [警告] 读取 blocks_index.npy 失败: {e}")

    # ── 策略2: norm_stats.npy 或 file_meta_info.npy ──────────
    for meta_name in ('norm_stats.npy', 'file_meta_info.npy'):
        stats_path = pred_dir / meta_name
        if not stats_path.exists():
            continue
        try:
            stats = np.load(str(stats_path), allow_pickle=True).item()
            if 'n_rows' in stats and 'n_cols' in stats:
                n_rows = int(stats['n_rows'])
                n_cols = int(stats['n_cols'])
                padded_h = (n_rows - 1) * stride + patch_size
                padded_w = (n_cols - 1) * stride + patch_size
                print(f"  [网格推算] 来自 {meta_name}: "
                      f"{n_rows}行×{n_cols}列, 合并尺寸({padded_h},{padded_w})")
                return n_rows, n_cols, padded_h, padded_w
            # file_meta_info.npy 用 padded_shape 推算
            if 'padded_shape' in stats and 'original_shape' in stats:
                ps = stats['padded_shape']
                padded_h, padded_w = int(ps[0]), int(ps[1])
                n_rows = (padded_h - patch_size) // stride + 1
                n_cols = (padded_w - patch_size) // stride + 1
                print(f"  [网格推算] 来自 {meta_name}(padded_shape): "
                      f"{n_rows}行×{n_cols}列, 合并尺寸({padded_h},{padded_w})")
                return n_rows, n_cols, padded_h, padded_w
        except Exception as e:
            print(f"  [警告] 读取 {meta_name} 失败: {e}")

    # ── 策略3: 枚举因子对，选最合理的（道数 >> 采样数）────────
    print(f"  [网格推算] 从文件数量({n_blocks})枚举网格...")
    candidates = []
    for nc in range(1, n_blocks + 1):
        if n_blocks % nc == 0:
            nr = n_blocks // nc
            ph = (nr - 1) * stride + patch_size
            pw = (nc - 1) * stride + patch_size
            # 地震数据通常 道数(ph) >> 采样数(pw)，期望宽高比 pw/ph < 0.5
            ratio = pw / ph
            candidates.append((nr, nc, ph, pw, abs(ratio - 0.2)))

    # 优先选宽高比接近0.2（地震数据典型比例）
    candidates.sort(key=lambda x: x[4])
    nr, nc, ph, pw, _ = candidates[0]
    print(f"  [网格推算] 自动选择: {nr}行×{nc}列, 合并尺寸({ph},{pw})")
    print(f"  ⚠️ 如结果不对，请在切割目录提供 blocks_index.npy 或 norm_stats.npy")
    return nr, nc, ph, pw


def run_merge(args):
    """
    步骤5: 合并 fake_B（或任意切割块）→ 完整图

    特性:
      - 不依赖 padding_info.npy，自动推算网格
      - 高斯加权平均消除拼缝
      - 可选反归一化（需 --norm_stats）
      - 可选与原始SEGY对比并绘制MSE图（需 --original）
      - 全程中文标注
    """
    merge_mode = getattr(args, 'merge_mode', 'gaussian')
    print("\n" + "=" * 60)
    mode_desc = "高斯加权" if merge_mode == 'gaussian' else "中心保留硬边界"
    print(f"步骤5: 合并 → 完整图（{mode_desc} + 自动网格推算）")
    print("=" * 60)

    pred_dir    = Path(args.input)
    output_path = Path(args.output)
    patch_size  = args.patch_size
    stride      = args.stride

    print(f"输入目录:  {pred_dir}")
    print(f"输出路径:  {output_path}")
    print(f"块大小:    {patch_size}, 步长: {stride}")

    # ── 获取块文件列表 ──────────────────────────────────────
    block_files = sorted(pred_dir.glob('block_*.npy'))
    n_blocks    = len(block_files)
    if n_blocks == 0:
        print(f"❌ 未找到 block_*.npy: {pred_dir}")
        return
    print(f"找到 {n_blocks} 个块文件")

    # ── 推算网格 ────────────────────────────────────────────
    ext_bi = getattr(args, 'blocks_index', None)
    n_rows, n_cols, padded_h, padded_w = _infer_grid(
        pred_dir, patch_size, stride,
        blocks_index_path=Path(ext_bi) if ext_bi else None
    )

    if n_rows * n_cols != n_blocks:
        print(f"  ⚠️ 网格({n_rows}×{n_cols}={n_rows*n_cols})与文件数({n_blocks})不符，"
              f"重新校正列数...")
        # 强制以 n_rows 为准，调整 n_cols
        n_cols = n_blocks // n_rows
        padded_w = (n_cols - 1) * stride + patch_size
        print(f"  调整后: {n_rows}行×{n_cols}列, 合并尺寸({padded_h},{padded_w})")

    # ── 读取第一个块判断维度 ──────────────────────────────────
    first = np.load(str(block_files[0]), allow_pickle=True)
    if first.dtype == object:
        first = first.item()
    first = np.array(first, dtype=np.float32)
    print(f"块形状: {first.shape}")

    merged    = np.zeros((padded_h, padded_w), dtype=np.float32)
    mem_avail = _check_mem_gb()
    print(f"可用内存: {mem_avail:.1f} GB")

    if merge_mode == 'gaussian':
        # ── 高斯加权合并 ─────────────────────────────────────
        sigma       = patch_size / 6.0
        kernel_base = np.ones((patch_size, patch_size), dtype=np.float32)
        kernel      = scipy.ndimage.gaussian_filter(kernel_base, sigma=sigma)
        kernel     /= kernel.max()
        print(f"高斯核: sigma={sigma:.1f}, 边缘权重≈{kernel[0, 0]:.4f}")
        print(f"正在高斯加权合并 {n_blocks} 个块...")

        weight_map = np.zeros((padded_h, padded_w), dtype=np.float32)

        for idx, block_file in enumerate(block_files):
            row = idx // n_cols
            col = idx  % n_cols
            i   = row  * stride
            j   = col  * stride

            block = np.load(str(block_file), allow_pickle=True)
            if block.dtype == object:
                block = block.item()
            block = np.array(block, dtype=np.float32)
            if block.ndim == 3:
                block = block[0] if block.shape[0] in [1, 3] else block[:, :, 0]

            ie = min(i + patch_size, padded_h)
            je = min(j + patch_size, padded_w)
            h, w = ie - i, je - j
            if h <= 0 or w <= 0:
                continue

            k = kernel[:h, :w]
            merged[i:ie, j:je]     += block[:h, :w] * k
            weight_map[i:ie, j:je] += k

            if (idx + 1) % 200 == 0:
                print(f"  进度: {idx + 1}/{n_blocks}", end='\r')
                gc.collect()

        print(f"  进度: {n_blocks}/{n_blocks} ✓")
        merged /= (weight_map + 1e-8)

    else:
        # ── 中心保留硬边界合并 ──────────────────────────────
        # 重叠宽度由 patch_size 和 stride 自动推算，无需硬编码
        overlap      = patch_size - stride          # 每侧重叠量
        half_overlap = overlap // 2                 # 各方向让出的像素数
        print(f"中心保留合并: patch={patch_size}, stride={stride}, "
              f"重叠={overlap}, 各侧让出={half_overlap}px")
        print(f"正在中心保留合并 {n_blocks} 个块...")

        for idx, block_file in enumerate(block_files):
            row = idx // n_cols
            col = idx  % n_cols

            # 当前块在本地坐标中"拥有"的范围
            r0_local = 0          if row == 0        else half_overlap
            r1_local = patch_size if row == n_rows - 1 else patch_size - half_overlap
            c0_local = 0          if col == 0        else half_overlap
            c1_local = patch_size if col == n_cols - 1 else patch_size - half_overlap

            # 映射到全局坐标
            gr0 = row * stride + r0_local
            gc0 = col * stride + c0_local
            gr1 = min(row * stride + r1_local, padded_h)
            gc1 = min(col * stride + c1_local, padded_w)
            if gr1 <= gr0 or gc1 <= gc0:
                continue

            block = np.load(str(block_file), allow_pickle=True)
            if block.dtype == object:
                block = block.item()
            block = np.array(block, dtype=np.float32)
            if block.ndim == 3:
                block = block[0] if block.shape[0] in [1, 3] else block[:, :, 0]

            # 从块的对应局部区域取像素
            merged[gr0:gr1, gc0:gc1] = block[r0_local: r0_local + (gr1 - gr0),
                                              c0_local: c0_local + (gc1 - gc0)]

            if (idx + 1) % 200 == 0:
                print(f"  进度: {idx + 1}/{n_blocks}", end='\r')
                gc.collect()

        print(f"  进度: {n_blocks}/{n_blocks} ✓")

    print(f"合并后范围: [{merged.min():.4f}, {merged.max():.4f}]")

    # ── 裁剪到原始形状（从norm_stats或original SEGY获取）──────
    orig_shape = None
    stats_path = pred_dir / 'norm_stats.npy'
    if stats_path.exists():
        try:
            stats = np.load(str(stats_path), allow_pickle=True).item()
            if 'original_shape' in stats:
                orig_shape = tuple(int(x) for x in stats['original_shape'])
                print(f"从 norm_stats.npy 获取原始形状: {orig_shape}")
        except Exception:
            pass

    original_segy = getattr(args, 'original', None)
    if original_segy and Path(original_segy).exists():
        import segyio
        with segyio.open(str(original_segy), "r", ignore_geometry=True) as f:
            orig_shape_from_file = (f.tracecount, len(f.samples))
        if orig_shape is None:
            orig_shape = orig_shape_from_file
        print(f"原始SEGY形状: {orig_shape_from_file}")

    merged_final = _crop_to_original(merged, orig_shape)
    print(f"裁剪后形状: {merged_final.shape}")

    # ── 反归一化（支持 norm_stats.npy 和 file_meta_info.npy）──
    norm_stats_path = getattr(args, 'norm_stats', None)
    merged_denorm   = None
    d_min = d_max   = None

    # 若未指定，自动在 pred_dir 查找
    if not norm_stats_path:
        for meta_name in ('norm_stats.npy', 'file_meta_info.npy'):
            candidate = pred_dir / meta_name
            if candidate.exists():
                norm_stats_path = str(candidate)
                print(f"  自动找到归一化文件: {meta_name}")
                break

    if norm_stats_path and Path(norm_stats_path).exists():
        stats = np.load(str(norm_stats_path), allow_pickle=True).item()
        d_min = float(stats.get('data_min', stats.get('min', 0)))
        d_max = float(stats.get('data_max', stats.get('max', 1)))
        # 只有确实归一化过才做反归一化
        norm_flag = stats.get('normalized', True)
        if norm_flag and abs(d_max - d_min) > 1e-6:
            merged_denorm = merged_final * (d_max - d_min) + d_min
            print(f"反归一化: [{d_min:.2f}, {d_max:.2f}]  "
                  f"结果范围: [{merged_denorm.min():.2f}, {merged_denorm.max():.2f}]")
        else:
            print(f"  数据未归一化或范围极小，跳过反归一化")

    # ── 保存结果 ──────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = merged_denorm if merged_denorm is not None else merged_final
    np.save(str(output_path), save_data)
    print(f"✅ 保存: {output_path}  形状={save_data.shape}")

    # ── 生成合并剖面图 ────────────────────────────────────────
    png_path = output_path.parent / (output_path.stem + '_合并剖面图.png')
    _plot_merged_profile(save_data, png_path)

    # ── MSE 对比图（需 --original）───────────────────────────
    if original_segy and Path(original_segy).exists():
        import segyio
        with segyio.open(str(original_segy), "r", ignore_geometry=True) as f:
            orig_data = f.trace.raw[:].astype(np.float32)

        # 裁剪到相同大小
        mh = min(save_data.shape[0], orig_data.shape[0])
        mw = min(save_data.shape[1], orig_data.shape[1])
        sd = save_data[:mh, :mw]
        od = orig_data[:mh, :mw]

        mse_png = output_path.parent / (output_path.stem + '_MSE对比图.png')
        _plot_mse_comparison(od, sd, mse_png,
                             orig_name='原始数据（重采样）',
                             pred_name='合并预测结果')


def _crop_to_original(merged: np.ndarray, orig_shape) -> np.ndarray:
    """将合并结果裁剪到原始形状（居中裁剪）"""
    if orig_shape is None:
        return merged
    mh, mw = merged.shape
    oh, ow = orig_shape
    # 原始尺寸不能超过合并尺寸
    oh = min(oh, mh)
    ow = min(ow, mw)
    pad_h = mh - oh
    pad_w = mw - ow
    top   = pad_h // 2
    left  = pad_w // 2
    return merged[top:top + oh, left:left + ow]


def _plot_merged_profile(data: np.ndarray, png_path: Path):
    """合并结果剖面图（中文）"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    lo, hi = np.percentile(data, [2, 98])
    im = axes[0].imshow(data.T, aspect='auto', cmap='seismic', vmin=lo, vmax=hi)
    axes[0].set_title(f'合并结果剖面  {data.shape[0]}道×{data.shape[1]}采样点', fontsize=11)
    axes[0].set_xlabel('道号')
    axes[0].set_ylabel('采样点')
    plt.colorbar(im, ax=axes[0], fraction=0.02)

    axes[1].hist(data.flatten(), bins=80, color='steelblue', alpha=0.75, density=True)
    axes[1].set_title('合并结果振幅分布', fontsize=11)
    axes[1].set_xlabel('值')
    axes[1].set_ylabel('概率密度')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('CycleGAN 合并结果', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  合并剖面图已保存: {png_path.name}")


def _plot_mse_comparison(original: np.ndarray, predicted: np.ndarray,
                         png_path: Path,
                         orig_name: str = '原始数据',
                         pred_name: str = '预测数据'):
    """
    与原始数据对比图 + 逐道MSE折线图（中文）
    """
    # 全局MSE
    mse_global = float(np.mean((original - predicted) ** 2))
    rmse_global = float(np.sqrt(mse_global))
    # 逐道MSE
    mse_per_trace = np.mean((original - predicted) ** 2, axis=1)

    fig = plt.figure(figsize=(20, 14))
    gs  = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)

    lo_o, hi_o = np.percentile(original,  [2, 98])
    lo_p, hi_p = np.percentile(predicted, [2, 98])
    diff = predicted - original
    lo_d, hi_d = np.percentile(np.abs(diff), [2, 98])

    # ── 原始数据剖面 ──────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(original.T, aspect='auto', cmap='seismic', vmin=lo_o, vmax=hi_o)
    ax0.set_title(orig_name, fontsize=10)
    ax0.set_xlabel('道号'); ax0.set_ylabel('采样点')
    plt.colorbar(im0, ax=ax0, fraction=0.03)

    # ── 预测结果剖面 ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(predicted.T, aspect='auto', cmap='seismic', vmin=lo_p, vmax=hi_p)
    ax1.set_title(pred_name, fontsize=10)
    ax1.set_xlabel('道号'); ax1.set_ylabel('采样点')
    plt.colorbar(im1, ax=ax1, fraction=0.03)

    # ── 差异图（绝对值）─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(np.abs(diff).T, aspect='auto', cmap='hot', vmin=0, vmax=hi_d)
    ax2.set_title('绝对差异图', fontsize=10)
    ax2.set_xlabel('道号'); ax2.set_ylabel('采样点')
    plt.colorbar(im2, ax=ax2, fraction=0.03)

    # ── 逐道MSE折线图 ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(mse_per_trace, color='crimson', linewidth=0.8, alpha=0.9)
    ax3.fill_between(np.arange(len(mse_per_trace)), mse_per_trace,
                     alpha=0.3, color='crimson')
    ax3.axhline(mse_global, color='navy', linestyle='--', linewidth=1.5,
                label=f'全局MSE = {mse_global:.6f}')
    ax3.set_title('逐道均方误差（MSE）', fontsize=11)
    ax3.set_xlabel('道号')
    ax3.set_ylabel('MSE')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── 中间道对比 ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    mid = original.shape[0] // 2
    ax4.plot(original[mid],  np.arange(original.shape[1]),  'b-',  lw=1,   label=orig_name, alpha=0.8)
    ax4.plot(predicted[mid], np.arange(predicted.shape[1]), 'r--', lw=1.2, label=pred_name, alpha=0.9)
    ax4.invert_yaxis()
    ax4.set_title(f'中间道对比 (道#{mid})', fontsize=10)
    ax4.set_xlabel('振幅'); ax4.set_ylabel('采样点')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 频谱对比 ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    spec_o = np.abs(np.fft.fft(original[mid]))
    spec_p = np.abs(np.fft.fft(predicted[mid]))
    nf = len(spec_o) // 2
    ax5.plot(spec_o[:nf], 'b-',  lw=1,   label=orig_name, alpha=0.8)
    ax5.plot(spec_p[:nf], 'r--', lw=1.2, label=pred_name, alpha=0.9)
    ax5.set_title('中间道频谱对比', fontsize=10)
    ax5.set_xlabel('频率 (idx)'); ax5.set_ylabel('振幅谱')
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, nf // 2)

    # ── 统计信息文字框 ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    info = (
        f"误差统计\n"
        f"{'─'*28}\n"
        f"全局 MSE : {mse_global:.6f}\n"
        f"全局 RMSE: {rmse_global:.6f}\n\n"
        f"原始数据范围:\n"
        f"  [{original.min():.4f}, {original.max():.4f}]\n"
        f"预测数据范围:\n"
        f"  [{predicted.min():.4f}, {predicted.max():.4f}]\n\n"
        f"数据尺寸:\n"
        f"  原始: {original.shape[0]}×{original.shape[1]}\n"
        f"  预测: {predicted.shape[0]}×{predicted.shape[1]}"
    )
    ax6.text(0.05, 0.95, info, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('预测结果与原始数据对比（MSE评估）', fontsize=14, fontweight='bold')
    plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  MSE对比图已保存: {png_path.name}")
    print(f"  全局 MSE={mse_global:.6f},  RMSE={rmse_global:.6f}")


# ============================================================
# 完整一键流程
# ============================================================

def _step_done(label, check_fn, force):
    """检查某步骤是否已完成，询问用户是否跳过，返回 True 表示跳过"""
    if force:
        return False
    if check_fn():
        while True:
            ans = input(f"\n  [{label}] 检测到已有输出，是否跳过？(y=跳过 / n=重新运行): ").strip().lower()
            if ans in ('y', 'yes', ''):
                print(f"  ⏭  跳过 {label}")
                return True
            if ans in ('n', 'no'):
                return False
    return False


def run_all(args):
    """一键运行完整预处理流程 1→2→3→4a+4b"""
    print("\n" + "=" * 60)
    print("完整预处理流程（重采样→正演→增强→归一化+切割）")
    print("=" * 60)
    t0    = time.time()
    force = getattr(args, 'force', False)

    resampled_dir = Path(args.output_root) / 'resampled'
    synthetic_dir = Path(args.output_root) / 'synthetic'
    aug_dir       = Path(args.output_root) / 'augmented'
    trainA_dir    = Path(args.trainA_dir)
    trainB_dir    = Path(args.trainB_dir)

    # ── 步骤1: 重采样 ────────────────────────────────────────
    if not _step_done(
        '步骤1 重采样',
        lambda: any(resampled_dir.glob('*.segy')),
        force
    ):
        run_resample(args)

    # ── 步骤2: 正演 ──────────────────────────────────────────
    if not _step_done(
        '步骤2 正演',
        lambda: any(synthetic_dir.glob('*.segy')),
        force
    ):
        run_forward(args)

    # ── 步骤3: 数据增强 ──────────────────────────────────────
    def _aug_done():
        for sub in ('noise', 'time_shift', 'amplitude', 'bandpass'):
            d = aug_dir / sub
            if d.exists() and any(d.glob('*.segy')):
                return True
        return False

    if not _step_done('步骤3 数据增强', _aug_done, force):
        run_augment(args)

    # ── 步骤4a: 切割地震 → trainA（原始正演 + 增强数据各自切割）──
    # 原始正演切割
    if not _step_done(
        '步骤4a-1 切割原始正演(trainA)',
        lambda: any((trainA_dir / '0度处理').rglob('block_*.npy')),
        force
    ):
        args.input  = str(synthetic_dir)
        args.output = str(trainA_dir)
        run_cut(args)

    # 增强数据切割（每个增强子目录）
    for aug_sub in ('noise', 'time_shift', 'amplitude', 'bandpass'):
        aug_sub_dir = aug_dir / aug_sub
        if not aug_sub_dir.exists() or not any(aug_sub_dir.glob('*.segy')):
            print(f"  ⚠  {aug_sub} 无增强数据，跳过")
            continue
        if not _step_done(
            f'步骤4a-2 切割增强数据({aug_sub})',
            lambda d=aug_sub_dir: any((trainA_dir / d.name).rglob('block_*.npy')),
            force
        ):
            args.input  = str(aug_sub_dir)
            args.output = str(trainA_dir / aug_sub)
            run_cut(args)

    # ── 步骤4b: 切割阻抗 → trainB ────────────────────────────
    if not _step_done(
        '步骤4b 切割阻抗(trainB)',
        lambda: any(trainB_dir.rglob('block_*.npy')),
        force
    ):
        args.input  = args.impedance
        args.output = str(trainB_dir)
        run_cut_impedance(args)

    print(f"\n✅ 全流程完成，总耗时: {time.time() - t0:.1f} 秒")


# ============================================================
# 命令行解析
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description='CycleGAN 地震数据流水线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    # ── resample ─────────────────────────────────────────────
    p = sub.add_parser('resample', help='步骤1: 重采样SEGY文件（生成SEGY+对比图）')
    p.add_argument('--input_dir',      default=DEFAULTS['original_dir'],
                   help='原始未缩放SEGY文件目录')
    p.add_argument('--output_root',    default=DEFAULTS['output_root'])
    p.add_argument('--original_shape', nargs=2, type=int,
                   default=list(DEFAULTS['original_shape']), metavar=('TRACES', 'SAMPLES'))
    p.add_argument('--scale_factor',   type=float, default=DEFAULTS['scale_factor'])
    p.add_argument('--dt',             type=float, default=0.001)
    p.add_argument('--dx',             type=float, default=12.5)
    p.add_argument('--show_plot',      action='store_true')

    # ── forward ──────────────────────────────────────────────
    p = sub.add_parser('forward', help='步骤2: 正演模拟（生成SEGY+成像图）')
    p.add_argument('--resampled_dir',  default=str(Path(DEFAULTS['output_root']) / 'resampled'))
    p.add_argument('--output_root',    default=DEFAULTS['output_root'])
    p.add_argument('--vp_file',        default='MODEL_P-WAVE_VELOCITY_1.25m_resampled_6800x1400.segy')
    p.add_argument('--vs_file',        default='MODEL_S-WAVE_VELOCITY_1.25m_resampled_6800x1400.segy')
    p.add_argument('--den_file',       default='MODEL_DENSITY_1.25m_resampled_6800x1400.segy')
    p.add_argument('--angles',         default=','.join(map(str, DEFAULTS['angles'])),
                   help='角度列表，逗号分隔，如 0,5,10')
    p.add_argument('--max_workers',    type=int, default=DEFAULTS['max_workers'])

    # ── augment ──────────────────────────────────────────────
    p = sub.add_parser('augment', help='步骤3: 数据增强（生成SEGY+中文对比图）')
    p.add_argument('--seismic',        default=DEFAULTS['seismic_segy'])
    p.add_argument('--impedance',      default=DEFAULTS['impedance_segy'])
    p.add_argument('--output_root',    default=DEFAULTS['output_root'])
    p.add_argument('--aug_count',      type=int, default=DEFAULTS['aug_count'])
    p.add_argument('--no_noise',       action='store_true')
    p.add_argument('--no_time_shift',  action='store_true')
    p.add_argument('--no_amplitude',   action='store_true')
    p.add_argument('--no_bandpass',    action='store_true')

    # ── cut (A域，地震，3通道) ────────────────────────────────
    p = sub.add_parser('cut', help='步骤4a: 地震切割（A域，3通道编码NPY）')
    p.add_argument('--input',          default=str(Path(DEFAULTS['output_root']) / 'synthetic'))
    p.add_argument('--output',         default=DEFAULTS['trainA_dir'])
    p.add_argument('--patch_size',     type=int, default=DEFAULTS['patch_size'])
    p.add_argument('--stride',         type=int, default=DEFAULTS['stride'])
    p.add_argument('--max_workers',    type=int, default=DEFAULTS['max_workers'])
    p.add_argument('--memory_limit',   type=float, default=8.0)
    p.add_argument('--recursive',      action='store_true', default=True)

    # ── cut_impedance (B域，阻抗，单通道) ─────────────────────
    p = sub.add_parser('cut_impedance', help='步骤4b: 阻抗切割（B域，单通道，归一化到[0,1]）')
    p.add_argument('--input',          default=DEFAULTS['impedance_segy'])
    p.add_argument('--output',         default=DEFAULTS['trainB_dir'])
    p.add_argument('--patch_size',     type=int, default=DEFAULTS['patch_size'])
    p.add_argument('--stride',         type=int, default=DEFAULTS['stride'])

    # ── merge ─────────────────────────────────────────────────
    p = sub.add_parser('merge',
                       help='步骤5: 合并fake_B → 完整图（不依赖padding_info，支持MSE对比）')
    p.add_argument('--input',       default=DEFAULTS['merge_input'],
                   help='fake_B_npy 目录（含 block_*.npy）')
    p.add_argument('--output',      default=DEFAULTS['merge_output'],
                   help='输出 .npy 路径')
    p.add_argument('--patch_size',  type=int, default=DEFAULTS['patch_size'])
    p.add_argument('--stride',      type=int, default=DEFAULTS['stride'])
    p.add_argument('--norm_stats',    default=None,
                   help='norm_stats.npy 或 file_meta_info.npy 路径（用于反归一化，可选，不填则自动查找）')
    p.add_argument('--blocks_index', default=None,
                   help='blocks_index.npy 路径（可来自 trainB 目录，用于网格推算，可选）')
    p.add_argument('--original',     default=None,
                   help='原始SEGY路径（用于MSE对比图，可选）')
    p.add_argument('--merge_mode',   default='gaussian',
                   choices=['gaussian', 'center'],
                   help='合并策略: gaussian=高斯加权(默认), center=保留中心硬边界')

    # ── all ──────────────────────────────────────────────────
    p = sub.add_parser('all', help='一键完整预处理流程 (1→2→3→4a+4b)')
    p.add_argument('--output_root',    default=DEFAULTS['output_root'])
    p.add_argument('--input_dir',      default=DEFAULTS['original_dir'],
                   help='原始未缩放SEGY文件目录')
    p.add_argument('--resampled_dir',  default=str(Path(DEFAULTS['output_root']) / 'resampled'))
    p.add_argument('--original_shape', nargs=2, type=int, default=list(DEFAULTS['original_shape']))
    p.add_argument('--scale_factor',   type=float, default=DEFAULTS['scale_factor'])
    p.add_argument('--dt',             type=float, default=0.001)
    p.add_argument('--dx',             type=float, default=12.5)
    p.add_argument('--vp_file',        default='MODEL_P-WAVE_VELOCITY_1.25m_resampled_6800x1400.segy')
    p.add_argument('--vs_file',        default='MODEL_S-WAVE_VELOCITY_1.25m_resampled_6800x1400.segy')
    p.add_argument('--den_file',       default='MODEL_DENSITY_1.25m_resampled_6800x1400.segy')
    p.add_argument('--angles',         default=','.join(map(str, DEFAULTS['angles'])))
    p.add_argument('--max_workers',    type=int, default=DEFAULTS['max_workers'])
    p.add_argument('--aug_count',      type=int, default=DEFAULTS['aug_count'])
    p.add_argument('--seismic',        default=DEFAULTS['seismic_segy'])
    p.add_argument('--impedance',      default=DEFAULTS['impedance_segy'])
    p.add_argument('--input',          default=str(Path(DEFAULTS['output_root']) / 'synthetic'))
    p.add_argument('--trainA_dir',     default=DEFAULTS['trainA_dir'],
                   help='trainA 输出目录（地震块，可覆盖默认值）')
    p.add_argument('--trainB_dir',     default=DEFAULTS['trainB_dir'],
                   help='trainB 输出目录（阻抗块，可覆盖默认值）')
    p.add_argument('--patch_size',     type=int, default=DEFAULTS['patch_size'])
    p.add_argument('--stride',         type=int, default=DEFAULTS['stride'])
    p.add_argument('--memory_limit',   type=float, default=8.0)
    p.add_argument('--recursive',      action='store_true', default=True)
    p.add_argument('--no_noise',       action='store_true')
    p.add_argument('--no_time_shift',  action='store_true')
    p.add_argument('--no_amplitude',   action='store_true')
    p.add_argument('--no_bandpass',    action='store_true')
    p.add_argument('--show_plot',      action='store_true')
    p.add_argument('--force',          action='store_true',
                   help='强制重跑所有步骤，忽略已有结果')

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # 将 no_xxx 转成正向标志
    if hasattr(args, 'no_noise'):
        args.noise      = not args.no_noise
        args.time_shift = not args.no_time_shift
        args.amplitude  = not args.no_amplitude
        args.bandpass   = not args.no_bandpass

    dispatch = {
        'resample':       run_resample,
        'forward':        run_forward,
        'augment':        run_augment,
        'cut':            run_cut,
        'cut_impedance':  run_cut_impedance,
        'merge':          run_merge,
        'all':            run_all,
    }
    dispatch[args.cmd](args)


if __name__ == '__main__':
    main()
