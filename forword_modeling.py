import segyio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")

# === 4. 向量化Aki-Richards ===
def aki_richards_vectorized(vp, vs, rho, angle_deg):
    angle_rad = np.radians(angle_deg)
    tan_angle_squared = np.tan(angle_rad) ** 2

    d_vp = np.diff(vp, axis=1, prepend=vp[:, 0:1])
    d_vs = np.diff(vs, axis=1, prepend=vs[:, 0:1])
    d_rho = np.diff(rho, axis=1, prepend=rho[:, 0:1])

    avg_vp = (vp + np.roll(vp, -1, axis=1)) / 2
    avg_vs = (vs + np.roll(vs, -1, axis=1)) / 2
    avg_rho = (rho + np.roll(rho, -1, axis=1)) / 2

    eps = 1e-10
    avg_vp = np.maximum(avg_vp, eps)
    avg_vs = np.maximum(avg_vs, eps)
    avg_rho = np.maximum(avg_rho, eps)

    term1 = 0.5 * (d_rho / avg_rho + d_vp / avg_vp)
    term2 = d_vp / avg_vp
    term3 = 4 * (avg_vs ** 2 / avg_vp ** 2) * (d_rho / avg_rho + 2 * d_vs / avg_vs)

    Rc = term1 + tan_angle_squared * (term2 - term3)

    return Rc


def ricker(f, length, dt):
    t0 = np.arange(-length / 2, length / 2, dt)
    y = (1.0 - 2 * (np.pi * f * t0) ** 2) * np.exp(- (np.pi * f * t0) ** 2)
    return t0, y


def process_angle(angle, vp, vs, rho, del_D, n_xl, n_sam, output_dir=None):
    """处理单个角度（独立任务）"""
    print(f"开始处理角度 {angle}°...")

    # 反射系数
    Rc_aki = aki_richards_vectorized(vp, vs, rho, angle)

    # 双程走时
    vp_safe = np.where(vp == 0, np.nan, vp)
    dt_iterval = 2 * del_D / 1000.0 / vp_safe
    dt_iterval = np.nan_to_num(dt_iterval, nan=0.0, posinf=0.0, neginf=0.0)
    TWT = np.cumsum(dt_iterval, axis=1)

    # Ricker子波
    f = 20
    length = 0.512
    _, wavelet = ricker(f, length, del_D / 1e6)

    # 合成地震
    synthetic_aki = np.zeros_like(Rc_aki)
    for i in range(n_xl):
        synthetic_aki[i] = np.convolve(wavelet, Rc_aki[i], mode='same')

    # 绘图
    clip_percentile = 98
    vm = np.percentile(np.abs(synthetic_aki), clip_percentile)

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 1, 1)
    extent = [1, n_xl, TWT[0][-1] * 1000, TWT[0][0] * 1000]
    im = ax.imshow(synthetic_aki.T, cmap='seismic', vmin=-vm, vmax=vm, aspect='auto', extent=extent)
    ax.set_title(f'合成地震剖面（Aki-Richards, 入射角={angle}°）')
    ax.set_xlabel('CDP 道号')
    ax.set_ylabel('双程走时 [ms]')
    plt.colorbar(im, ax=ax, label='振幅')
    print(f"时间范围: {TWT[0][0] * 1000:.2f} ms 到 {TWT[0][-1] * 1000:.2f} ms")

    # 确定输出目录
    if output_dir is None:
        output_dir = r'C:\PythonProject\cyclegan\BIGONE\reseismic2'

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存图像 - 使用传入的output_dir
    image_save_path = os.path.join(output_dir, f'{angle}度处理.png')
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 保存SEGY - 使用传入的output_dir
    segy_save_path = os.path.join(output_dir, f'{angle}度处理.segy')

    spec = segyio.spec()
    spec.tracecount = n_xl
    spec.samples = np.arange(n_sam)
    spec.format = 5

    with segyio.create(segy_save_path, spec) as dst:
        dst.text[0] = f'{angle} degree Aki-Richards Synthetic Seismic'
        dst.bin[segyio.BinField.Interval] = del_D
        dst.bin[segyio.BinField.Samples] = n_sam
        for i in range(n_xl):
            dst.trace[i] = synthetic_aki[i]

    print(f"角度 {angle}° 完成！文件保存在: {output_dir}")
    return angle

# === 5. 并行处理所有角度（直接运行时使用）===
if __name__ == "__main__":
    # ── 在这里修改输入文件路径和参数 ──────────────────────────
    _vp_path  = r'C:\PythonProject\cyclegan\BIGONE\resampled_data\MODEL_P-WAVE_VELOCITY_1.25m_resampled_3400x700.segy'
    _vs_path  = r'C:\PythonProject\cyclegan\BIGONE\resampled_data\MODEL_S-WAVE_VELOCITY_1.25m_resampled_3400x700.segy'
    _den_path = r'C:\PythonProject\cyclegan\BIGONE\resampled_data\MODEL_DENSITY_1.25m_resampled_3400x700.segy'
    _output_dir = r'C:\PythonProject\cyclegan\BIGONE\reseismic2'
    _angles = list(range(0, 31, 5))   # [0,5,10,15,20,25,30]

    # 加载数据
    with segyio.open(_vp_path,  "r", ignore_geometry=True) as f: vp  = f.trace.raw[:].astype(np.float32)
    with segyio.open(_vs_path,  "r", ignore_geometry=True) as f: vs  = f.trace.raw[:].astype(np.float32)
    with segyio.open(_den_path, "r", ignore_geometry=True) as f:
        rho   = f.trace.raw[:].astype(np.float32)
        del_D = f.bin[segyio.BinField.Interval]
    n_xl, n_sam = rho.shape

    print(f"实际数据维度: {n_xl}道 × {n_sam}采样点")
    print(f"采样间隔: {del_D} μs")
    print(f"开始并行处理 {len(_angles)} 个角度...")
    print(f"CPU核心数: {os.cpu_count()}")

    with ProcessPoolExecutor(max_workers=4) as executor:
        func = partial(process_angle, vp=vp, vs=vs, rho=rho,
                       del_D=del_D, n_xl=n_xl, n_sam=n_sam,
                       output_dir=_output_dir)
        results = list(executor.map(func, _angles))

    print("\n" + "=" * 50)
    print("所有角度处理完成！")
    print(f"生成文件位置: {_output_dir}")
    print("=" * 50)


def run_forward_modeling(vp_path, vs_path, den_path, output_dir, angles, max_workers=4):
    """正演模拟函数接口"""
    import segyio
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    print(f"正演输出目录: {output_dir}")

    # 加载数据
    vp = segyio.open(vp_path, ignore_geometry=True).trace.raw[:]
    vs = segyio.open(vs_path, ignore_geometry=True).trace.raw[:]
    rho = segyio.open(den_path, ignore_geometry=True).trace.raw[:]

    # 获取参数
    n_xl, n_sam = rho.shape
    del_D = segyio.open(den_path, ignore_geometry=True).bin[segyio.BinField.Interval]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 并行处理 - 传入output_dir
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        func = partial(process_angle, vp=vp, vs=vs, rho=rho,
                       del_D=del_D, n_xl=n_xl, n_sam=n_sam,
                       output_dir=output_dir)  # 传入output_dir
        results = list(executor.map(func, angles))

    print(f"\n✅ 所有角度处理完成！结果保存在: {output_dir}")
    return results