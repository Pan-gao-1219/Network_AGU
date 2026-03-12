import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import segyio
from pathlib import Path
import os
import warnings  # 添加这行
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 屏蔽所有字体警告
warnings.filterwarnings("ignore", message="Glyph.*missing from font.*")  # 添加这行

class MarmousiResampler:
    """
    Marmousi模型重采样器 - 可自定义目标大小，自动画图对比
    """

    def __init__(self,
                 original_shape=(13568, 2688),
                 dt=0.001,  # 原始采样间隔 1ms
                 dx=12.5):  # 原始道间距 12.5m

        self.original_shape = original_shape
        self.original_traces, self.original_samples = original_shape
        self.dt = dt
        self.dx = dx

        # 原始物理尺寸
        self.original_width_m = self.original_traces * dx
        self.original_length_s = self.original_samples * dt

        print("=" * 60)
        print("Marmousi模型重采样器")
        print("=" * 60)
        print(f"原始尺寸: {original_shape[0]}道 × {original_shape[1]}采样点")
        print(f"物理尺寸: {self.original_width_m / 1000:.2f} km × {self.original_length_s:.2f} s")
        print(f"分辨率: {dx:.1f} m/道, {dt * 1000:.1f} ms/采样点")
        print()

    def calculate_target_size(self, scale_factor=None, target_traces=None, target_samples=None):
        """
        计算目标大小
        """
        if scale_factor:
            target_traces = int(self.original_traces * scale_factor)
            target_samples = int(self.original_samples * scale_factor)
        elif target_traces and target_samples:
            scale_factor = (target_traces / self.original_traces +
                            target_samples / self.original_samples) / 2
        else:
            # 默认缩小到1/4
            scale_factor = 0.25
            target_traces = int(self.original_traces * scale_factor)
            target_samples = int(self.original_samples * scale_factor)

        # 新的物理参数
        new_dx = self.original_width_m / target_traces
        new_dt = self.original_length_s / target_samples

        return {
            'traces': target_traces,
            'samples': target_samples,
            'scale_factor': scale_factor,
            'dx': new_dx,
            'dt': new_dt,
            'compression_ratio': self.original_traces * self.original_samples / (target_traces * target_samples)
        }

    def resample_file(self,
                      input_path: str,
                      output_dir: str = "./resampled",
                      scale_factor=None,
                      target_traces=None,
                      target_samples=None,
                      method='zoom',
                      order=1,
                      show_plot=True):
        """
        重采样SEGY文件 - 同时保存NPY和SEGY格式
        """
        input_path = Path(input_path)

        # 读取数据
        print(f"\n正在读取: {input_path.name}")

        # 判断输入文件类型
        if input_path.suffix.lower() in ['.segy', '.sgy']:
            with segyio.open(input_path, "r", ignore_geometry=True) as f:
                data = f.trace.raw[:].astype(np.float32)
                # 保存原始文件的采样间隔和道间距等信息
                try:
                    self.dt = f.bin[segyio.BinField.Interval] / 1e6  # 微秒转秒
                except:
                    pass  # 如果读不到就用初始值
        else:
            data = np.load(input_path)

        # 更新原始形状为实际数据形状
        self.original_shape = data.shape
        self.original_traces, self.original_samples = data.shape
        self.original_width_m = self.original_traces * self.dx
        self.original_length_s = self.original_samples * self.dt

        print(f"数据维度: {data.shape}")
        print(f"数据范围: [{data.min():.4f}, {data.max():.4f}]")

        # 计算目标大小
        target_info = self.calculate_target_size(scale_factor, target_traces, target_samples)
        target_traces = target_info['traces']
        target_samples = target_info['samples']

        print(f"\n目标尺寸: {target_traces}道 × {target_samples}采样点")
        print(f"缩放因子: {target_info['scale_factor']:.3f}")
        print(f"压缩比: {target_info['compression_ratio']:.1f}:1")
        print(f"新分辨率: {target_info['dx']:.1f} m/道, {target_info['dt'] * 1000:.1f} ms/采样点")

        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 重采样
        print(f"\n正在重采样...")

        if method == 'zoom':
            zoom_factor = (target_traces / data.shape[0],
                           target_samples / data.shape[1])
            resampled = zoom(data, zoom_factor, order=order)
        else:
            from skimage.transform import resize
            resampled = resize(data, (target_traces, target_samples),
                               order=order, preserve_range=True, anti_aliasing=True)

        print(f"重采样完成! 新维度: {resampled.shape}")
        print(f"新数据范围: [{resampled.min():.4f}, {resampled.max():.4f}]")

        # === 生成文件名（基于输入文件名）===
        base_filename = input_path.stem  # 获取不带扩展名的文件名
        npy_filename = f"{base_filename}_resampled_{target_traces}x{target_samples}.npy"
        segy_filename = f"{base_filename}_resampled_{target_traces}x{target_samples}.segy"

        # === 保存NPY格式 ===
        npy_path = output_dir / npy_filename
        np.save(npy_path, resampled)
        print(f"\n✅ NPY已保存: {npy_path}")

        # === 保存SEGY格式 ===
        segy_path = output_dir / segy_filename

        # 创建SEGY文件
        spec = segyio.spec()
        spec.sorting = 1  # 道顺序
        spec.format = 1  # IBM浮点
        spec.samples = range(target_samples)  # 采样点数
        spec.tracecount = target_traces  # 道数

        with segyio.create(segy_path, spec) as f:
            # 写入道数据
            for i in range(target_traces):
                f.trace[i] = resampled[i, :]

            # 设置采样间隔（微秒）
            dt_microseconds = int(target_info['dt'] * 1e6)
            f.bin[segyio.BinField.Interval] = dt_microseconds

            # 设置道间距（如果知道）
            # 可以在这里设置更多道头信息

        print(f"✅ SEGY已保存: {segy_path}")

        # 保存元数据
        meta = {
            'original_file': str(input_path),
            'original_shape': data.shape,
            'resampled_shape': resampled.shape,
            'scale_factor': target_info['scale_factor'],
            'target_traces': target_traces,
            'target_samples': target_samples,
            'method': method,
            'order': order,
            'original_dx': self.dx,
            'original_dt': self.dt,
            'new_dx': target_info['dx'],
            'new_dt': target_info['dt'],
            'data_min': float(resampled.min()),
            'data_max': float(resampled.max()),
            'data_mean': float(resampled.mean()),
            'data_std': float(resampled.std()),
            'npy_file': npy_filename,
            'segy_file': segy_filename
        }
        np.save(output_dir / f"{base_filename}_resampled_{target_traces}x{target_samples}.meta.npy", meta)

        # 画图对比（图片名也基于输入文件名）
        if show_plot:
            self.plot_comparison(data, resampled, target_info, output_dir, order, base_filename)

        return resampled, target_info

    def plot_comparison(self, original, resampled, target_info, output_dir, order=1, base_filename="comparison"):
        """
        画对比图 - 图片名基于输入文件名
        """
        fig = plt.figure(figsize=(16, 12))

        # 使用百分位数增强对比度
        lo_orig, hi_orig = np.percentile(original, [2, 98])
        lo_res, hi_res = np.percentile(resampled, [2, 98])

        # 原始数据
        ax1 = plt.subplot(3, 3, 1)
        im1 = ax1.imshow(original.T, aspect='auto', cmap='RdBu',
                         vmin=lo_orig, vmax=hi_orig)
        ax1.set_title(f'原始数据\n{original.shape[0]}×{original.shape[1]}', fontsize=10)
        ax1.set_xlabel('Trace')
        ax1.set_ylabel('Sample')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # 重采样数据
        ax2 = plt.subplot(3, 3, 2)
        im2 = ax2.imshow(resampled.T, aspect='auto', cmap='RdBu',
                         vmin=lo_res, vmax=hi_res)
        ax2.set_title(f'重采样后\n{resampled.shape[0]}×{resampled.shape[1]}', fontsize=10)
        ax2.set_xlabel('Trace')
        ax2.set_ylabel('Sample')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 局部放大对比
        ax3 = plt.subplot(3, 3, 3)

        # 取中间区域，确保不超过边界
        mid_i = min(original.shape[0] // 2, original.shape[0] - 200)
        mid_j = min(original.shape[1] // 2, original.shape[1] - 200)
        size = min(100, mid_i, mid_j)  # 减小size确保不超出

        # 原始数据局部
        orig_zoom = original[mid_i - size:mid_i + size, mid_j - size:mid_j + size]

        # 对应重采样中的区域
        scale = target_info['scale_factor']
        res_i_start = int((mid_i - size) * scale)
        res_i_end = int((mid_i + size) * scale)
        res_j_start = int((mid_j - size) * scale)
        res_j_end = int((mid_j + size) * scale)

        # 确保不超出重采样数据边界
        res_i_start = max(0, res_i_start)
        res_i_end = min(resampled.shape[0], res_i_end)
        res_j_start = max(0, res_j_start)
        res_j_end = min(resampled.shape[1], res_j_end)

        if res_i_end > res_i_start and res_j_end > res_j_start:
            res_zoom = resampled[res_i_start:res_i_end, res_j_start:res_j_end]

            # 将重采样局部放大到和原始局部相同大小以便比较
            from scipy.ndimage import zoom
            zoom_factor_y = orig_zoom.shape[0] / res_zoom.shape[0]
            zoom_factor_x = orig_zoom.shape[1] / res_zoom.shape[1]
            res_zoom_expanded = zoom(res_zoom, (zoom_factor_y, zoom_factor_x), order=1)

            # 计算差异
            diff = np.abs(orig_zoom - res_zoom_expanded)

            im3 = ax3.imshow(diff.T, aspect='auto', cmap='hot')
            ax3.set_title('局部差异热图', fontsize=10)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        else:
            ax3.text(0.5, 0.5, '局部区域太小\n无法对比', ha='center', va='center')
            ax3.set_title('局部差异', fontsize=10)

        # 道对比
        ax4 = plt.subplot(3, 3, 4)
        trace_idx = original.shape[0] // 2
        trace_orig = original[trace_idx, :]
        trace_res = resampled[int(trace_idx * target_info['scale_factor']), :]

        # 插值重采样道以便对比
        trace_res_interp = np.interp(
            np.linspace(0, 1, len(trace_orig)),
            np.linspace(0, 1, len(trace_res)),
            trace_res
        )

        ax4.plot(trace_orig, 'b-', alpha=0.7, label='原始', linewidth=1)
        ax4.plot(trace_res_interp, 'r--', alpha=0.7, label='重采样', linewidth=1)
        ax4.set_title('中间道对比', fontsize=10)
        ax4.set_xlabel('采样点')
        ax4.set_ylabel('振幅')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 频谱对比
        ax5 = plt.subplot(3, 3, 5)
        from scipy import fft

        spec_orig = np.abs(fft.fft(trace_orig))
        spec_res = np.abs(fft.fft(trace_res_interp))

        freq = np.fft.fftfreq(len(trace_orig), self.dt)
        freq = freq[:len(freq) // 2]

        ax5.plot(freq, spec_orig[:len(freq)], 'b-', alpha=0.7, label='原始', linewidth=1)
        ax5.plot(freq, spec_res[:len(freq)], 'r--', alpha=0.7, label='重采样', linewidth=1)
        ax5.set_title('频谱对比', fontsize=10)
        ax5.set_xlabel('频率 (Hz)')
        ax5.set_ylabel('振幅')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 250)

        # 直方图对比
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(original.flatten(), bins=50, alpha=0.5, label='原始', density=True)
        ax6.hist(resampled.flatten(), bins=50, alpha=0.5, label='重采样', density=True)
        ax6.set_title('数值分布对比', fontsize=10)
        ax6.set_xlabel('值')
        ax6.set_ylabel('密度')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # === 修复误差分析部分 ===
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')

        # 创建用于误差分析的公共网格
        # 将重采样数据插值回原始尺寸的网格
        from scipy.ndimage import zoom

        # 计算缩放回原始尺寸的因子
        zoom_back_y = original.shape[0] / resampled.shape[0]
        zoom_back_x = original.shape[1] / resampled.shape[1]

        # 将重采样数据放大回原始尺寸
        resampled_back = zoom(resampled, (zoom_back_y, zoom_back_x), order=1)

        # 裁剪到相同大小（可能会有微小差异）
        min_y = min(original.shape[0], resampled_back.shape[0])
        min_x = min(original.shape[1], resampled_back.shape[1])

        orig_crop = original[:min_y, :min_x]
        resampled_crop = resampled_back[:min_y, :min_x]

        # 计算误差
        mae = np.mean(np.abs(orig_crop - resampled_crop))
        rmse = np.sqrt(np.mean((orig_crop - resampled_crop) ** 2))
        relative_error = (mae / original.std()) * 100 if original.std() > 0 else 0

        info_text = f"""
        重采样信息:
        ------------------------
        原始尺寸: {original.shape[0]}×{original.shape[1]}
        新尺寸: {resampled.shape[0]}×{resampled.shape[1]}
        缩放因子: {target_info['scale_factor']:.3f}
        压缩比: {target_info['compression_ratio']:.1f}:1

        物理参数:
        原始道间距: {self.dx:.1f} m
        新道间距: {target_info['dx']:.1f} m
        原始采样: {self.dt * 1000:.1f} ms
        新采样: {target_info['dt'] * 1000:.1f} ms

        数据统计:
        原始范围: [{original.min():.2f}, {original.max():.2f}]
        新范围: [{resampled.min():.2f}, {resampled.max():.2f}]
        原始均值: {original.mean():.2f}
        新均值: {resampled.mean():.2f}
        """
        ax7.text(0.1, 0.95, info_text, fontsize=8, verticalalignment='top',
                 fontfamily='monospace', transform=ax7.transAxes)

        # 误差统计
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')

        error_text = f"""
        误差分析:
        ------------------------
        MAE: {mae:.4f}
        RMSE: {rmse:.4f}
        相对误差: {relative_error:.2f}%

        插值参数:
        方法: zoom
        阶数: {order}

        建议:
        • 新分辨率 {target_info['dx']:.1f}m, {target_info['dt'] * 1000:.1f}ms
        • 适合CycleGAN训练
        """
        ax8.text(0.1, 0.95, error_text, fontsize=8, verticalalignment='top',
                 fontfamily='monospace', transform=ax8.transAxes)

        # 空白
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        plt.suptitle(f'Marmousi模型重采样对比 (插值阶数={order})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # 保存图片 - 使用base_filename
        plot_filename = f"{base_filename}_comparison_{resampled.shape[0]}x{resampled.shape[1]}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"✅ 对比图已保存: {plot_path}")

        plt.show()

    def interactive_resample(self, input_path):
        """
        交互式重采样
        """
        print("\n" + "=" * 60)
        print("交互式重采样")
        print("=" * 60)

        scales = [1.0, 0.5, 0.25, 0.125, 0.0625]
        results = []

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, scale in enumerate(scales[:6]):
            if scale == 1.0:
                # 原始数据
                if input_path.suffix.lower() in ['.segy', '.sgy']:
                    with segyio.open(input_path, "r", ignore_geometry=True) as f:
                        data = f.trace.raw[:].astype(np.float32)
                else:
                    data = np.load(input_path)

                ax = axes[idx]
                lo, hi = np.percentile(data, [2, 98])
                im = ax.imshow(data.T, aspect='auto', cmap='RdBu', vmin=lo, vmax=hi)
                ax.set_title(f'原始 ({data.shape[0]}×{data.shape[1]})', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                results.append(('original', data.shape))
            else:
                # 重采样
                resampled, info = self.resample_file(
                    input_path,
                    output_dir="./resampled_temp",
                    scale_factor=scale,
                    show_plot=False
                )

                ax = axes[idx]
                lo, hi = np.percentile(resampled, [2, 98])
                im = ax.imshow(resampled.T, aspect='auto', cmap='RdBu', vmin=lo, vmax=hi)
                ax.set_title(f'{scale:.3f} ({resampled.shape[0]}×{resampled.shape[1]})', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                results.append((f'scale_{scale}', resampled.shape))

        plt.suptitle('不同缩放因子对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print("\n各方案对比:")
        for i, (name, shape) in enumerate(results):
            mem = shape[0] * shape[1] * 4 / 1024 ** 2  # MB
            print(f"{i + 1}. {name}: {shape[0]}×{shape[1]}, 内存: {mem:.1f} MB")

        return results


# === 使用示例 ===
if __name__ == "__main__":

    # 1. 初始化重采样器
    resampler = MarmousiResampler(
        original_shape=(13568, 2688),
        dt=0.001,  # 1ms采样
        dx=12.5  # 12.5m道间距
    )

    # 2. 输入文件路径
    input_file = r'C:\PythonProject\cyclegan\model pre+imp\MODEL_AI_1.25m.segy'


    # 执行重采样
    resampled_data, info = resampler.resample_file(
        input_path=input_file,
        output_dir=r"C:\PythonProject\cyclegan\BIGONE\resampled_data",
        scale_factor=0.25,  # 1/4
        method='zoom',
        order=1,  # 线性插值
        show_plot=True
    )

    print("\n" + "=" * 60)
    print("重采样完成!")
    print(f"最终数据: {resampled_data.shape}")
    print(f"保存位置: C:\\PythonProject\\cyclegan\\BIGONE\\resampled_data")
    print("=" * 60)