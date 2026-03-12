# segy_cutter_parallel_auto.py
"""
自动并行SEGY切割器 - 自动检测文件夹中的SEGY文件并并行切割
输出三通道 (H, W, 3)，严格保证数据保真
针对10GB内存优化
"""

import segyio
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
import gc
import psutil
import json
from scipy.signal import hilbert

class AutoParallelSEGYCutter:
    """
    自动并行SEGY切割器
    """

    def __init__(self,
                 input_dir: str,
                 patch_size: int = 256,
                 stride: int = 128,
                 output_base_dir: str = "./cut_results_parallel",
                 norm_range: Tuple[float, float] = (0, 1),
                 normalize: bool = True,  # 默认True，但使用时设为False
                 file_extensions: List[str] = ['.segy', '.sgy'],
                 recursive: bool = True,
                 max_workers: Optional[int] = None,
                 memory_limit_gb: float = 9.0):

        self.input_dir = Path(input_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.output_base_dir = Path(output_base_dir)
        self.norm_range = norm_range
        self.normalize = normalize
        self.file_extensions = file_extensions
        self.memory_limit_gb = memory_limit_gb
        self.recursive = recursive

        # 自动检测SEGY文件
        self.segy_files = self._detect_segy_files()

        if len(self.segy_files) == 0:
            raise ValueError(f"在目录 {input_dir} 中未找到SEGY文件")

        # 计算最优并行数
        self.max_workers = self._calculate_optimal_workers() if max_workers is None else max_workers

        # 创建输出目录
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        self.config = {
            'input_dir': str(input_dir),
            'patch_size': patch_size,
            'stride': stride,
            'output_base_dir': str(output_base_dir),
            'norm_range': norm_range,
            'normalize': normalize,
            'file_extensions': file_extensions,
            'max_workers': self.max_workers,
            'memory_limit_gb': memory_limit_gb,
            'detected_files': [str(f) for f in self.segy_files]
        }

        self._save_config()

        print(f"\n{'=' * 60}")
        print(f"🚀 自动并行SEGY切割器初始化")
        print(f"{'=' * 60}")
        print(f"📁 输入目录: {input_dir}")
        print(f"📊 检测到 {len(self.segy_files)} 个SEGY文件:")
        for i, f in enumerate(self.segy_files[:5]):  # 只显示前5个
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {i + 1}. {f.name} ({size_mb:.1f} MB)")
        if len(self.segy_files) > 5:
            print(f"   ... 还有 {len(self.segy_files) - 5} 个文件")

        print(f"\n⚙️ 配置参数:")
        print(f"   - 块大小: {patch_size}×{patch_size}")
        print(f"   - 步长: {stride}")
        print(f"   - 归一化范围: {norm_range}")
        print(f"   - 并行进程数: {self.max_workers}")
        print(f"   - 内存限制: {memory_limit_gb} GB")
        print(f"   - 输出目录: {output_base_dir}")
        print(f"{'=' * 60}\n")

    def _detect_segy_files(self) -> List[Path]:
        """自动检测目录中的所有SEGY/NPY文件（支持递归搜索子文件夹）"""
        segy_files = []

        # 同时搜索 .segy, .sgy, .npy 文件
        extensions = self.file_extensions + ['.npy']  # 添加 .npy 支持

        if self.recursive:
            for ext in extensions:
                segy_files.extend(self.input_dir.rglob(f"*{ext}"))
                segy_files.extend(self.input_dir.rglob(f"*{ext.upper()}"))
        else:
            for ext in extensions:
                segy_files.extend(self.input_dir.glob(f"*{ext}"))
                segy_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

        # 去重
        segy_files = list(set(segy_files))
        segy_files.sort(key=lambda x: x.stat().st_size)

        return segy_files

    def _calculate_optimal_workers(self) -> int:
        """根据内存和文件数量计算最优并行数"""
        try:
            # 获取可用内存
            available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
            safe_memory = min(available_memory, self.memory_limit_gb)

            # 估计最大文件大小
            max_file_size = max(f.stat().st_size for f in self.segy_files) / (1024 ** 3)  # GB

            # 每个文件处理需要的内存估计（数据大小的3-4倍）
            per_file_memory = max_file_size * 3.5

            # 基于内存的并行数
            workers_by_memory = max(1, int(safe_memory / per_file_memory))

            # 基于CPU核心数的并行数
            cpu_count = mp.cpu_count()

            # 基于文件数量的并行数（不要超过文件数）
            workers_by_files = len(self.segy_files)

            # 取最小值
            optimal_workers = min(workers_by_memory, cpu_count, workers_by_files, 4)  # 最多4个

            print(f"\n📊 自动计算并行数:")
            print(f"   - 可用内存: {available_memory:.1f} GB")
            print(f"   - 安全内存: {safe_memory:.1f} GB")
            print(f"   - 最大文件大小: {max_file_size:.2f} GB")
            print(f"   - 每文件估计内存: {per_file_memory:.2f} GB")
            print(f"   - 基于内存的并行数: {workers_by_memory}")
            print(f"   - CPU核心数: {cpu_count}")
            print(f"   - 文件数量: {workers_by_files}")
            print(f"   - 最终选择: {optimal_workers}")

            return optimal_workers

        except Exception as e:
            print(f"⚠️ 自动计算并行数失败，使用默认值2: {e}")
            return 2

    def _save_config(self):
        """保存配置信息"""
        config_path = self.output_base_dir / '切割配置.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _process_single_file(self, file_path: Path) -> Dict:
        """处理单个文件（在独立进程中执行）"""
        file_start_time = time.time()
        result = {
            'file': str(file_path),
            'success': False,
            'blocks': 0,
            'error': None,
            'file_size_mb': file_path.stat().st_size / (1024 * 1024)
        }

        try:
            print(f"\n  🔄 [{os.getpid()}] 开始处理: {file_path.name}")
            print(f"  🔍 [{os.getpid()}] self.normalize = {self.normalize}")

            # 为每个文件创建独立的输出目录
            file_output_dir = self.output_base_dir / file_path.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # 加载数据
            data = self._load_data(file_path)

            # 打印信息
            if data.ndim == 2:
                print(f"  📊 [{os.getpid()}] {file_path.name}: 维度 {data.shape}, 范围 [{data.min():.4f}, {data.max():.4f}]")
            else:
                print(f"  📊 [{os.getpid()}] {file_path.name}: 维度 {data.shape}, 各通道范围:")
                for c in range(data.shape[2]):
                    ch_data = data[:, :, c]
                    print(f"      通道{c}: [{ch_data.min():.4f}, {ch_data.max():.4f}]")

            # 保存元数据
            self._save_file_metadata(file_output_dir, file_path.name, data)

            # 切割数据（包含填充逻辑）
            blocks_count = self._cut_file_data(data, file_output_dir)

            # 清理内存
            del data
            gc.collect()

            elapsed = time.time() - file_start_time
            result['success'] = True
            result['blocks'] = blocks_count
            result['elapsed'] = elapsed

            print(f"  ✅ [{os.getpid()}] 完成: {file_path.name} - {blocks_count}个块, 耗时 {elapsed:.1f}秒")

        except Exception as e:
            import traceback
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            print(f"  ❌ [{os.getpid()}] 失败: {file_path.name} - {e}")

        return result

    def _load_data(self, file_path: Path) -> np.ndarray:
        """根据文件扩展名加载数据"""
        suffix = file_path.suffix.lower()

        if suffix in ['.segy', '.sgy']:
            # 加载 SEGY 文件
            return self._load_segy(file_path)
        elif suffix == '.npy':
            # 直接加载 NPY 文件
            print(f"  📊 [{os.getpid()}] 加载NPY文件: {file_path.name}")
            data = np.load(file_path).astype(np.float32)
            return data
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def _load_segy(self, file_path: Path) -> np.ndarray:
        """加载SEGY文件（支持多种格式）"""
        try:
            with segyio.open(file_path, "r", ignore_geometry=True) as f:
                print(f"  📊 [{os.getpid()}] 文件信息: tracecount={f.tracecount}, format={f.format}")

                # 方法1: 尝试不同的读取方式
                try:
                    # 首先尝试标准读取
                    data = f.trace.raw[:].astype(np.float32)
                    return data
                except:
                    try:
                        # 方法2: 逐道读取
                        traces = []
                        for i in range(min(f.tracecount, 10)):  # 先读10道测试
                            trace = f.trace[i]
                            print(f"    道 {i} 长度: {len(trace)}")
                            traces.append(trace)

                        # 找到最大长度
                        max_len = max(len(t) for t in traces)

                        # 读取所有道
                        data = np.zeros((f.tracecount, max_len), dtype=np.float32)
                        for i in range(f.tracecount):
                            trace = f.trace[i]
                            data[i, :len(trace)] = trace

                        return data
                    except:
                        # 方法3: 使用ilines方式
                        data = np.array([f.trace[i] for i in range(f.tracecount)])
                        return data.astype(np.float32)

        except Exception as e:
            print(f"  ❌ [{os.getpid()}] 加载失败: {file_path.name}")
            print(f"     错误类型: {type(e).__name__}")
            print(f"     错误信息: {e}")
            raise

    def _save_file_metadata(self, out_dir: Path, filename: str, data: np.ndarray) -> None:
        """保存文件级元数据"""
        meta_info = {
            'source_file': filename,
            'original_shape': data.shape,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'channels': 3,
            'channel_mode': 'replicated',
            'data_min': float(data.min()),
            'data_max': float(data.max()),
            'data_mean': float(data.mean()),
            'data_std': float(data.std()),
            'normalized': self.normalize,
            'norm_range': self.norm_range
        }
        np.save(out_dir / 'file_meta_info.npy', meta_info)

    def _cut_file_data(self, data: np.ndarray, out_dir: Path) -> int:
        """执行单个文件的数据切割（支持2D和3D），并进行对称填充确保覆盖边缘"""
        print(f"  🔍 切割数据 - 输入数据范围: [{data.min():.6f}, {data.max():.6f}]")

        # 判断维度
        if data.ndim == 2:
            n_traces, n_samples = data.shape
            is_2d = True
            print(f"  📊 检测到2D数据: {n_traces}×{n_samples}")
        elif data.ndim == 3:
            n_traces, n_samples, n_channels = data.shape
            is_2d = False
            print(f"  📊 检测到3D数据: {n_traces}×{n_samples}×{n_channels}")
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")

        patch_size = self.patch_size
        stride = self.stride

        # ===== 新增：对称填充以确保边缘覆盖 =====
        # 计算需要填充到的目标尺寸
        target_traces = n_traces
        target_samples = n_samples

        if (target_traces - patch_size) % stride != 0:
            target_traces = ((target_traces - patch_size) // stride + 1) * stride + patch_size
        if (target_samples - patch_size) % stride != 0:
            target_samples = ((target_samples - patch_size) // stride + 1) * stride + patch_size

        # 计算每边填充量（对称填充）
        pad_top = (target_traces - n_traces) // 2
        pad_bottom = target_traces - n_traces - pad_top
        pad_left = (target_samples - n_samples) // 2
        pad_right = target_samples - n_samples - pad_left

        # 执行填充
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            if is_2d:
                # 2D数据填充：axis=(0,1)
                data_padded = np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')
            else:
                # 3D数据填充：axis=(0,1)，通道轴(2)不填充
                data_padded = np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), mode='symmetric')
            print(f"  📏 填充前尺寸: ({n_traces},{n_samples})，填充后尺寸: {data_padded.shape[:2]}，填充量: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        else:
            data_padded = data
            print("  📏 无需填充")

        # 保存填充信息供合并时使用
        padding_info = {
            'original_shape': (n_traces, n_samples) if is_2d else (n_traces, n_samples, n_channels),
            'padded_shape': data_padded.shape,
            'padding': (pad_top, pad_bottom, pad_left, pad_right),
            'is_2d': is_2d
        }
        np.save(out_dir / 'padding_info.npy', padding_info)

        # 后续切割使用填充后的数据
        if is_2d:
            n_traces_padded, n_samples_padded = data_padded.shape
        else:
            n_traces_padded, n_samples_padded, _ = data_padded.shape

        block_id = 0

        # 计算总块数
        n_i = (n_traces_padded - patch_size) // stride + 1
        n_j = (n_samples_padded - patch_size) // stride + 1
        total_blocks = n_i * n_j

        print(f"  预计生成块数: {total_blocks}")

        blocks_info = []

        # 滑窗切割
        for i in range(0, n_traces_padded - patch_size + 1, stride):
            for j in range(0, n_samples_padded - patch_size + 1, stride):

                if is_2d:
                    # 2D数据：需要编码为3通道
                    block_2d = data_padded[i:i + patch_size, j:j + patch_size].copy()
                    block_3d = self._encode_2d_to_3channel(block_2d)
                else:
                    # 3D数据：直接切割
                    block_3d = data_padded[i:i + patch_size, j:j + patch_size, :].copy()

                # 验证块的范围（只打印第一个块）
                if block_id == 0:
                    print(f"  🔍 第一个块形状: {block_3d.shape}")
                    print(f"  🔍 第一个块范围: [{block_3d.min():.6f}, {block_3d.max():.6f}]")
                    if block_3d.ndim == 3:
                        for c in range(block_3d.shape[2]):
                            print(f"     通道{c}范围: [{block_3d[:, :, c].min():.6f}, {block_3d[:, :, c].max():.6f}]")

                # 保存NPY
                block_path = out_dir / f'block_{block_id:06d}.npy'
                np.save(block_path, block_3d)

                # 记录块信息
                blocks_info.append({
                    'block_id': block_id,
                    'file': f'block_{block_id:06d}.npy',
                    'position': (i, j),
                    'shape': block_3d.shape,
                    'dtype': str(block_3d.dtype)
                })

                block_id += 1

        # 保存索引
        np.save(out_dir / 'blocks_index.npy', blocks_info)
        print(f"  ✅ 完成切割，生成 {block_id} 个块")

        return block_id

    def _encode_2d_to_3channel(self, data):
        """将2D数据编码为3通道，保持z-score范围"""
        from scipy.signal import hilbert

        h, w = data.shape

        # 创建3通道输出
        enhanced = np.zeros((h, w, 3), dtype=np.float32)

        # 通道0: 原始z-score数据 (保持范围)
        enhanced[:, :, 0] = data  # 直接使用，不改变范围

        # 计算解析信号用于通道1和2
        analytic_signal = hilbert(data, axis=0)

        # 通道1: 瞬时振幅 (归一化到[0,1])
        amplitude = np.abs(analytic_signal)
        amp_min, amp_max = amplitude.min(), amplitude.max()
        if amp_max - amp_min > 1e-10:
            enhanced[:, :, 1] = (amplitude - amp_min) / (amp_max - amp_min)

        # 通道2: 瞬时相位 (归一化到[0,1])
        phase = np.angle(analytic_signal)
        enhanced[:, :, 2] = (phase + np.pi) / (2 * np.pi)

        return enhanced

    def run(self) -> Dict:
        """
        运行并行切割
        """
        start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"🚀 开始并行切割 {len(self.segy_files)} 个文件")
        print(f"   并行进程数: {self.max_workers}")
        print(f"{'=' * 60}\n")

        # 使用进程池并行处理文件
        results = {
            'success': [],
            'failed': [],
            'total_blocks': 0,
            'file_results': []
        }

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self._process_single_file, file_path): file_path
                for file_path in self.segy_files
            }

            # 等待任务完成
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=3600)  # 1小时超时
                    results['file_results'].append(result)

                    if result['success']:
                        results['success'].append(result['file'])
                        results['total_blocks'] += result['blocks']
                    else:
                        results['failed'].append({
                            'file': result['file'],
                            'error': result['error']
                        })

                except Exception as e:
                    results['failed'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })

        # 计算统计信息
        end_time = time.time()
        total_time = end_time - start_time

        # 生成汇总报告
        self._generate_report(results, total_time)

        return results

    def _generate_report(self, results: Dict, total_time: float):
        """生成汇总报告"""
        print(f"\n{'=' * 60}")
        print(f"📊 切割完成汇总报告")
        print(f"{'=' * 60}")
        print(f"总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")
        print(f"成功文件: {len(results['success'])}/{len(self.segy_files)}")
        print(f"总生成块数: {results['total_blocks']}")

        if results['failed']:
            print(f"\n❌ 失败文件 ({len(results['failed'])}):")
            for f in results['failed']:
                print(f"   - {Path(f['file']).name}: {f['error']}")

        # 保存运行报告
        report = {
            'run_time': datetime.now().isoformat(),
            'total_time': total_time,
            'total_files': len(self.segy_files),
            'success_files': len(results['success']),
            'failed_files': len(results['failed']),
            'total_blocks': results['total_blocks'],
            'config': self.config,
            'results': results['file_results']
        }

        report_path = self.output_base_dir / '运行报告.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📁 输出目录: {self.output_base_dir}")
        print(f"📄 报告文件: {report_path}")
        print(f"{'=' * 60}")

    def verify_output(self) -> Dict:
        """验证输出结果"""
        print(f"\n{'=' * 60}")
        print(f"🔍 验证输出结果")
        print(f"{'=' * 60}")

        verification = {
            'total_files': 0,
            'total_blocks': 0,
            'files_with_blocks': [],
            'errors': []
        }

        for file_dir in self.output_base_dir.iterdir():
            if file_dir.is_dir():
                blocks_dir = file_dir
                blocks_files = list(blocks_dir.glob("block_*.npy"))

                if blocks_files:
                    verification['total_files'] += 1
                    verification['total_blocks'] += len(blocks_files)
                    verification['files_with_blocks'].append(file_dir.name)

                    # 验证第一个块
                    first_block = np.load(blocks_files[0])
                    if first_block.shape[-1] != 3:
                        verification['errors'].append(f"{file_dir.name}: 通道数错误 {first_block.shape}")
                    elif first_block.shape[0] != self.patch_size or first_block.shape[1] != self.patch_size:
                        verification['errors'].append(f"{file_dir.name}: 尺寸错误 {first_block.shape}")

        print(f"验证结果:")
        print(f"  - 成功处理文件数: {verification['total_files']}")
        print(f"  - 总块数: {verification['total_blocks']}")
        print(f"  - 错误数: {len(verification['errors'])}")

        if verification['errors']:
            print(f"  - 错误详情:")
            for err in verification['errors'][:5]:
                print(f"    * {err}")

        return verification


# ============================================================
# 使用示例
# ============================================================

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🚀 自动并行SEGY切割器")
    print("=" * 60)

    # 设置输入目录（包含多个子文件夹的根目录）
    input_dir = r'C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\resampled'

    # 创建切割器
    cutter = AutoParallelSEGYCutter(
        input_dir=input_dir,
        patch_size=256,
        stride=128,
        output_base_dir=r"C:\PythonProject\pg_Cyclegan_predata\data_pipeline_output\cut\0度",
        norm_range=(0, 1),
        normalize=True,
        recursive=True,  # 确保开启递归搜索
        max_workers=None,  # 自动计算
        memory_limit_gb=10.0
    )

    # 运行并行切割
    results = cutter.run()

    # 验证输出
    verification = cutter.verify_output()

    print("\n" + "=" * 60)
    print("✅ 处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    # Windows需要这行
    mp.freeze_support()
    main()