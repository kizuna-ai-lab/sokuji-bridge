#!/usr/bin/env python3
"""
GPU 优化版本的 SimulStreaming 测试

在显存有限的情况下使用更小的模型和优化配置
"""

import asyncio
import numpy as np
import time
import torch
from loguru import logger

# Setup logging
logger.add("simulstreaming_gpu_test.log", level="DEBUG")

from src.providers.base import AudioChunk
from src.providers.stt.simulstreaming_provider import SimulStreamingProvider


async def test_initialization():
    """Test provider initialization with GPU memory optimization"""
    logger.info("=" * 60)
    logger.info("Test: GPU Memory Optimized Initialization")
    logger.info("=" * 60)

    # 检查 GPU 可用显存
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_free = (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated(0)) / 1024**3
        logger.info(f"GPU 总显存: {gpu_mem:.2f} GiB")
        logger.info(f"GPU 可用显存: {gpu_free:.2f} GiB")

        # 清理缓存
        torch.cuda.empty_cache()
        logger.info("已清理 CUDA 缓存")

    # 显存优化配置
    config = {
        # 使用 small 模型 (~1GB vs large-v3 ~3GB)
        "model_size": "small",
        "model_path": "./small.pt",

        # GPU 配置
        "device": "cuda",
        "compute_type": "int8",  # INT8 量化降低 ~50% 显存

        # AlignAtt 配置
        "frame_threshold": 30,
        "beam_size": 1,  # Greedy 解码降低显存

        # 音频缓冲配置
        "audio_max_len": 20.0,  # 降低最大缓冲 (30s → 20s)
        "audio_min_len": 0.3,
        "min_chunk_size": 0.3,

        # 禁用 VAD 简化测试
        "vad_enabled": False,

        # 语言
        "language": "zh",
    }

    provider = SimulStreamingProvider(config)

    try:
        logger.info("初始化 provider (medium 模型 + INT8 量化)...")
        start = time.time()
        await provider.initialize()
        elapsed = time.time() - start

        logger.info(f"✅ 初始化成功，耗时 {elapsed:.2f}s")
        logger.info(f"Provider: {provider}")

        # 检查显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU 已分配显存: {allocated:.2f} GiB")
            logger.info(f"GPU 已保留显存: {reserved:.2f} GiB")

        # Health check
        is_healthy = await provider.health_check()
        logger.info(f"健康检查: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")

        return provider

    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        raise


async def test_quick_transcription(provider: SimulStreamingProvider):
    """快速转录测试"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Quick Transcription")
    logger.info("=" * 60)

    # 生成 1 秒静音测试
    sample_rate = 16000
    duration_s = 1.0
    audio_data = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    audio_int16 = (audio_data * 32768).astype(np.int16)

    chunk = AudioChunk(
        data=audio_int16.tobytes(),
        sample_rate=sample_rate,
        timestamp=time.time(),
        channels=1,
        format="int16",
    )

    try:
        start = time.time()
        result = await provider.transcribe(chunk, language="zh")
        elapsed = time.time() - start

        logger.info(f"⏱️ 转录耗时: {elapsed:.3f}s")
        logger.info(f"📝 结果: {result}")
        logger.info(f"✅ 转录成功")

    except Exception as e:
        logger.error(f"❌ 转录失败: {e}")
        raise


async def test_cleanup(provider: SimulStreamingProvider):
    """清理测试"""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Cleanup")
    logger.info("=" * 60)

    try:
        await provider.cleanup()

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("✅ 清理成功")

    except Exception as e:
        logger.error(f"❌ 清理失败: {e}")
        raise


async def main():
    """运行优化测试"""
    logger.info("🚀 SimulStreaming GPU 优化测试")
    logger.info("=" * 60)

    try:
        # 初始化
        provider = await test_initialization()

        # 快速转录
        await test_quick_transcription(provider)

        # 清理
        await test_cleanup(provider)

        logger.info("\n" + "=" * 60)
        logger.info("🎉 所有测试通过!")
        logger.info("=" * 60)

        logger.info("\n💡 提示:")
        logger.info("- 如果仍然显存不足，请关闭其他 GPU 进程")
        logger.info("- 或者使用 CPU 模式: device='cpu'")
        logger.info("- medium 模型显存需求: ~1.5-2GB")
        logger.info("- large-v3 模型显存需求: ~3-4GB")

    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error(f"❌ 测试失败: {e}")
        logger.error(f"{'=' * 60}")

        logger.error("\n🔧 故障排除:")
        logger.error("1. 关闭其他占用 GPU 的进程")
        logger.error("2. 使用 device='cpu' 切换到 CPU 模式")
        logger.error("3. 使用更小的模型: 'small' 或 'base'")
        logger.error("4. 设置环境变量: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        raise


if __name__ == "__main__":
    asyncio.run(main())
