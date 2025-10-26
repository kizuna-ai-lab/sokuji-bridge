#!/usr/bin/env python3
"""
麦克风测试脚本

快速测试麦克风是否正常工作，无需加载 STT 模型
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.microphone import MicrophoneInput


async def test_microphone():
    """测试麦克风音频捕获"""

    print("\n🎤 麦克风测试")
    print("=" * 60)

    # 列出可用设备
    print("\n📋 可用音频设备:")
    print("-" * 60)
    import sounddevice as sd
    print(sd.query_devices())
    print("-" * 60)

    # 初始化麦克风
    print("\n⏳ 初始化麦克风...")
    mic = MicrophoneInput(
        device=None,  # 使用默认设备
        sample_rate=16000,
        channels=1,
        block_size=1024,
    )

    try:
        await mic.start()
        print("✅ 麦克风启动成功!")
        print()
        print("🎙️  正在监听... (按 Ctrl+C 停止)")
        print("💬 请说话，我会显示音量级别")
        print()

        chunk_count = 0
        max_duration = 10  # 最多测试 10 秒

        start_time = asyncio.get_event_loop().time()

        async for chunk in mic.stream():
            chunk_count += 1

            # 计算音量 (RMS)
            audio_array = np.frombuffer(chunk.data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array**2))
            volume_db = 20 * np.log10(rms + 1e-10)

            # 显示音量条
            bar_length = int((volume_db + 80) / 80 * 50)  # -80dB to 0dB
            bar = "█" * max(0, min(bar_length, 50))

            print(f"\r🔊 音量: {volume_db:6.1f} dB [{bar:<50}]", end="", flush=True)

            # 检查是否超时
            if asyncio.get_event_loop().time() - start_time > max_duration:
                print("\n\n⏱️  测试时间到 (10秒)")
                break

    except KeyboardInterrupt:
        print("\n\n⏹️  用户停止")

    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        print("\n💡 故障排除:")
        print("   1. 检查麦克风是否连接")
        print("   2. 检查系统麦克风权限")
        print("   3. 尝试指定设备: device=<设备ID>")
        return False

    finally:
        await mic.stop()
        print("\n✅ 麦克风已停止")

    print()
    print("=" * 60)
    print(f"📊 统计: 接收了 {chunk_count} 个音频块")
    print(f"   每块: {chunk.duration_ms:.1f}ms")
    print(f"   总计: {chunk_count * chunk.duration_ms / 1000:.1f}s")
    print()
    print("✅ 麦克风测试成功!")
    print()
    print("💡 下一步:")
    print("   运行实时转录: python examples/microphone_simulstreaming_simple.py")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = asyncio.run(test_microphone())
    sys.exit(0 if success else 1)
