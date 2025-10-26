# 麦克风实时转录使用指南

使用 SimulStreaming 进行实时语音转录的完整指南。

## 📋 概述

两个示例脚本：

1. **极简版** (`microphone_simulstreaming_simple.py`) - 50 行代码，快速测试
2. **完整版** (`microphone_simulstreaming.py`) - 功能完整，带统计和错误处理

## 🚀 快速开始

### 1. 运行极简版（推荐初次使用）

```bash
python examples/microphone_simulstreaming_simple.py
```

**预期输出**:
```
🎙️  SimulStreaming 实时转录 (极简版)

⏳ 加载模型...
✅ 模型就绪

🎤 初始化麦克风...
✅ 麦克风就绪

🎙️  开始监听... (按 Ctrl+C 停止)

------------------------------------------------------------
📝 [0.8s] 你好
✅ [1.2s] 你好世界
📝 [2.1s] 这是
✅ [2.5s] 这是一个测试

⏹️  停止录音

✅ 清理完成
```

### 2. 运行完整版（带统计信息）

```bash
python examples/microphone_simulstreaming.py
```

**预期输出**:
```
🎙️  SimulStreaming 实时转录
======================================================================
📊 配置:
   - 模型: small
   - 设备: auto
   - 语言: zh
   - 延迟目标: <2秒

⏳ 初始化模型...
✅ 模型加载完成 (1.8s)

🎤 初始化麦克风...
✅ 麦克风就绪

开始监听... (按 Ctrl+C 停止)
======================================================================

📝 [0.75s] 你好 (延迟: 0.753s)
✅ [0.82s] 你好世界 (延迟: 0.821s)
📝 [2.10s] 这是一个 (延迟: 0.892s)
✅ [2.45s] 这是一个测试 (延迟: 0.756s)

⏹️  停止录音...

======================================================================
📊 转录统计:
   - 总结果: 8 个
   - 最终结果: 3 个
   - 平均延迟: 0.805s
   - 运行时长: 45.2s
   - 延迟范围: 0.756s - 0.892s

🧹 清理资源...
✅ 清理完成
======================================================================
```

## ⚙️ 配置说明

### 极简版配置

编辑 `microphone_simulstreaming_simple.py`:

```python
provider = SimulStreamingProvider({
    "model_size": "small",      # 模型大小
    "device": "auto",           # 设备选择
    "frame_threshold": 30,      # 延迟控制
    "language": "zh",           # 语言
})
```

### 完整版配置

编辑 `microphone_simulstreaming.py` 的 `main()` 函数:

```python
# 语言选择
LANGUAGE = "zh"  # zh, en, ja, ko, auto

# 模型选择
MODEL_SIZE = "small"  # tiny, base, small, medium, large-v3

# 设备选择
DEVICE = "auto"  # auto, cuda, cpu

# 麦克风设备 (None 使用默认)
MIC_DEVICE = None  # 或指定设备 ID
```

## 🎛️ 性能调优

### 低延迟模式（<1秒）

```python
config = {
    "model_size": "small",
    "frame_threshold": 25,      # 降低到 25 frames
    "audio_min_len": 0.2,       # 降低到 200ms
    "beam_size": 1,             # Greedy 解码
}
```

**效果**:
- 延迟: 0.6-0.9s ⚡
- 准确率: ~83% 🟡

### 平衡模式（1-2秒）- 默认推荐

```python
config = {
    "model_size": "small",
    "frame_threshold": 30,      # 默认
    "audio_min_len": 0.3,       # 默认
    "beam_size": 1,             # Greedy
}
```

**效果**:
- 延迟: 0.8-1.2s ✅
- 准确率: ~85% ✅

### 高质量模式（2-3秒）

```python
config = {
    "model_size": "medium",     # 更大模型
    "frame_threshold": 40,
    "audio_min_len": 0.5,
    "beam_size": 3,             # Beam search
}
```

**效果**:
- 延迟: 1.5-2.5s 🟡
- 准确率: ~88% 🟢

## 🎤 麦克风设备选择

### 查看可用设备

```python
from src.utils.audio_io import list_audio_devices
list_audio_devices()
```

或使用命令行:
```bash
python -m sounddevice
```

**输出示例**:
```
  0 Built-in Microphone, Core Audio (2 in, 0 out)
  1 USB Audio Device, Core Audio (1 in, 0 out)
* 2 Default, Core Audio (2 in, 2 out)
```

### 指定设备

```python
# 使用设备 1 (USB 麦克风)
mic = MicrophoneInput(device=1, sample_rate=16000, channels=1)
```

或在完整版配置中:
```python
MIC_DEVICE = 1  # 使用 USB 麦克风
```

## 🌐 多语言支持

### 支持的语言

| 语言 | 代码 | 示例 |
|------|------|------|
| 中文 | zh | 你好世界 |
| 英文 | en | Hello world |
| 日文 | ja | こんにちは |
| 韩文 | ko | 안녕하세요 |
| 自动检测 | auto | 自动识别 |

### 切换语言

```python
# 英文转录
LANGUAGE = "en"

# 日文转录
LANGUAGE = "ja"

# 自动检测
LANGUAGE = "auto"
```

## 📊 模型选择对比

| 模型 | 显存 | 延迟 | 准确率 | 推荐场景 |
|------|------|------|--------|----------|
| **tiny** | 0.5GB | 0.5s | 70% | 原型测试 |
| **base** | 1GB | 0.7s | 75% | 资源受限 |
| **small** | 1.5GB | 0.8s | **85%** | **推荐** ⭐ |
| **medium** | 3GB | 1.5s | 88% | 高质量 |
| **large-v3** | 6GB | 2.0s | 92% | 最高质量 |

## 🐛 故障排除

### 问题 1: 麦克风初始化失败

**错误**:
```
❌ 麦克风初始化失败: No default input device
```

**解决**:
```bash
# 1. 查看可用设备
python -m sounddevice

# 2. 指定设备
# 编辑脚本，设置 MIC_DEVICE = <设备ID>
```

### 问题 2: CUDA 显存不足

**错误**:
```
RuntimeError: CUDA out of memory
```

**解决**:
```python
# 方案 1: 使用 CPU
DEVICE = "cpu"

# 方案 2: 使用更小的模型
MODEL_SIZE = "base"  # 或 "tiny"

# 方案 3: 清理其他 GPU 进程
# nvidia-smi
# kill -9 <PID>
```

### 问题 3: 没有转录输出

**可能原因**:
1. 麦克风音量太小
2. 说话太短 (< 300ms)
3. 语言设置错误

**解决**:
```python
# 1. 降低最小音频长度
"audio_min_len": 0.2,  # 降低到 200ms

# 2. 检查麦克风音量
# 系统设置 → 声音 → 输入

# 3. 使用自动语言检测
LANGUAGE = "auto"
```

### 问题 4: 延迟太高

**当前延迟 > 2 秒**

**解决**:
```python
# 1. 降低 frame_threshold
"frame_threshold": 25,  # 默认 30

# 2. 使用 GPU
DEVICE = "cuda"

# 3. 使用更小的模型
MODEL_SIZE = "small"  # 或 "base"

# 4. 使用 Greedy 解码
"beam_size": 1,
```

### 问题 5: Triton 警告

**警告**:
```
UserWarning: Failed to launch Triton kernels...
```

**说明**:
- ✅ 不影响功能
- ✅ 延迟仍然正常
- 可以忽略

如果想消除警告:
```bash
pip install triton==2.0.0
```

## 💡 使用技巧

### 1. 清晰的音频输入

- 使用质量好的麦克风
- 保持安静的环境
- 距离麦克风 10-30cm
- 说话清晰、速度适中

### 2. 优化转录质量

```python
# 添加术语提示
config = {
    "init_prompt": "医学专业术语: 心肌梗塞 脑血栓",  # 帮助识别专业词汇
    "static_init_prompt": "会议参与者: 张医生 李护士",  # 静态上下文
}
```

### 3. 监控性能

```python
# 在转录循环中添加
async for result in provider.transcribe_stream(...):
    latency = time.time() - result.timestamp
    print(f"延迟: {latency:.3f}s")

    if latency > 2.0:
        print("⚠️  延迟过高!")
```

### 4. 保存转录结果

```python
# 保存到文件
with open("transcript.txt", "w", encoding="utf-8") as f:
    async for result in provider.transcribe_stream(...):
        if result.is_final:
            f.write(f"{result.text}\n")
            f.flush()  # 立即写入
```

## 📝 代码示例

### 最小可运行示例（20行）

```python
import asyncio
from src.utils.microphone import MicrophoneInput
from src.providers.stt.simulstreaming_provider import SimulStreamingProvider

async def main():
    # 初始化
    provider = SimulStreamingProvider({"model_size": "small", "language": "zh"})
    await provider.initialize()

    mic = MicrophoneInput(sample_rate=16000, channels=1)
    await mic.start()

    # 转录
    async def audio_stream():
        async for chunk in mic.stream():
            yield chunk

    async for result in provider.transcribe_stream(audio_stream()):
        print(result.text)

asyncio.run(main())
```

### 带错误处理的示例

```python
async def main():
    provider = None
    mic = None

    try:
        # 初始化 provider
        provider = SimulStreamingProvider({"model_size": "small"})
        await provider.initialize()

        # 初始化麦克风
        mic = MicrophoneInput(sample_rate=16000)
        await mic.start()

        # 转录
        async def audio_stream():
            async for chunk in mic.stream():
                yield chunk

        async for result in provider.transcribe_stream(audio_stream()):
            print(f"{result.text}")

    except KeyboardInterrupt:
        print("\n停止")

    except Exception as e:
        print(f"错误: {e}")

    finally:
        # 清理
        if mic:
            await mic.stop()
        if provider:
            await provider.cleanup()
```

## 🚀 高级功能

### 1. 实时翻译

```python
from src.providers.translation.nllb_provider import NLLBProvider

# 初始化翻译器
translator = NLLBProvider({"model_name": "facebook/nllb-200-distilled-600M"})
await translator.initialize()

# 转录 + 翻译
async for result in provider.transcribe_stream(...):
    if result.is_final and result.language == "zh":
        translation = await translator.translate(
            result.text,
            source_lang="zh",
            target_lang="en"
        )
        print(f"中文: {result.text}")
        print(f"English: {translation.translated_text}")
```

### 2. 实时 TTS 播放

```python
from src.providers.tts.piper_provider import PiperProvider

# 初始化 TTS
tts = PiperProvider({"model_path": "voices/zh_CN-huayan-medium.onnx"})
await tts.initialize()

# 转录 + 朗读
async for result in provider.transcribe_stream(...):
    if result.is_final:
        audio = await tts.synthesize(result.text, voice_id="default")
        # 播放音频 (需要额外的播放代码)
```

### 3. WebSocket 实时传输

```python
import websockets

async def websocket_transcription(websocket):
    provider = SimulStreamingProvider({"model_size": "small"})
    await provider.initialize()

    # 接收音频并转录
    async def audio_stream():
        async for message in websocket:
            chunk = AudioChunk(
                data=message,
                sample_rate=16000,
                timestamp=time.time(),
            )
            yield chunk

    # 发送结果
    async for result in provider.transcribe_stream(audio_stream()):
        await websocket.send(result.text)
```

## 📚 参考文档

- SimulStreaming Setup: `docs/SIMULSTREAMING_SETUP.md`
- 模型选择指南: `docs/MODEL_SELECTION_GUIDE.md`
- API 文档: `src/providers/stt/simulstreaming_provider.py`
- 麦克风 API: `src/utils/microphone.py`

## ❓ 常见问题

**Q: 可以同时支持多个语言吗？**

A: 使用 `language="auto"` 自动检测，或者手动切换：
```python
provider.language = "en"  # 运行时切换
```

**Q: 如何提高实时性？**

A: 降低 `frame_threshold` 和 `audio_min_len`:
```python
"frame_threshold": 20,  # 最低可到 15
"audio_min_len": 0.15,  # 最低可到 0.1
```

**Q: 转录结果可以保存吗？**

A: 可以，参考"使用技巧"第 4 点。

**Q: 支持离线使用吗？**

A: 完全支持！模型下载后即可离线运行。

**Q: CPU 模式够用吗？**

A: 可以，但延迟会增加 3-5倍（2-6秒）。

## 🎉 总结

现在你已经掌握了：

✅ 运行实时转录的两种方式（极简版/完整版）
✅ 配置和优化性能
✅ 选择合适的模型和语言
✅ 处理常见问题
✅ 扩展高级功能

开始使用吧！🚀

```bash
python examples/microphone_simulstreaming_simple.py
```
