# VAD (语音活动检测) 修复说明

## 问题描述

运行 `examples/microphone_to_speaker.py` 时，即使没有说话，也会产生重复的虚假转录内容（例如："It's the first time I have been to Toronto"）。

## 原因分析

### Whisper 幻觉现象

这是 OpenAI Whisper 模型的已知问题，称为"幻觉"(Hallucination)：

1. **没有语音活动检测**：麦克风持续捕获音频，包括静音和背景噪音
2. **Whisper 处理静音**：当 Whisper 处理低质量或静音音频时，会生成虚假的转录内容
3. **重复同样内容**：通常会重复生成相同的句子

### 技术原因

配置文件中虽然设置了 VAD 参数：

```yaml
stt:
  optimization:
    enable_vad_filter: true
    vad_threshold: 0.5
```

但是示例代码只传递了 `stt.config`，没有包含 `stt.optimization`：

```python
# 原代码
stt_provider = FasterWhisperProvider(config.stt.config)  # ❌ 缺少 optimization 设置
```

## 解决方案

### 修改内容

在 `examples/microphone_to_speaker.py` 第259-270行，合并 config 和 optimization 设置：

```python
# 修复后的代码
# Merge STT config with optimization settings (especially VAD filter)
stt_config = {
    **config.stt.config,
    "vad_filter": config.stt.optimization.get("enable_vad_filter", False),
    "vad_threshold": config.stt.optimization.get("vad_threshold", 0.5),
}

stt_provider = FasterWhisperProvider(stt_config)
print(f"  Initializing STT...")
print(f"  VAD Filter: {'Enabled' if stt_config.get('vad_filter') else 'Disabled'} "
      f"(threshold: {stt_config.get('vad_threshold', 'N/A')})")
await stt_provider.initialize()
```

### VAD 工作原理

VAD (Voice Activity Detection) 会：

1. **检测语音能量**：分析音频信号的能量水平
2. **过滤静音**：丢弃低于阈值的音频块
3. **只处理语音**：仅将包含实际语音的音频发送给 Whisper

## 运行验证

### 1. 查看启动日志

启用 VAD 后，运行时会显示：

```
🔧 Initializing translation providers...
  Initializing STT...
  VAD Filter: Enabled (threshold: 0.5)
```

### 2. 测试静音

保持安静不说话：
- ✅ 预期：不应该产生任何转录输出
- ❌ 之前：会产生重复的虚假转录

### 3. 测试语音

正常说话：
- ✅ 应该正确转录你说的话
- ✅ 停止说话后应该停止输出

### 4. 观察处理频率

- ✅ 启用 VAD：只有说话时才处理音频
- ❌ 未启用：每2秒左右处理一次（包括静音）

## 调整 VAD 灵敏度

如果遇到以下问题，可以调整 `configs/default.yaml` 中的阈值：

### 问题：说话时没有检测到

**症状**：即使说话也没有转录输出

**解决**：降低 VAD 阈值（更敏感）

```yaml
stt:
  optimization:
    vad_threshold: 0.3  # 从 0.5 降低到 0.3
```

### 问题：静音时仍有虚假转录

**症状**：安静时仍然产生一些转录

**解决**：提高 VAD 阈值（更严格）

```yaml
stt:
  optimization:
    vad_threshold: 0.7  # 从 0.5 提高到 0.7
```

### 阈值范围

- **0.0 - 0.3**：非常敏感，可能检测到微弱声音
- **0.3 - 0.5**：中等敏感（推荐用于安静环境）
- **0.5 - 0.7**：较不敏感（推荐用于嘈杂环境）
- **0.7 - 1.0**：非常不敏感，只检测大声说话

## Pipeline-Level VAD

配置文件中还有 pipeline-level 的 VAD 设置：

```yaml
pipeline:
  vad:
    enabled: true
    model: "silero"  # silero 或 webrtc
    threshold: 0.5
    min_speech_duration_ms: 250
    max_speech_duration_s: 30
    min_silence_duration_ms: 300
```

这个设置用于更高级的语音分段，当前示例使用的是 Whisper 内置的 VAD。未来可以实现 pipeline-level VAD 以获得更好的控制。

## 性能影响

启用 VAD 的好处：

1. **减少计算资源**：不处理静音，降低 CPU/GPU 使用率
2. **提高准确性**：避免 Whisper 对静音产生幻觉
3. **节省成本**：如果使用云 API，减少 API 调用次数
4. **更好的用户体验**：只在实际说话时有输出

## 参考资料

- [Whisper Hallucination Issue](https://github.com/openai/whisper/discussions/679)
- [Faster-Whisper VAD Documentation](https://github.com/SYSTRAN/faster-whisper#vad-filter)
- [Silero VAD](https://github.com/snakers4/silero-vad)

## 相关文件

- `examples/microphone_to_speaker.py`: 实时翻译示例（已修复）
- `configs/default.yaml`: VAD 配置文件
- `src/providers/stt/faster_whisper_provider.py`: VAD 实现
