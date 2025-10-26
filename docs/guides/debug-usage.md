# Debug 模式使用说明

## 快速开始

### 启用 Debug 模式

运行示例时添加 `--debug` 参数：

```bash
python examples/microphone_to_speaker.py --debug
```

### 启用 Debug 模式 + 保存音频

```bash
python examples/microphone_to_speaker.py --debug --save-audio
```

### 自定义 Debug 输出目录

```bash
python examples/microphone_to_speaker.py --debug --debug-dir ./my_debug_session
```

## 完整示例

```bash
# 运行 60 秒的翻译，启用 debug 并保存音频
python examples/microphone_to_speaker.py \
    --source zh \
    --target en \
    --duration 60 \
    --debug \
    --save-audio
```

## 启动时的输出

启用 debug 模式后，程序会显示：

```
======================================================================
📊 DEBUG MODE ENABLED
======================================================================
Session ID: 20250120_153022
Output Directory: /home/user/sokuji-bridge/debug_output

Debug files will be saved to:
  📝 Events:  /home/user/sokuji-bridge/debug_output/text/events_20250120_153022.jsonl
  📊 Metrics: /home/user/sokuji-bridge/debug_output/metrics/metrics_20250120_153022.json
  🔊 Audio:   /home/user/sokuji-bridge/debug_output/audio/*.wav
======================================================================

======================================================================
🌉 Sokuji-Bridge - Real-time Translation Demo
======================================================================
Source Language: zh
Target Language: en
Duration: 60 seconds
...
```

## Debug 文件说明

### 1. Events 文件 (JSONL 格式)

**位置**: `debug_output/text/events_<session_id>.jsonl`

**内容**: 每行一个 JSON 对象，记录转录和翻译事件

```json
{"timestamp": "2025-01-20T15:30:22.123", "type": "transcription", "data": {"chunk_id": 1, "text": "你好", "language": "zh"}}
{"timestamp": "2025-01-20T15:30:22.456", "type": "translation", "data": {"chunk_id": 1, "source": "你好", "target": "Hello", "target_language": "en"}}
```

**用途**:
- 查看所有转录和翻译的文本内容
- 验证语言识别是否正确
- 检查翻译质量

### 2. Metrics 文件 (JSON 格式)

**位置**: `debug_output/metrics/metrics_<session_id>.json`

**内容**: 性能指标和统计信息

```json
{
  "session_id": "20250120_153022",
  "start_time": "2025-01-20T15:30:22.000",
  "end_time": "2025-01-20T15:31:22.000",
  "total_chunks": 15,
  "chunks": [
    {
      "chunk_id": 1,
      "timestamp": "2025-01-20T15:30:22.123",
      "stt_latency_ms": 150.5,
      "translation_latency_ms": 45.2,
      "tts_latency_ms": 320.1,
      "total_latency_ms": 515.8
    }
  ],
  "averages": {
    "stt_latency_ms": 155.3,
    "translation_latency_ms": 42.8,
    "tts_latency_ms": 310.5,
    "total_latency_ms": 508.6
  }
}
```

**用途**:
- 性能分析和优化
- 识别瓶颈（STT/翻译/TTS）
- 追踪性能趋势

### 3. Audio 文件 (WAV 格式)

**位置**: `debug_output/audio/*.wav`

**格式**: 16-bit PCM, 16kHz, Mono

**文件命名**: `output_<session_id>_<chunk_number>.wav`

**用途**:
- 验证 TTS 输出质量
- 听取翻译后的音频
- 检查发音和流畅度

**注意**: 只有使用 `--save-audio` 参数时才会保存音频文件

## 分析 Debug 输出

### 使用分析脚本

```bash
# 分析最新的 debug 会话
python examples/analyze_debug_output.py

# 分析指定目录
python examples/analyze_debug_output.py ./my_debug_session
```

### 分析输出示例

```
📁 Analyzing debug session:
  Events: events_20250120_153022.jsonl
  Metrics: metrics_20250120_153022.json

======================================================================
📊 PERFORMANCE METRICS
======================================================================
Session ID: 20250120_153022
Start Time: 2025-01-20T15:30:22.000
End Time: 2025-01-20T15:31:22.000
Total Chunks: 15

Average Latencies:
  STT:         155.3 ms
  Translation: 42.8 ms
  TTS:         310.5 ms
  Total:       508.6 ms

Latency Range:
  Min: 450.2 ms
  Max: 620.5 ms

======================================================================
📝 TRANSLATION EVENTS
======================================================================
Total Transcriptions: 15
Total Translations: 15

Transcription → Translation Pairs:
----------------------------------------------------------------------

[1] Chunk 1
  Source (zh): 你好
  Target (en): Hello

[2] Chunk 2
  Source (zh): 今天天气很好
  Target (en): The weather is nice today

...

======================================================================
🔊 AUDIO FILES
======================================================================
Total Audio Files: 15
Total Size: 2.34 MB
Audio Directory: /home/user/sokuji-bridge/debug_output/audio

======================================================================
✅ Analysis Complete
======================================================================
```

## 手动检查文件

### 查看 Events 日志

```bash
# 查看所有事件
cat debug_output/text/events_*.jsonl

# 使用 jq 格式化输出
cat debug_output/text/events_*.jsonl | jq '.'

# 只查看转录内容
cat debug_output/text/events_*.jsonl | jq 'select(.type == "transcription") | .data.text'

# 只查看翻译内容
cat debug_output/text/events_*.jsonl | jq 'select(.type == "translation") | .data.target'
```

### 查看 Metrics

```bash
# 查看完整 metrics
cat debug_output/metrics/metrics_*.json | jq '.'

# 查看平均延迟
cat debug_output/metrics/metrics_*.json | jq '.averages'

# 查看特定 chunk 的延迟
cat debug_output/metrics/metrics_*.json | jq '.chunks[] | select(.chunk_id == 1)'
```

### 播放音频文件

```bash
# 播放所有输出音频
for file in debug_output/audio/*.wav; do
    echo "Playing: $file"
    aplay "$file"  # Linux
    # afplay "$file"  # macOS
done

# 查看音频文件信息
file debug_output/audio/*.wav
```

## 常见用途

### 1. 验证翻译质量

```bash
# 运行 debug 模式
python examples/microphone_to_speaker.py --debug

# 查看所有翻译结果
cat debug_output/text/events_*.jsonl | \
    jq 'select(.type == "translation") | "\(.data.source) → \(.data.target)"'
```

### 2. 性能优化

```bash
# 收集性能数据
python examples/microphone_to_speaker.py --debug --duration 120

# 分析瓶颈
python examples/analyze_debug_output.py

# 根据结果调整配置
```

### 3. 问题排查

```bash
# 启用 debug 模式重现问题
python examples/microphone_to_speaker.py --debug --save-audio

# 检查 events 日志定位问题
cat debug_output/text/events_*.jsonl | jq '.'

# 听取音频验证输出
aplay debug_output/audio/*.wav
```

### 4. 对比不同配置

```bash
# 配置 A
python examples/microphone_to_speaker.py --debug --debug-dir ./config_a

# 配置 B (修改配置文件后)
python examples/microphone_to_speaker.py --debug --debug-dir ./config_b

# 对比性能
python examples/analyze_debug_output.py ./config_a > config_a_report.txt
python examples/analyze_debug_output.py ./config_b > config_b_report.txt
diff config_a_report.txt config_b_report.txt
```

## 命令行参数完整列表

```bash
python examples/microphone_to_speaker.py [OPTIONS]

Options:
  --source LANG          源语言代码 (默认: zh)
  --target LANG          目标语言代码 (默认: en)
  --duration SECONDS     运行时长（秒） (默认: 30)
  --debug                启用 debug 模式
  --debug-dir PATH       Debug 输出目录 (默认: ./debug_output)
  --save-audio           保存输出音频文件 (仅在 --debug 时有效)
  --list-devices         列出可用音频设备并退出
```

## 清理 Debug 文件

```bash
# 删除特定会话
rm -rf debug_output/text/events_20250120_*.jsonl
rm -rf debug_output/metrics/metrics_20250120_*.json
rm -rf debug_output/audio/output_20250120_*.wav

# 删除所有 debug 输出
rm -rf debug_output/

# 只保留最新的 5 个会话
cd debug_output/metrics
ls -t metrics_*.json | tail -n +6 | xargs rm -f
cd ../../
```

## 最佳实践

1. **定期清理**: Debug 文件会占用磁盘空间，定期清理旧的会话
2. **合理使用 --save-audio**: 音频文件较大，只在需要时启用
3. **使用有意义的目录名**: 使用 `--debug-dir` 为不同测试创建独立目录
4. **分析后再删除**: 运行分析脚本生成报告后再删除原始文件
5. **版本控制排除**: 在 `.gitignore` 中添加 `debug_output/`

## 示例工作流

```bash
# 1. 运行测试会话
python examples/microphone_to_speaker.py \
    --debug \
    --save-audio \
    --debug-dir ./test_$(date +%Y%m%d_%H%M%S) \
    --duration 60

# 2. 分析结果
python examples/analyze_debug_output.py ./test_*

# 3. 检查特定问题
cat ./test_*/text/events_*.jsonl | jq 'select(.data.text | contains("错误"))'

# 4. 保存报告
python examples/analyze_debug_output.py ./test_* > test_report.txt

# 5. 清理（可选）
# rm -rf ./test_*
```

## 相关文档

- [DEBUG_MODE.md](DEBUG_MODE.md) - Debug 模式详细说明
- [VAD_FIX.md](VAD_FIX.md) - VAD 配置和故障排查
- [AUDIO_IO_SUMMARY.md](../AUDIO_IO_SUMMARY.md) - 音频 I/O 文档
