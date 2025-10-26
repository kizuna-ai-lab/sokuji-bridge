# SimulStreaming 模型选择指南

根据你的硬件和需求选择合适的 Whisper 模型。

## 📊 模型对比表

| 模型 | 参数量 | 文件大小 | GPU 显存 | 准确率 | 速度 | 推荐场景 |
|------|--------|---------|----------|--------|------|----------|
| **tiny** | 39M | 150 MB | ~0.5 GB | ~70% | 最快 | 原型测试 |
| **base** | 74M | 290 MB | ~1 GB | ~75% | 很快 | 资源极度受限 |
| **small** | 244M | 960 MB | **~1-2 GB** | **~85%** | 快 | **显存受限 (<4GB)** ⭐ |
| **medium** | 769M | 1.5 GB | ~2-3 GB | ~88% | 中等 | 平衡选择 |
| **large-v2** | 1.5B | 2.9 GB | ~4-5 GB | ~90% | 慢 | 高质量需求 |
| **large-v3** | 1.5B | 2.88 GB | ~4-6 GB | ~92% | 慢 | 最高质量 |

## 🎯 选择建议

### 场景 1: 显存受限 (< 4GB 可用)
**推荐: small + INT8 量化** ⭐

```yaml
stt:
  provider: "simulstreaming"
  config:
    model_size: "small"
    compute_type: "int8"       # 降低 50% 显存
    beam_size: 1               # Greedy 解码
    audio_max_len: 20.0        # 降低缓冲
```

**性能预期**:
- 显存: ~1-1.5 GB ✅
- 延迟: 1.5-2s ✅
- 准确率: ~85% ✅
- 适合: 实时对话、快速原型

### 场景 2: 平衡性能 (4-6GB 可用)
**推荐: medium + beam search**

```yaml
stt:
  provider: "simulstreaming"
  config:
    model_size: "medium"
    compute_type: "float16"
    beam_size: 3               # Beam search
    audio_max_len: 30.0
```

**性能预期**:
- 显存: ~2-3 GB ✅
- 延迟: <2s ✅
- 准确率: ~88% ✅
- 适合: 生产环境、多数场景

### 场景 3: 高质量优先 (> 6GB 可用)
**推荐: large-v3 + beam search**

```yaml
stt:
  provider: "simulstreaming"
  config:
    model_size: "large-v3"
    compute_type: "float16"
    beam_size: 5               # 更大的 beam
    audio_max_len: 30.0
```

**性能预期**:
- 显存: ~4-6 GB ⚠️
- 延迟: <1.5s ✅
- 准确率: ~92% 🎯
- 适合: 高质量转录、专业应用

### 场景 4: CPU 模式 (无 GPU)
**推荐: small/base + INT8**

```yaml
stt:
  provider: "simulstreaming"
  config:
    model_size: "small"        # 或 base
    device: "cpu"
    compute_type: "int8"       # CPU 必须用 int8
    beam_size: 1               # CPU 建议 greedy
```

**性能预期**:
- 内存: ~2-4 GB
- 延迟: 3-5s ⚠️ (CPU 慢 3-5x)
- 准确率: ~85%
- 适合: 无 GPU 环境、离线处理

## 💡 显存优化技巧

### 1. INT8 量化
降低 ~50% 显存，轻微影响准确率 (~1-2%)

```yaml
compute_type: "int8"
```

### 2. 减小 Beam Size
```yaml
beam_size: 1    # Greedy: 最低显存，快速
beam_size: 3    # 平衡
beam_size: 5    # 高质量，高显存
```

### 3. 降低音频缓冲
```yaml
audio_max_len: 15.0   # 最小 15s (vs 默认 30s)
audio_max_len: 20.0   # 推荐 20s
audio_max_len: 30.0   # 默认
```

### 4. 减少上下文 Token
```yaml
max_context_tokens: 128   # 低显存
max_context_tokens: 224   # 默认
```

## 🔍 你的情况分析

根据错误信息:
```
GPU 总容量: 11.58 GiB
已占用: ~10 GiB (其他进程)
可用: 仅 22 MiB
```

### 解决方案

**选项 1: 清理 GPU 后使用 small 模型** (推荐)

1. 查看 GPU 占用:
```bash
nvidia-smi
```

2. 关闭其他进程 (PID 4890, 4888, 7713)

3. 使用 small 模型:
```bash
python test_simulstreaming.py
```

**选项 2: 直接使用 CPU 模式**

```bash
# 测试脚本会自动检测显存不足并使用 CPU
python test_simulstreaming.py
```

## 📈 性能对比

### 延迟对比 (AlignAtt 低延迟模式)

| 模型 | GPU 延迟 | CPU 延迟 |
|------|---------|---------|
| tiny | 0.8s | 2s |
| base | 1.0s | 2.5s |
| **small** | **1.5s** | **3s** |
| medium | 1.8s | 4s |
| large-v3 | 2.0s | 6s |

### 准确率对比 (中文语音)

| 模型 | Greedy | Beam=3 | Beam=5 |
|------|--------|--------|--------|
| tiny | 68% | 70% | 71% |
| base | 73% | 76% | 77% |
| **small** | **83%** | **86%** | **87%** |
| medium | 87% | 89% | 90% |
| large-v3 | 90% | 92% | 93% |

## 🚀 快速开始

### 1. 检查可用显存
```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
```

### 2. 根据显存选择配置

**< 2GB 可用**:
```bash
cp configs/simulstreaming_lowmem.yaml configs/my_config.yaml
```

**2-4GB 可用**:
```yaml
# 使用 small 模型 + beam search
model_size: "small"
beam_size: 3
```

**> 4GB 可用**:
```yaml
# 使用 medium 或 large-v3
model_size: "medium"  # 或 large-v3
beam_size: 5
```

### 3. 测试
```bash
python test_simulstreaming.py
```

## 📝 最佳实践

### 开发阶段
- 使用 **small** 模型快速迭代
- Greedy 解码 (beam_size=1)
- 较小的音频缓冲

### 测试阶段
- 使用 **medium** 模型
- Beam search (beam_size=3)
- 测试不同场景

### 生产环境
- 根据质量要求选择 **medium** 或 **large-v3**
- Beam search (beam_size=5)
- 完整配置和监控

## ❓ 常见问题

**Q: small 模型准确率够用吗？**

A: 对于大多数中文对话场景，small 模型的 ~85% 准确率已经很好。如果需要专业术语识别，考虑使用 medium 或添加 `init_prompt`。

**Q: INT8 量化会影响质量吗？**

A: 影响很小 (~1-2% 准确率下降)，但显存降低 50%，非常值得。

**Q: 如何在 CPU 和 GPU 之间切换？**

A: 设置 `device: "auto"` 自动检测，或手动指定 `device: "cuda"` / `device: "cpu"`。

**Q: 可以动态切换模型吗？**

A: 需要重新初始化 provider。建议在启动时根据可用资源选择合适的模型。

## 📚 参考资料

- SimulStreaming GitHub: https://github.com/ufal/SimulStreaming
- Whisper 模型性能: https://github.com/openai/whisper#available-models-and-languages
- 显存优化指南: `docs/SIMULSTREAMING_SETUP.md`
