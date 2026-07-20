# 大模型训练显存碎片分析原型

这是一个实验室级轻量原型，针对 PyTorch native CUDA caching allocator：

- `phases.py`：用结构化 `record_function`/NVTX 标记训练逻辑阶段。
- `collector.py`：指定 rank 采完整 memory snapshot，其余 rank 记录轻量统计。
- `analyzer.py`：拆分 cache 与碎片、尝试精确重放，并为事件关联逻辑阶段。
- `megatron_instrumentation_example.py`：Megatron 调度循环中的逻辑插桩位置。

## 1. Megatron 自动 patch

自动 patch 会加入 collector 启动、iteration 采样，以及初始化、pipeline、梯度/参数同步和优化器阶段标记。先查询当前工具支持的 Megatron-LM 版本：

```bash
megatron-memfrag-patch --list-supported
```

当前支持列表：

| Megatron 版本 | Commit | 源码日期 | 验证状态 |
| --- | --- | --- | --- |
| `core_r0.13.0` | `c550cf6c41c31cd3ec72e05c25ea0c979f2b6631` | 2025-07-25 | apply、Python AST 解析及 revert 通过 |

完整兼容性说明见 [`SUPPORTED_MEGATRON_VERSIONS.md`](../SUPPORTED_MEGATRON_VERSIONS.md)。

最简使用流程：

```bash
# 检查 commit 和当前 patch 状态，不修改源码
megatron-memfrag-patch /path/to/Megatron-LM --check

# 生成并审阅补丁内容，不修改源码
megatron-memfrag-patch /path/to/Megatron-LM --dry-run > memtrace.patch

# 应用插桩
megatron-memfrag-patch /path/to/Megatron-LM --apply
```

需要撤销时执行：

```bash
megatron-memfrag-patch /path/to/Megatron-LM --revert
```

`--apply` 会拒绝修改目标文件已经存在未提交改动的工作树。非支持列表中的 commit 默认拒绝；实验性使用 `--force-version` 前必须先检查 `--dry-run`，且所有源码结构锚点仍需准确匹配。

## 2. 训练侧手工接入

将本目录的父目录加入 `PYTHONPATH`，在首次 CUDA allocation 前调用：

```python
from memory_fragmentation import get_collector

collector = get_collector()
collector.start()
```

典型环境变量：

```bash
export MEMORY_TRACE_RANKS=0,255
export MEMORY_TRACE_END_ITER=20
export MEMORY_TRACE_MAX_ENTRIES=1000000
export MEMORY_TRACE_DIR=memory-traces
```

在 pipeline scheduler 已经知道阶段、microbatch 和 model chunk 的地方加入：

```python
from memory_fragmentation import memory_phase

with memory_phase(
    "pipeline/steady/forward",
    iteration=iteration,
    microbatch=microbatch_id,
    direction="forward",
    model_chunk=model_chunk_id,
    pp_stage=pp_rank,
):
    forward_step(...)
```

每个 iteration 结束调用：

```python
collector.sample(iteration, pp_stage=pp_rank, tp_rank=tp_rank, ep_rank=ep_rank)
```

若训练入口捕获异常，可在重新抛出异常前调用：

```python
try:
    train()
except BaseException as exc:
    collector.dump_on_oom(exc)
    raise
```

完整示例见 `megatron_instrumentation_example.py`。对异步 grad/parameter sync，分别标记启动和完成函数；不要根据 tracer 自己重新计算 pipeline 阶段。

## 3. 离线分析

安装仓库后执行：

```bash
megatron-memfrag analyze_input.pickle -o analysis --top 30 --top-k 10
```

可选的 Top-K 过滤参数：

```bash
--fragmentation-threshold 0.10  # 碎片率至少 10%
--min-reserved-mib 1024        # 忽略 reserved 小于 1 GiB 的启动状态
--min-stranded-mib 64          # stranded free 至少 64 MiB
```

这些过滤条件严格生效；无匹配时 Dashboard 保留时间线，但 Top-K 列表为空。

输出：

- `summary.json`：reserved、active、releasable cache、stranded free、internal waste。
- `replay.json`：历史模式、精确重放失败原因、反向重放假设和可用性。
- `incidents.json` / `incidents.csv`：fragment-created、failed-fit 和 segment-pinned 事件及逻辑阶段。
- `dashboard.html`：单文件交互式窗口，查看碎片率时间线和默认 Top-10 allocator 状态。

直接用浏览器打开 `dashboard.html`。窗口内可以调整显示的 K（最多使用预计算的候选时刻）、碎片率阈值和逻辑阶段；点击 Top-K 条目或时间线圆点即可切换 allocator 布局，不会生成任何中间图片。

原始 pickle 仍可直接拖入 <https://docs.pytorch.org/memory_viz> 查看地址布局。

## 4. 精确与近似结果

精确重放仅支持：

- `backend:native`
- `expandable_segments=False`
- 默认 512B rounding（未启用 `roundup_power2_divisions`）
- trace 从 allocator 空状态开始且没有 ring overflow

如果旧 snapshot 从中途开始或已经截断，工具会从最终 snapshot 反向撤销 retained trace 中的事件，生成近似时间线和 Top-10：

- 历史请求按 snapshot 配置取整；缺失配置使用 PyTorch native 默认值。
- `reserved`、segment 存在性和 allocation 存活关系由最终状态与地址事件确定。
- 历史 active block 使用取整后的下界尺寸；可能的 large-pool no-split 尾部作为不确定性。
- Dashboard 蓝线是 stranded/碎片率的保守上界，橙色虚线是估计下界。
- 反向模式不会输出近似 `failed-fit`；只有精确正向重放才产生该事件。

`replay.json` 的 `mode` 为 `exact_forward`、`reverse_approximate` 或 `final_snapshot_only`。旧 trace 仍使用函数栈给出粗粒度阶段，无法可靠恢复 microbatch 时会明确保留 unknown。

`device_memory_used - memory_reserved` 只能作为 PyTorch allocator 外部显存的粗略信号，不能解释 CUDA driver 的物理页碎片。
