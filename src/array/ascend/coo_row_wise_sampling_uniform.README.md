# CooRowWiseSamplingUniform（NPU 版）

对 COO 格式图按源节点做等概率随机邻边采样的昇腾实现。

## 产品支持情况

| 产品 | 是否支持 | 说明 |
|------|:---:|------|
| Atlas A2 训练系列（Ascend910B1~B4） | ✅ 已验证 | 910B3 真机全量回归 |
| Atlas A3 训练/推理系列（Ascend910_93） | ✅ 代码自适应 | 核数/UB 运行时查询，无硬编码 |
| Ascend 950PR/950DT | ✅ 代码自适应 | UB 248KB 自动适配 |
| vNPU 裁剪实例 | ✅ 代码自适应 | 核数查询返回实际可用值 |

## 功能说明

- **算子功能**：对 COO 格式图，按 `rows` 中每个源节点从其出边中等概率随机采样
  `fanout` 条边，返回子图 COOMatrix（row=真实节点 ID，col=邻居，data=原边 ID 映射）。
  是 `dgl.sampling.sample_neighbors(prob=None, edge_dir='out')` 在 COO 格式图上的 NPU 路径。
- **实现路线**：`COOToCSR`（全图）→ 多核 CSR 采样 kernel → rows 直通（免行号回映射），
  与 CUDA 上游结构一致（上游同样先转 CSR 再采样）。
- **语义基线**：与 CPU 版逐语义对齐；随机场景不可逐元素对拍（RNG 不同源），
  确定性场景（select-all / fanout≥度）与 CPU 逐边相等。

### 多核 kernel 设计

```
host:  行度数 D2H 一次 → 每行采样数 → nnz 均衡行分区 + 每块输出前缀和
       → 输出精确分配 + 三表流有序上传
kernel: 全部 vector 核（AIV_ONLY），每核一段行区间
       每行: CSR 窗口 DataCopyPad 批量入 UB（VECIN 双缓冲）→ 标量采样
              → VECCALC 暂存 → VECOUT 队列整周期搬出
       度数超过 UB 窗口的行自动回退直读 GM（倾斜图内存有界）
```

### 硬件泛化

| 参数 | 获取方式 |
|------|---------|
| vector 核数 | `aclrtGetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM)` 运行时查询 |
| 每核 UB | `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE` 查询，经 tiling 下发 kernel |
| dtype | C++ 模板（int32 / int64） |

kernel 侧不持有任何 SoC 相关常量；host 查询失败时回退 910B 族默认值。

## 参数说明

| 参数 | 方向 | 类型 | 说明 |
|------|------|------|------|
| `mat` | 输入 | COOMatrix | row/col 同 dtype（int32/int64）；允许未按行排序 |
| `rows` | 输入 | IdArray | 种子节点；须与图同 dtype（否则 CHECK 报错） |
| `num_samples` | 输入 | int64 | ≥0 或 -1（select-all，全选并强制无放回） |
| `replace` | 输入 | bool | true=有放回（可重复边） |
| 返回 | 输出 | COOMatrix | 精确长度分配，无 over-allocate |

## 约束说明

- weighted 路径（prob/mask 非 null）在 Ascend 上尚不支持，显式 CHECK 报错（不静默降级）。
- 越界种子 id：kernel 侧按空行静默丢弃（与 CPU 路径行为一致），属防御纵深。
- `num_rows × fanout` 溢出时响亮报错，不回绕欠分配。
- 未排序 COO 输入正确（内部排序），大图性能见下节瓶颈说明。

## 调用说明

```python
import dgl
g = dgl.graph((src, dst), num_nodes=n).formats("coo").to("npu")
sg = dgl.sampling.sample_neighbors(g, nodes, fanout=10, edge_dir="out")
```

测试：`pytest tests/ascend/test_coo_row_wise_sampling_npu.py`

## 场景覆盖

测试维度：dtype（int32/int64）× replace（有/无放回）× select-all × fanout 边界
（0 / 超度 / 度 1）× 空输入（空 rows / 空图）× 输入有序性（排序 / 未排序）×
采样方向（out 直通 / in 转置）× 边 ID 映射 × 非法输入防御（越界 id、dtype 不匹配）×
统计均匀性 × 压力规模 × NPU 路径守卫。共 21 个测试项。

## 性能报告

**环境**：Ascend910B3 / CANN 9.1.0 / torch 2.12 + torch_npu；中位数 5 次，含同步。

### 端到端（COO 图，含格式转换）

| 场景 | NPU | CPU | CPU/NPU |
|------|----:|----:|--------:|
| 1k 节点 / 1万边 / fanout 10 | 11.2 ms | 13.4 ms | 1.19x |
| 10万节点 / 100万边 / fanout 10 | 1239 ms | 521 ms | 0.42x |
| 同上，边乱序 | 1227 ms | 531 ms | 0.43x |
| 同上，select-all | 1225 ms | 542 ms | 0.44x |
| 同上，fanout 50 | 1258 ms | 543 ms | 0.43x |

### 分阶段拆解（10万行图）

| 阶段 | 单核旧版 | 多核新版 | 加速 |
|------|--------:|---------:|-----:|
| 采样 kernel 本体 | ~1200 ms | 18 ms | ~65x |
| COOToCSR 格式转换（上游共享算子） | ~1190 ms | ~1190 ms | 未改动 |

### 说明

1. 小图已反超 CPU；大图端到端瓶颈在上游 `COOToCSR`（占 93%+，其内部 CPU 排序与
   多流同步主导），与本采样 kernel 无关，将单独优化。
2. 多核版启用全部 vector 核 + UB 双缓冲队列流水（旧版单核逐元素 GM 访问）。
3. 对比基线说明：业界无公开的同类 NPU COO 采样实现可对标；CUDA 上游无独立 COO
   采样 kernel（内部转 CSR 复用）。故对比基线为 DGL CPU 路径与本项目旧版本。

## 已知限制

| 项 | 说明 |
|----|------|
| RNG 质量 | xorshift32（乘法去偏），统计检验通过；弱于 CUDA 的 Philox，高要求场景待升级 |
| COOToCSR 端到端瓶颈 | 见性能报告；多核 bincount 替代 CPU 排序可再提速 |
| weighted COO 采样 | 尚不支持（后续任务） |
| A3 / 950 真机验证 | 代码已自适应，待验证环境 |
