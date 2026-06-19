# Ascend NPU RandomWalk 适配说明

本文档记录将 DGL `random_walk` 适配到 Ascend NPU 的改动内容和操作步骤。当前路线使用 `torch_npu` 创建 NPU Tensor，并通过 DGL 的 DLPack 入口进入自定义 `RandomWalkCustom` ACLNN 算子。

## 当前数据流

1. Python 侧通过 `torch_npu` 创建 NPU seeds，例如 `seeds.to("npu:0")`。
2. DGL 通过 DLPack 接收 PyTorch NPU Tensor。
3. DGL 根据 `kDGLAscend` 调度到 `randomwalk_npu.cc`。
4. NPU 实现从 CPU 图读取 CSR，在 host 侧生成随机数，然后调用 `aclnnRandomWalkCustom`。
5. DGL 在 NPU 上创建 `traces` / `eids` 输出，再交还给 PyTorch。

当前图结构仍保留在 CPU，只有 `seeds`、`traces`、`eids` 使用 NPU context。

## 修改文件清单

### 构建系统

`CMakeLists.txt`

- 新增 `USE_ASCEND` 和 `ASCEND_PATH`。
- 编译时定义 `DGL_USE_ASCEND`。
- 添加 Ascend CANN include、custom op API include。
- 链接 Ascend 相关库：
  - `ascendcl`
  - `cust_opapi`
  - `acl_op_compiler`
  - `nnopbase`
- `USE_ASCEND=ON` 时加入：
  - `src/graph/sampling/randomwalks/randomwalk_npu.cc`
  - `src/runtime/ascend/ascend_device_api.cc`

### Runtime / Context

`include/dgl/runtime/c_runtime_api.h`

- 使用同步后分支已有的设备类型 `kDGLAscend = 12`。

`src/runtime/c_runtime_api.cc`

- 将 `kDGLAscend` 转换为字符串 `"npu"`。

`include/dgl/runtime/ndarray.h`

- 支持 NPU context 的字符串显示。

`src/runtime/dlpack_convert.cc`

- 将 PyTorch `torch_npu` DLPack 设备类型 `kDLExtDev` / PrivateUse1 映射为 DGL `kDGLAscend`。
- 将 DGL NPU Tensor 导出回 DLPack 时映射为 `kDLExtDev`。

`src/runtime/ascend/ascend_device_api.cc`

- 使用现有 `device_api.ascend`。
- 使用 ACL runtime 实现 NPU 内存分配、释放、拷贝、stream 创建和同步。
- 不再调用 `aclInit`，ACL 初始化交给 `torch_npu`。
- 即使使用 `torch_npu`，该文件仍然需要保留，因为 DGL 需要在 NPU 上创建输出 NDArray。

### ATen Dispatch

`include/dgl/aten/macro.h`

- 新增 `ATEN_XPU_SWITCH_CPU_NPU`。
- 只在指定算子中支持 CPU / NPU dispatch，避免把通用 CUDA 模板全部实例化为 NPU。

### RandomWalk Dispatcher

`src/graph/sampling/randomwalks/randomwalks.cc`

- 允许 CPU graph + NPU seeds 的组合。
- `RandomWalk` 调度改为使用 `ATEN_XPU_SWITCH_CPU_NPU`。
- 当前真正实现的是 uniform `RandomWalk`；restart 相关接口在 NPU 上仍未实现。

### NPU RandomWalk 实现

`src/graph/sampling/randomwalks/randomwalk_npu.cc`

- 新增 Ascend NPU random walk 实现。
- 从 DGL graph 中读取 CSR。
- 当前 custom kernel 只接收一份 CSR，因此要求 metapath 各步 CSR 内容一致。
- 将 NPU seeds 拷贝到 host。
- host 侧生成随机数，布局为 seed-major：

```text
random[seed * num_steps + step]
```

- 将 row ptr、col ind、seeds、metapath、random、output 转成 `int32` 传给 custom op。
- 调用：
  - `aclnnRandomWalkCustomGetWorkspaceSize`
  - `aclnnRandomWalkCustom`
- 将 custom op 输出写回 DGL NPU NDArray。
- 增加 `aclGetRecentErrMsg()`，方便输出 ACL / ACLNN 详细错误。

### Python 后端支持

`python/dgl/_ffi/runtime_ctypes.py`

- 增加 `"npu"` 与设备类型 `3` 的映射。

`python/dgl/backend/pytorch/tensor.py`

- 支持 `torch.device("npu")`。
- `device_id()` 支持 `torch.npu.current_device()`。
- `to_backend_ctx()` 可将 DGL NPU context 转为 PyTorch NPU device。
- `copy_to()` 支持 `input.to("npu")`。

`python/dgl/utils/checks.py`

- 放宽检查，允许 CPU graph + NPU tensor 的 Ascend random walk 路径。

`python/dgl/ndarray.py`

- 已移除临时的 `dgl.nd.npu()` 入口。
- 当前路线要求通过 `torch_npu` 创建 NPU Tensor，再传入 DGL。

### 验证脚本

`/home/xty/Ascend_self_work/randomwalkCall/scripts/casual.py`

- Cora 图 random walk 验证脚本。
- CPU 路径使用 PyTorch CPU seeds。
- NPU 路径使用 `torch_npu`：
  - `torch.device("npu:0")`
  - `seeds.to("npu:0")`
  - `dgl.sampling.random_walk(...)`

`/home/xty/Ascend_self_work/randomwalkCall/scripts/diagnose_torch_npu.py`

- 诊断 `torch_npu` 环境。
- 输出 Python 路径、`LD_PRELOAD`、`torch` 版本、`torch_npu` 版本、NPU 可用性。
- 用于定位 `torch` / `torch_npu` / CANN 版本不匹配问题。

## 环境准备

当前验证通过的版本：

```text
torch 2.8.0+cpu
torch_npu 2.8.0
```

进入环境：

```bash
conda activate dgl_env
unset LD_PRELOAD
source /home/xty/miniconda3/envs/dgl_env/Ascend/cann-9.0.0-beta.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
```

注意：不要让 `LD_PRELOAD` 指向：

```text
/home/xty/Ascend/cann-9.0.0-beta.1/lib64/libhccl.so
```

该文件不存在，会导致 `torch_npu` 加载 HCCL 时出错。

## 部署自定义算子

`torch_npu` 使用的是 conda 环境里的 CANN，因此 `RandomWalkCustom` 也必须部署到同一套 CANN 目录下。

复制 custom OPP：

```bash
cp -a \
  /home/xty/Ascend/cann-9.0.0-beta.1/opp/vendors/customize \
  /home/xty/miniconda3/envs/dgl_env/Ascend/cann-9.0.0-beta.1/opp/vendors/
```

启用 `customize` vendor：

```bash
printf 'load_priority=customize\n' \
  > /home/xty/miniconda3/envs/dgl_env/Ascend/cann-9.0.0-beta.1/opp/vendors/config.ini
```

关键文件：

```text
opp/vendors/customize/op_api/include/aclnn_random_walk_custom.h
opp/vendors/customize/op_api/lib/libcust_opapi.so
opp/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json
opp/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/random_walk_custom.json
```

如果 `config.ini` 缺失，常见错误为：

```text
binary_info_config.json of socVersion [ascend910b] does not support opType [RandomWalkCustom]
```

## 编译 DGL

```bash
cd /home/xty/dgl-ascend
mkdir -p build
cd build

cmake .. \
  -DUSE_ASCEND=ON \
  -DASCEND_PATH=/home/xty/miniconda3/envs/dgl_env/Ascend/cann-9.0.0-beta.1

make -j4 dgl
```

如果已有 build 目录曾经配置为其他 `ASCEND_PATH`，请重新执行 CMake，或删除 build 后重新生成。

## 运行验证

```bash
unset LD_PRELOAD
source /home/xty/miniconda3/envs/dgl_env/Ascend/cann-9.0.0-beta.1/set_env.sh

export LD_LIBRARY_PATH=\
/home/xty/miniconda3/envs/dgl_env/Ascend/cann-9.0.0-beta.1/opp/vendors/customize/op_api/lib:\
/home/xty/dgl-ascend/build:\
$LD_LIBRARY_PATH

export PYTHONPATH=/home/xty/dgl-ascend/python:/home/xty/dgl-ascend/build:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0

/home/xty/miniconda3/envs/dgl_env/bin/python \
  /home/xty/Ascend_self_work/randomwalkCall/scripts/casual.py
```

如果输出中出现：

```text
torch_npu available: True
=== random_walk (NPU via torch_npu) ===
=== summary (torch_npu) ===
```

说明以下链路已经走通：

- `torch_npu` 可用。
- PyTorch 可以创建 NPU tensor。
- DGL 可以通过 DLPack 接收 NPU tensor。
- DGL dispatcher 可以进入 `randomwalk_npu.cc`。
- CANN 可以找到 `RandomWalkCustom`。
- `RandomWalkCustom` 可以完成 Cora smoke test。

## 排障记录

如果后续重新部署 custom op 后再次出现：

```text
UB address accessed by the VEC instruction is not aligned
```

这通常不是 `torch_npu` 导入失败，也不是 DGL dispatch 失败，而是 `RandomWalkCustom` kernel 内部问题。

之前定位到的主要风险点在：

```text
/home/xty/Ascend_self_work/randomwalk_gen/op_kernel/random_walk_custom.cpp
```

原因是 gather 阶段存在原地读写：

- 从 `walksLocal` gather。
- 又写回 `walksLocal`。
- 部分 source index 已经被前面的写入覆盖。

建议修复：

- 使用第二个 UB buffer 作为 gather 输出，避免 in-place gather。
- 或修改 layout，使 gather 的读区和写区不重叠。

修完 custom kernel 后，需要重新编译并部署 custom OPP，然后重新运行验证脚本。

