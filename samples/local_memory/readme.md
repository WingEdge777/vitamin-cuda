# local memory test

## 0. 概念

网上关于 NVIDIA 存储层次架构的介绍文章数不胜数，但大多集中在 global memory、shared memory、constant memory、texture memory、L2/L1 cache 以及 registers 等。提及 local memory 的文章相对较少。前置基础暂且略过，今天我们直奔主题，聊聊个人对 local memory 的理解。

首先，local memory 的 local 是**逻辑上的 local**，指的是对每个线程来说，local memory 是其私有的，但其逻辑和物理上的地址（相当于“户口地址”）实际上是在 global memory 空间中的。local memory 变量的访问模式遵从 global memory 的规则，比如访问会经过 L1、L2 cache。此外，NV 保证了 local memory 是根据线程 ID 在 global memory 中以 32 bits 连续分布的。因此，warp 对于 local memory 的访问能够自动做到事务合并（Coalesced）。

但是要注意，global memory 访问的时延数量级是寄存器的几百倍，而 L1/L2 cache 的访问时延数量级也是寄存器的几十倍。因此，即使 local memory 的访问是合并的，甚至命中了 L1/L2 cache，其性能也远低于寄存器访问。

**如果你的目的是将变量保存在寄存器中，那么就要极力避免编译器将其分配到 local memory 中。**

### local memory 的场景

在 nv 《CUDA Programming Guide》 关于 local memory 有这样一段介绍：

> Local memory is thread local storage similar to registers and managed by NVCC, but the physical location of local memory is in the global memory space. The ‘local’ label refers to its logical scope, not its physical location. Local memory is used for thread local storage during the execution of a kernel. Automatic variables that the compiler is likely to place in local memory are:
>
> - Arrays for which it cannot determine that they are indexed with constant quantities,
> - Large structures or arrays that would consume too much register space,
> - Any variable if the kernel uses more registers than available, that is register spilling.
> Because the local memory space resides in device memory, local memory accesses have the same latency and bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in Coalesced Global Memory Access. Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address, such as the same index in an array variable or the same member in a structure variable.

官方直接说明的，有三种情况编译器会有可能将变量分配到 local memory 中：

- 无法被确定为通过常量索引的数组：
  - 这句话有点绕，其实意思是指，访问数组的下标不是编译期能确定的常量。例如：

```cpp

**global** void kernel() {
    float arr[4];
    int idx = threadIdx.x % 4;
    arr[idx] = 1.0f;        // ❌ 下标 idx 不是常量 → arr 在 local memory
}

```

- 占用过多寄存器空间的大型结构体或数组
- 变量超过寄存器限制大小，导致**寄存器溢出**

第 2 点看起来和第 3 点有相关性，但具体含义不同：第 2 点是说如果声明的结构体变量很复杂或数组很大，编译器会预判这个变量不太可能常驻物理寄存器，因而直接分配到 local memory 里；

第 3 点则完全是硬件限制，即使你的代码写得符合规范且优雅，但只要 Kernel 使用的寄存器数量超过了 GPU 的硬件限制，编译器就会把一些变量强行溢出到 local memory 中。

### 其他触发场景

除了该官方文档提到的三种情况，还有一些其他场景可能导致变量被分配到 local memory：

- 某些数学函数（如 sin(), cos()）的底层实现路径中可能隐式地使用了 local memory；
- 还有一个大坑点：对变量进行取地址（&）操作，会导致变量被强制分配到 local memory；
- 其他一些编译器认为无法进行有效优化并映射到物理寄存器的情况等等。

这里的第二点是个隐蔽的大坑。

## 1. 踩坑示例

当你想用向量化 load/store fp16/bf16 来提升算子的 global memory 访问性能时，你可能会习惯性的写下如下代码：

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

__device__ __forceinline__ void load_fp16x8(const half* input) {
  half2 pack[4];
  FLOAT4(pack[0]) = FLOAT4(input[idx]);  // ❌ 取地址 → 强制分配到 local memory
  //后续操作中 计算使用的是 local memory
}
```

**怎么确认有 local memory 使用？**

可以使用 nvcc 编译时加上 `-Xptxas -v` 参数查看编译输出。如果有如下提示：

```bash
xxx bytes stack frame / xxx bytes cumulative stack size # 栈上变量，也是 local memory

xxx bytes lmem # local memory 缩写

xxx bytes spill stores/loads # 寄存器溢出
```

恭喜你，你中招了~

### 正确姿势

- 方式1：直接使用原生向量类型
- 方式2：使用 union 强绑定（实质上还是使用原生向量类型）

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// 方式 1：直接使用原生向量类型
__device__ __forceinline__ void load_fp16x8(const half* input) {
  float4 pack;
  pack = FLOAT4(input[idx]);  // ✅ 直接加载到 float4 寄存器
  // 后续操作：将 pack.x, pack.y ... 拆分为 half2
}

// 方式 2： 使用 union 强绑定
union alignas(16) Pack128 {
    float4 f4;
    half2 h2[4];
};
__device__ __forceinline__ void load_fp16x8(const half* input) {
  Pack128 pack;
  pack.f4 = FLOAT4(input[idx]);  // ✅ 直接加载到 float4 寄存器
  // 后续操作：使用 pack.h2
}
```

那么问题来了：如果使用了非规范的指针强转写法，就一定会导致溢出到 Local Memory 吗？

其实不一定。现代编译器的静态分析极其强大，它完全有可能识破你简单的逻辑，帮你兜底，把变量强行捞回到物理寄存器中。但强烈建议不要依赖编译器优化，这块是玄学，别去赌别深究。

原因有二：一是编译器不一定能优化成功；二是哪怕变量侥幸留在了物理寄存器中，危机也并没有解除，反而可能引发极其严重的逻辑 Bug。

原因很简单：物理寄存器是不存在内存地址的。 如果变量留在了寄存器中，而你还在使用不同类型的指针（如 float4* 和 half2*）对它进行跨类型强转和访问，这属于未定义行为（UB）。编译器为了极致性能，可能会利用严格别名规则（Strict Aliasing）将这两条认为“不相干”的指令进行乱序重排，直接导致程序读取到寄存器的垃圾比特流，吐出 NaN。这种幽灵 Bug 的排查难度，远比性能下降要可怕得多。

## 2. 测试验证

我们用具体的代码来验证一下：

```cpp
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

__device__ __noinline__ void scale_by_ptr(float4 *ptr) {
    half2 *h2_ptr = reinterpret_cast<half2 *>(ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        h2_ptr[i] = __hadd2(h2_ptr[i], h2_ptr[i]);
    }
}

__device__ __noinline__ float4 scale_by_val(float4 val) {
    union { float4 f; half2 h[4]; } tmp;
    tmp.f = val;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp.h[i] = __hadd2(tmp.h[i], tmp.h[i]);
    }
    return tmp.f;
}

// 常规写法，溢出，但实际没溢出（编译器帮你兜底了）
__global__ void load_fp16x8_native_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    half2 pack[4];
    FLOAT4(pack[0]) = FLOAT4(input[idx]); // ❌ 对 pack 取地址 → 强制分配到 local memory

    FLOAT4(output[idx]) = FLOAT4(pack[0]); // ❌ 对 pack 取地址 → 强制分配到 local memory
}

// bad 示例：外部函数调用，指针逃逸，避免编译器优化回物理寄存器
__global__ void load_fp16x8_bad_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    half2 pack[4];
    FLOAT4(pack[0]) = FLOAT4(input[idx]);               // ❌ 对 pack 取地址 → 强制分配到 local memory
    scale_by_ptr(reinterpret_cast<float4 *>(&pack[0])); // ❌ 对 pack 取地址 → 强制分配到 local memory

    FLOAT4(output[idx]) = FLOAT4(pack[0]); // ❌ 对 pack 取地址 → 强制分配到 local memory
}
// good 示例
__global__ void load_fp16x8_good_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    float4 pack = FLOAT4(input[idx]); // ✅ 纯值拷贝，毫无指针痕迹
    pack = scale_by_val(pack);

    FLOAT4(output[idx]) = pack; // ✅ 纯值拷贝，毫无指针痕迹
}
```

编译输出分析：

```bash
ptxas info    : 28 bytes gmem
ptxas info    : Compiling entry function '_Z23load_fp16x8_good_kernelP6__halfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z23load_fp16x8_good_kernelP6__halfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers
ptxas info    : Compile time = 4.757 ms
ptxas info    : Function properties for _Z12scale_by_val6float4
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z22load_fp16x8_bad_kernelP6__halfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z22load_fp16x8_bad_kernelP6__halfS0_i
    16 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 16 bytes cumulative stack size
ptxas info    : Compile time = 1.667 ms
ptxas info    : Function properties for _Z12scale_by_ptrP6float4
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Compiling entry function '_Z25load_fp16x8_native_kernelP6__halfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z25load_fp16x8_native_kernelP6__halfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 14 registers, used 0 barriers
ptxas info    : Compile time = 0.767 ms
```

从编译日志中可以清晰地看到：

- good_kernel 和 native_kernel 的 stack frame 都是 0 bytes，说明变量留在/被优化回了寄存器。
- 而发生指针逃逸的 bad_kernel，出现了明确的 16 bytes stack frame，且使用了 16 bytes cumulative stack size，验证了强转取地址导致的 Local Memory 溢出

## 3. 警示

除了部分特殊情况，大部分人在大部分时候都不会在主观意图上使用 local memory。恰恰相反，我们更需要警惕的是**意外触发 local memory 分配**，尤其是以下三类高频“坑”：

- **使用非常量索引的数组**：如 `arr[idx]` 且 `idx` 非编译期常量，会导致整个数组被放入 local memory。
- **寄存器溢出**：kernel 使用的寄存器超过硬件限制，编译器会自动将部分变量 spill 到 local memory。
- **对变量取地址**：如 `&variable` 或通过指针操作，会强制变量溢出到 local memory；

再加上编译器的“神通广大”，当我们为了向量化，对变量进行取地址和指针强转（如 `float4*` 写，`half2*` 读）时，我们甚至可能在毫无察觉的情况下收获编译器准备的两个“惊喜”：

- 性能杀手（Local Memory 溢出）：如果 Kernel 逻辑复杂、发生指针逃逸，编译器无法追踪生命周期，就会将变量强制分配到 Local Memory，性能瞬间雪崩。
- 逻辑爆炸（静默的 NaN）：如果编译器强大的 SROA（Scalar Replacement of Aggregates，聚合体标量替换）优化帮你兜底，把变量留在了物理寄存器中。此时，致命的**严格别名规则（Strict Aliasing）**生效了。
  - 编译器内心戏：“刚才用 float4 *写数据，现在用 half* 读。合理假设：既然类型不同，它们必然不关联！”
  - 于是，编译器为了极致性能，大胆地将寄存器读取指令提前到了写入指令之前。你的代码读到了寄存器里还没被初始化的垃圾比特流，生成了 NaN，最终导致整个计算逻辑完全崩溃！

## 4. 总结

本文介绍了 CUDA 中 local memory 的概念及其产生的场景，并通过一段直观的代码验证了 local memory 溢出的确认方法以及底层潜藏的 UB 风险。

当然了，local memory 并非洪水猛兽，合理利用它（比如在无法放入寄存器的大数组场景）可以提升性能；又比如，为了保障充分利用寄存器同时不影响 Occupancy，我们会用 `__launch_bounds__` 来强制设定 block 数量，这本质上也是在主动拥抱寄存器溢出。

再比如 CUDA 13，甚至引入了显式的寄存器溢出到 Shared Memory 机制，参考：<https://developer.nvidia.com/blog/how-to-improve-CUDA-kernel-performance-with-shared-memory-register-spilling/>

如有错误，欢迎指正。完整 kernel 和测试代码可以从 github 获取，欢迎大家关注我的手撕算子系列 vitamin-cuda 项目：

以上

## 附录：测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```
