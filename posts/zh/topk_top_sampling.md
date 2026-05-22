# [CUDA 优化实战] topk_topp_sampling - 乱拳打死老师傅 ： 暴力插入排序 topK、block reduce array merge

> 本文适用于有一定 CUDA 编程基础阅读，but，即使无相关基础感兴趣的读取也可以阅读，哈哈~
>
> 完整 kernel 和测试代码可以点击 [flash_decode](/kernels/flash_decode) 查看
>
## 0. 序 - llm 推理的最后一公里 ： token sampling

有一段时间没更新文章了，今天来聊聊 token sampling。做 LLM 或用过 LLM 的从业者，大概都了解，LLM 实质就是预测下一个 token。LM 模型整个前向推理完后，最终的 lm_head 会输出一个向量 logits。而为了把这个 logits 变成 token 以及最终的文本（英文/中文单词、符号等），整个文本生成（Decoding）其实整了非常多的花活：

确定性解码（deterministic decoding）：主要有 greedy search（贪心搜索） 和 beam search（束搜索）。

- greedy search：直接取最大的 logits，毫无随机性。
- beam search：维护 n 条当前最优的候选序列，每一步扩展时只保留整体得分最高的 n 个序列。

随机性采样解码（sampling decoding）：发现上述确定性生成容易陷入死循环或胡言乱语，所以引入了基于概率分布的采样，并衍生出了各种在 Logits 阶段进行扰动/过滤 的参数

- temperature：温度，用来缩放 logits(logits/T)。温度越高分布越平滑，越低越陡峭。
- top_k：按概率大小降序排列，只从前 k 个 token 中选取
- top_p: 按概率大小降序排列，只从取概率和为 top_p 以内的 token 集合中选取
- min_p：只保留概率 大于 min_p*P(max) 的 token，使用的话一般会设定一个较小的值，例如：min_p=0.05，假设概率最大的 token 概率为 0.5，那么概率<0.05*0.5=0.025 的 token 就都过滤了
- penalty：各种惩罚项（重复、频率等），对对应 token 施加惩罚，降低 logits 数值或直接抹为-inf（概率为 0）

更别提还有结合语法树的 guided sampling（结构化输出）等等，不过这超出了单个 kernel 级别的讨论范围，本文暂不展开。

尽管如此，据个人观察，除了特定需要绝对确定性输出的结构化解析场景，确定性解码基本没人用了。而 sampling decoding 已经进入返璞归真的时期了，大家发现真正既有用、性价比又高的参数还是 topk，topp，最多加个 min_p。

这里有个前提，当代先进模型因为要支持多语言能力，词表大小已经达到了几十万的级别。这个前提下，一方面是因为大模型能力越来越强，本身输出的概率分布可能就是最佳分布；另一方面某些算法/参数由于占用显存大、计算量高，在模型输出质量足够的情况下，还继续使用就得不偿失了（如 beamsearch，各种惩罚等）。

举个例子，deepseek 官方建议部署时，采样参数为 temperature = 1.0, top_p = 1.0。

>For local deployment, we recommend setting the sampling parameters to temperature = 1.0, top_p = 1.0. For the Think Max reasoning mode, we recommend setting the context window to at least 384K tokens.

不亏是国模之光，就是自信！这两个配置的意思是不扰动/不过滤任何 token 的概率，相信模型本身输出的 token 概率分布就是合理的。

这在算法效果上完全没问题，不过从工程方面考虑，不建议在无 topk 的情况下设置 top_p=1.0。因为 top_p=1，又没有 top_k，这样就需要计算每一个 token 的概率，开销比较大（词表大），相当于做一个巨大的 softmax。当然，众所周知 deepseek 的 infra 团队非常之强，其 sampling kernel 可能有一些特殊的优化技巧，可以避免这种开销，那就不是我能了解到的了。

本文聚焦于随机采样算法 `topk_topp_sampling`，实际上最初是想学习一下 flashinfer 的 `top_k_top_p_sampling_from_logits`，但是学不进去。而等写完自己的初版暴力 kernel 实现后，惊讶的发现暴力插入排序后 top_p 过滤采样速度并不慢。

因此才有了这篇文章，本文将会给出两个 kernel 实现：

- sampling_topk_topp_batched （单线程暴力 topk 插入排序，block reduce 合并排序数组）
- sampling_topk_topp_split_k （同上，但是加入了 split-k 实现，用于加速小 batch size 情况下的 sampling）

## 1. pytorch native topk_topp_sampling

先给一个 pytorch 的 native 实现，取自 vllm 源码：

### pytorch native

```python
# vllm/vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)

def torch_topk_topp_sampling(logits, top_k, top_p, seed=42):
    k = torch.full((logits.shape[0],), top_k, dtype=torch.long, device=logits.device)
    p = torch.full((logits.shape[0],), top_p, dtype=torch.float32, device=logits.device)
    masked_logits = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    probs = torch.softmax(masked_logits, dim=-1)
    generators = {0: torch.Generator(device="cuda")}
    generators[0].manual_seed(seed)
    return random_sample(probs, generators)
```

可以看到，所谓的 topk 就是把 topk 之外的 token 概率抹为 0，top_p 就是把累计和超出 top_p 之后的 token 都抹为 0

在这样一个朴素的 pytorch 实现下，需要反复多次的读取 logits tensor，做全量 softmax，做过滤，然后采样。显然，显存读取和计算开销很大。因此工业界的采样算子一般都是一个 fused 的 kernel，避免多次读取 gmem.

实际上 vllm/sglang 等为了兼容温度、惩罚等参数，还会进行更多的 logits 操作造成不必要的开销。这就是通用库的无奈，如果是个人或者团队内部使用，很多东西都是可以砍掉的。

## 2. top_k_top_p_sampling

### flashinfer.sampling.top_k_top_p_sampling_from_logits

flashinfer 的开发者们（陈天奇等大佬们）非常厉害，他们的 fused kernel 是无显式排序采样算法，核心是拒绝采样，主要原理有点类似于二分/三分的思想，通过 pivot 阈值过滤，经过少量几趟的 pass 就可以把不符合条件的的 token 都筛掉，所以叫拒绝式采样。并且还通过理论证明该算法是正确的。

感兴趣的同学，具体可以参考 flashinfer 官方博客：<https://flashinfer.ai/2025/03/10/sampling.html> 进行学习。

![](https://flashinfer.ai/assets/imgs/sampling_blog/Rejection_Sampling.gif)

![](https://flashinfer.ai/assets/imgs/sampling_blog/dual-pivot-sampling.png)

我就是感兴趣的同学，看完文档和博客觉得很厉害，但等去翻看代码准备跟着大佬学习实现时，emmm 还是有点难绷。FlashInfer 作为一个通用库，其代码实现中有较多模板编程和宏函数的写法，难读，我就不想看了。

于是想着自己先搓个 naive 的算子对比一下

### topk + topp 的筛选和过滤

显然从 kernel 名字就能看出，算法核心是 topk 和 topp 的筛选和过滤。既然决定了自己实现，那么先初步理一理思绪，观察 pytorch 实现，我们可以看出，topk 和 topp 的过程都是把不满足要求的 token 概率抹为 0. 那么实际上我们可以把算法流程简化为

- 先做 topk，筛出 topk 个 token id 后，排序
- 然后直接在这 topk 个 token 计算 softmax 得到概率
- 计算概率前缀和，进行 topp 过滤
- 按概率采样出 token id

为什么可以这么做，仔细看上面 vllm 代码，其实质上就是把其他 token 概率抹为 0，那么就相当于其他 token 不存在了，因此 softmax 无需考虑其他 token 的。只要这 topk 个 token 概率和为 1 即可。

更妙的是，现代采样器 topk 一般都很小，大概 20~40 级别，所以完全可以考虑放在单线程的寄存器内维护一个 topk 最大堆。等扫完所有数据后，再把各个线程的数据合并求 topk 即可。

这样对 logits 只需要扫描一遍，访存压力大大减少。topk 不大，排序以及 softmax 的计算开销也不大，完美~

## 3. sampling_topk_topp_batched 实现

按照上面思路，我有了这个暴力插入排序的 `sampling_topk_topp_batched` 实现思路，

- 一个 block 负责一行 logits，block_size=256（没调，直接用的）
- block 内每个线程各自维护 topk 个 token，扫完 logits 后，block reduce merge 得到最终 topk 个 token
  - 线程内如何维护 topk 数组呢，使用暴力插入排序（都说了是 naive 实现嘛）

### 插入排序

首先，给出 单个线程在 device 上的插入排序函数，维护 score/token_id 寄存器数组，始终保持其降序性。

```cpp
// stable insertion sort
template <const int TOP_K>
__device__ __forceinline__ bool insert_sorted(float (&score)[TOP_K], int (&token_id)[TOP_K], float new_val, int new_id) {
    if (new_val <= score[TOP_K - 1])
        return false;

#pragma unroll
    for (int i = TOP_K - 1; i > 0; i--) {
        bool bigger_than_curr = (new_val > score[i]);
        bool bigger_than_prev = (new_val > score[i - 1]);

        score[i] = bigger_than_curr ? (bigger_than_prev ? score[i - 1] : new_val) : score[i];
        token_id[i] = bigger_than_curr ? (bigger_than_prev ? token_id[i - 1] : new_id) : token_id[i];
    }

    bool bigger_than_0 = (new_val > score[0]);
    score[0] = bigger_than_0 ? new_val : score[0];
    token_id[0] = bigger_than_0 ? new_id : token_id[0];
    return true;
}
```

坦白讲，这个插入排序函数也是花了我一些时间才写出来的。朴素的插排是找到 position，然后赋值+break，但这种写法在 gpu 上是行不通的。为什么？

如果你写下`score[insert_pos] = new_val；`这行代码，由于 insert_pos 是非编译期常量，会导致整个数组会被分配到 local memory，性能暴跌，关于 local memory 见我的 [local memory 文章]()
因此，为了规避这个问题，我们要进行完整的 TOP-1 次循环，只在赋什么值上做两次判断。

- 如果 new_val 大于 当前位置 i 上值，同时还大于 i-1 上的值，那么说明他的位置是在 i 之前，那么就把 i-1 的值放到当前位置上
- 如果 new_val 大于 当前位置 i 上值，但是小于 i-1 上的值，那么说明当前就是要插入的位置，因此直接 new_val 直接赋值给位置 i
- 最后一种情况就是 new_val 比当前位置 i 上值还小，那说明该值已经被插入到某个位置上了，i 位置是的值保持不变即可

也许有人会立马反驳，你这么写每次都要循环 TOP-1 次，这么多判断和冗余赋值操作，那性能能好吗？

多了判断和冗余赋值，我是承认的，但是这个性能考量要从三个方面考虑：

- 其实大量的数据进不了循环，因为我们维护的是 topk，数值较小的数据会在第一行 `if (new_val <= score[TOP_K - 1])`就被 pass 了
- 经过循环 unroll 展开后的指令，现代 gpu 的指令级并行能力非常好，所以性能没想像的那么糟，当然也有 topk 值比较小的缘故
- 相比于数组被踢到 local memory（实际上是 gmem），这点冗余操作的开销简直洒洒水，绝对是值得的

### merge array

好，上面我们说明了一个线程是如何维护 topk 寄存器数组的。那等遍历完 logits。如何合并呢。显然，直白且稍微高效一点的做法是先在 warp 内进行合并，比如都 merge 到 lane 0 上。然后 block 间通过共享内存再合并一次

#### warp reduce merge

这个其实很简单，利用`__shfl_down_sync` + 复用之前的插排代码就可以完成折半规约 merge 了，当然了看起来依然是很暴力：

```cpp
// step 2: warp reduce merge local array
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {

#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], delta);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], delta);

            if (lane_id < delta) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }
```

#### block merge

我们直接开辟 num_warps x top_k 大小的 smem，每个 warp lane 0 写入 smem。随后由 thread 0 进行最后的合并与收尾：

```cpp
// step 3: final reduce and sampling
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];

    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // curand
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        float u = curand_uniform(&state); // create a float from (0, 1]
        float random_val = 1.f - u;

        // block reduce
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }
        ...
    }
```

在以上过程中，最后的收尾工作也是 thread 0 做了，是不是很暴力。
但其实没那么暴力，我们已经强调了好几次了，topk 很小，我甚至于直接把 topk 做成了模板参数。BLOCK_SIZE = 256 时块内只有 8 个 warp，由线程 0 合并 7 个长度 topk 的数组，其实完全是在寄存器和 L1 Cache 级进行操作，没有任何线程间同步开销，并没有想象中那么慢。
当然，引入双调排序，或用 cub 进行排序也是可以的，但会引入高密集的块内同步和 smem 读写，收益未知，且增加复杂度和依赖。

### softmax + curand + top_p

这一部分就是对 topk 个数据进行 softmax + topp 过滤，最后采样了。依然是 thread 0 承担了所有（泪目）

```cpp
        // softmax
        float max_val = score[0];
        float sum_prob = 0.0f;
        float probs[TOP_K];
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            probs[i] = expf(score[i] - max_val);
            sum_prob += probs[i];
        }

        // top_p
        float cumsum = 0.0f;
        float trunc_sum = 0.0f;
        int last_idx = TOP_K - 1;

        for (int i = 0; i < TOP_K; i++) {
            float p = probs[i] / sum_prob;
            cumsum += p;
            if (cumsum >= top_p) {
                last_idx = i;
                trunc_sum = cumsum;
                break;
            }
        }
        if (trunc_sum == 0.0f)
            trunc_sum = cumsum;

        // sampling
        float r = random_val * trunc_sum;
        float cdf = 0.0f;
        int final_id = token_id[last_idx];

        for (int i = 0; i <= last_idx; i++) {
            cdf += probs[i] / sum_prob;
            if (cdf >= r) {
                final_id = token_id[i];
                break;
            }
        }

        // final token id
        output_ids[blockIdx.x] = final_id;
```

ok，完整代码就不贴了。请移步 github 查看：<https://github.com/WingEdge777/vitamin-cuda/tree/main/kernels/sampling>

## 4. sampling_topk_topp_split_k 实现

split-k，老生常谈了。我们在之前 softmax 文章和 flash decoding 文章中，都提到了 split-k。
在这个场景下的具体逻辑就是词表太大了，每次等一个 block 扫完 logits，太慢，batchsize 小的话，sm 利用率还低。因此要想办法切 chunk，然后分发给不同的 block 处理。

- pass_1_kernel：这里每个 block 处理的逻辑和上一个 kernel 的 topk 过程一模一样，处理完每个 block 把 topk 结果写入临时空间。
- pass_2_kernel：然后再用一个 block，读取临时空间中的数据，进行最后的 merge+softmax+topp+sampling。

直接上代码：

```cpp
// split-k pass 1: partial topk per split
template <const int TOP_K = 20, const int CHUNK_SIZE = 2048, const int BLOCK_SIZE = 256, typename T>
__global__ void sampling_topk_topp_split_k_pass1_kernel(T *logits, int vocab_size, float *ws_score, int *ws_id) {
    const int split_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int num_splits = gridDim.x;

    const int vocab_per_split = ((vocab_size + num_splits * 8 - 1) / (num_splits * 8)) * 8;
    const int vocab_start = split_id * vocab_per_split;
    const int vocab_end = min(vocab_start + vocab_per_split, vocab_size);

    T *row_ptr = logits + batch_id * vocab_size;
    int tid = threadIdx.x * 8;
    // step 1: maintain sorted local score/token_id array
    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }
    for (int idx = vocab_start + tid; idx < vocab_end; idx += CHUNK_SIZE) {
        pack128 tmp;
        tmp.f4 = FLOAT4(row_ptr[idx]);
        for (int x = 0; x < 8; x++) {
            float val = static_cast<float>(tmp.h[x]);
            int v_idx = idx + x;
            if (val > score[TOP_K - 1]) {
                insert_sorted<TOP_K>(score, token_id, val, v_idx);
            }
        }
    }
    // step 2: warp reduce merge local array
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], delta);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], delta);
            if (lane_id < delta) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }
    // step 3: block reduce merge and write to gmem
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // block reduce
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }
        int ws_offset = (batch_id * num_splits + split_id) * TOP_K;
        for (int i = 0; i < TOP_K; i++) {
            ws_score[ws_offset + i] = score[i];
            ws_id[ws_offset + i] = token_id[i];
        }
    }
}

// split-k pass 2: merge partial topk and sampling
template <const int TOP_K = 20, const int BLOCK_SIZE = 256>
__global__ void sampling_topk_topp_split_k_pass2_kernel(
    int *output_ids, float top_p, int64_t seed, int64_t offset, int num_splits, float *ws_score, int *ws_id) {
    const int batch_id = blockIdx.x;
    const int total = num_splits * TOP_K;
    const int ws_base = batch_id * total;

    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }

    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        insert_sorted<TOP_K>(score, token_id, ws_score[ws_base + idx], ws_id[ws_base + idx]);
    }

    // warp reduce merge
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], delta);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], delta);
            if (lane_id < delta) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }

    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {

        // block reduce
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }

        // softmax
        float max_val = score[0];
        float sum_prob = 0.0f;
        float probs[TOP_K];
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            probs[i] = expf(score[i] - max_val);
            sum_prob += probs[i];
        }

        // top_p
        float cumsum = 0.0f;
        float trunc_sum = 0.0f;
        int last_idx = TOP_K - 1;
        for (int i = 0; i < TOP_K; i++) {
            float p = probs[i] / sum_prob;
            cumsum += p;
            if (cumsum >= top_p) {
                last_idx = i;
                trunc_sum = cumsum;
                break;
            }
        }
        if (trunc_sum == 0.0f)
            trunc_sum = 1.0f;
        
        // curand
        curandStatePhilox4_32_10_t state;
        curand_init(seed, batch_id, offset, &state);
        float u = curand_uniform(&state);
        float random_val = 1.f - u;

        // sampling
        float r = random_val * trunc_sum;
        float cdf = 0.0f;
        int final_id = token_id[last_idx];
        for (int i = 0; i <= last_idx; i++) {
            cdf += probs[i] / sum_prob;
            if (cdf >= r) {
                final_id = token_id[i];
                break;
            }
        }

        // final token id
        output_ids[batch_id] = final_id;
    }
}
```

## 5. benchmark

不多说，直接上 benchmark 结果，和 vllm pytorch native 以及 flashinfer 进行对比，测试环境是 RTX 5060 移动版显卡：

```yaml
####################################################################################################
bs: 1, vocab_size: 256000
torch                                    mean time: 0.519055 ms, 0.99 GB/s
flashinfer                               mean time: 0.140417 ms, speedup: 3.70, 3.65 GB/s
sampling_topk_topp_batched               mean time: 0.084192 ms, speedup: 6.17, 6.08 GB/s
sampling_topk_topp_split_k               mean time: 0.036327 ms, speedup: 14.29, 14.09 GB/s
####################################################################################################
bs: 4, vocab_size: 256000
torch                                    mean time: 0.931264 ms, 2.20 GB/s
flashinfer                               mean time: 0.154697 ms, speedup: 6.02, 13.24 GB/s
sampling_topk_topp_batched               mean time: 0.084510 ms, speedup: 11.02, 24.23 GB/s
sampling_topk_topp_split_k               mean time: 0.051476 ms, speedup: 18.09, 39.79 GB/s
####################################################################################################
bs: 8, vocab_size: 256000
torch                                    mean time: 1.296351 ms, 3.16 GB/s
flashinfer                               mean time: 0.182281 ms, speedup: 7.11, 22.47 GB/s
sampling_topk_topp_batched               mean time: 0.084636 ms, speedup: 15.32, 48.40 GB/s
sampling_topk_topp_split_k               mean time: 0.075195 ms, speedup: 17.24, 54.47 GB/s
####################################################################################################
bs: 16, vocab_size: 256000
torch                                    mean time: 2.317547 ms, 3.53 GB/s
flashinfer                               mean time: 0.307686 ms, speedup: 7.53, 26.62 GB/s
sampling_topk_topp_batched               mean time: 0.102009 ms, speedup: 22.72, 80.31 GB/s
####################################################################################################
bs: 32, vocab_size: 256000
torch                                    mean time: 4.632301 ms, 3.54 GB/s
flashinfer                               mean time: 0.598686 ms, speedup: 7.74, 27.37 GB/s
sampling_topk_topp_batched               mean time: 0.127663 ms, speedup: 36.29, 128.34 GB/s
```

测试是随机生成 logits，然后在随机 5 个 spike 值（模拟真实世界少量 token 占大头）这里只保留了 vocab_size 为 256000 的结果。

从结果看

- `sampling_topk_topp_batched` 在任意 batchsize 下都能稳定超过 flashinfer
- split-k 算子在 低 batch size 下能提供更快的速度，符合预期
  - 但这个比较其实是不太合理的，因为 flashinfer 的拒绝式采样主打的就是真实概率分布场景下，少量 pass 就可完成采样，因此我还取了 qwen3-8b 的几十个 logits 进行 [real qwen3-8b logits test](https://github.com/WingEdge777/vitamin-cuda/blob/main/kernels/sampling/readme.md#real-qwen3-8b-logits-data)
  - 测试结果依然是 `sampling_topk_topp_batched` 胜出
- 原因我也不知道为什么，但我推测
  - 我们因为消费级显卡 sm120 架构 实在不受待见，以至于各种库都没有针对我们的卡做优化
  - 我的算子是把 topk 作为模板参数的（编译器常量），只支持较小的 topk（大了寄存器就爆了），所以编译器能进行极致的无分支展开提高指令级并行，而 flashinfer 是通用库，支持任意 topk，所以不得不在性能上做出妥协（可能有动态分支或复杂的规约状态机）
    - 通用库的无奈（again）
  - 在小 Batch Size 的 Decode 阶段，单行 Logits 的计算根本无法让硬件饱和。
    - split-k 方案通过空间换时间，人为增加了 Grid 密度，恰好填满了 SM 调度器。

## 6. 结束

不管怎么说，看到这里也不容易，感谢大家的观看。完整代码和测试脚本还请从 github 获取：<https://github.com/WingEdge777/vitamin-cuda/tree/main/kernels/sampling>

如有错误，欢迎指正。

以上
