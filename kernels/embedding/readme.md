# embedding

## 说明

embedding kernel

- [x] embedding fp32/fp16 版
- [x] embedding_fp32x4(fp32向量化)
- [x] embedding_fp32x4(fp32向量化, packed r/w)
- [x] embedding_fp16x2(fp16向量化)
- [x] embedding_fp16x8(fp16向量化)
- [x] embedding_fp16x8(fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

因embedding表过大，需要在显存较大的卡上测试

```bash
export TORCH_CUDA_ARCH_LIST=8.9
python test.py
```

### 输出

```bash
ninja: no work to do.
####################################################################################################
emd_size: 32, emd_dim: 8, seq_len: 1
torch                          mean time: 0.005559 ms
embedding                      mean time: 0.002156 ms
embedding_fp32x4               mean time: 0.002162 ms
embedding_fp32x4_packed        mean time: 0.002158 ms
torch                          mean time: 0.005546 ms
embedding_half                 mean time: 0.002123 ms
embedding_fp16x8               mean time: 0.002133 ms
embedding_fp16x8_packed        mean time: 0.002106 ms
####################################################################################################
emd_size: 32, emd_dim: 8, seq_len: 64
torch                          mean time: 0.007300 ms
embedding                      mean time: 0.002140 ms
embedding_fp32x4               mean time: 0.002133 ms
embedding_fp32x4_packed        mean time: 0.002139 ms
torch                          mean time: 0.007275 ms
embedding_half                 mean time: 0.002115 ms
embedding_fp16x8               mean time: 0.002163 ms
embedding_fp16x8_packed        mean time: 0.002121 ms
####################################################################################################
emd_size: 32, emd_dim: 8, seq_len: 256
torch                          mean time: 0.007329 ms
embedding                      mean time: 0.002154 ms
embedding_fp32x4               mean time: 0.002165 ms
embedding_fp32x4_packed        mean time: 0.002175 ms
torch                          mean time: 0.007307 ms
embedding_half                 mean time: 0.002158 ms
embedding_fp16x8               mean time: 0.002191 ms
embedding_fp16x8_packed        mean time: 0.002159 ms
####################################################################################################
emd_size: 32, emd_dim: 8, seq_len: 1024
torch                          mean time: 0.007308 ms
embedding                      mean time: 0.002218 ms
embedding_fp32x4               mean time: 0.002344 ms
embedding_fp32x4_packed        mean time: 0.002229 ms
torch                          mean time: 0.007329 ms
embedding_half                 mean time: 0.002226 ms
embedding_fp16x8               mean time: 0.002450 ms
embedding_fp16x8_packed        mean time: 0.002240 ms
####################################################################################################
emd_size: 32, emd_dim: 8, seq_len: 4096
torch                          mean time: 0.007292 ms
embedding                      mean time: 0.003824 ms
embedding_fp32x4               mean time: 0.003869 ms
embedding_fp32x4_packed        mean time: 0.003824 ms
torch                          mean time: 0.007263 ms
embedding_half                 mean time: 0.003826 ms
embedding_fp16x8               mean time: 0.004020 ms
embedding_fp16x8_packed        mean time: 0.003829 ms
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 1
torch                          mean time: 0.005434 ms
embedding                      mean time: 0.002083 ms
embedding_fp32x4               mean time: 0.002109 ms
embedding_fp32x4_packed        mean time: 0.002120 ms
torch                          mean time: 0.005496 ms
embedding_half                 mean time: 0.002107 ms
embedding_fp16x8               mean time: 0.002136 ms
embedding_fp16x8_packed        mean time: 0.002121 ms
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007360 ms
embedding                      mean time: 0.002156 ms
embedding_fp32x4               mean time: 0.002149 ms
embedding_fp32x4_packed        mean time: 0.002146 ms
torch                          mean time: 0.007318 ms
embedding_half                 mean time: 0.002123 ms
embedding_fp16x8               mean time: 0.002156 ms
embedding_fp16x8_packed        mean time: 0.002131 ms
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007303 ms
embedding                      mean time: 0.002166 ms
embedding_fp32x4               mean time: 0.002185 ms
embedding_fp32x4_packed        mean time: 0.002177 ms
torch                          mean time: 0.007319 ms
embedding_half                 mean time: 0.002154 ms
embedding_fp16x8               mean time: 0.002145 ms
embedding_fp16x8_packed        mean time: 0.002167 ms
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.007326 ms
embedding                      mean time: 0.002239 ms
embedding_fp32x4               mean time: 0.002347 ms
embedding_fp32x4_packed        mean time: 0.002249 ms
torch                          mean time: 0.007302 ms
embedding_half                 mean time: 0.002233 ms
embedding_fp16x8               mean time: 0.002458 ms
embedding_fp16x8_packed        mean time: 0.002234 ms
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 4096
torch                          mean time: 0.007354 ms
embedding                      mean time: 0.003827 ms
embedding_fp32x4               mean time: 0.003880 ms
embedding_fp32x4_packed        mean time: 0.003829 ms
torch                          mean time: 0.007334 ms
embedding_half                 mean time: 0.003826 ms
embedding_fp16x8               mean time: 0.003984 ms
embedding_fp16x8_packed        mean time: 0.003829 ms
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 1
torch                          mean time: 0.005517 ms
embedding                      mean time: 0.002099 ms
embedding_fp32x4               mean time: 0.002145 ms
embedding_fp32x4_packed        mean time: 0.002141 ms
torch                          mean time: 0.005513 ms
embedding_half                 mean time: 0.002090 ms
embedding_fp16x8               mean time: 0.002130 ms
embedding_fp16x8_packed        mean time: 0.002106 ms
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007330 ms
embedding                      mean time: 0.002129 ms
embedding_fp32x4               mean time: 0.002144 ms
embedding_fp32x4_packed        mean time: 0.002135 ms
torch                          mean time: 0.007319 ms
embedding_half                 mean time: 0.002116 ms
embedding_fp16x8               mean time: 0.002165 ms
embedding_fp16x8_packed        mean time: 0.002122 ms
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007315 ms
embedding                      mean time: 0.002144 ms
embedding_fp32x4               mean time: 0.002147 ms
embedding_fp32x4_packed        mean time: 0.002182 ms
torch                          mean time: 0.007360 ms
embedding_half                 mean time: 0.002141 ms
embedding_fp16x8               mean time: 0.002175 ms
embedding_fp16x8_packed        mean time: 0.002159 ms
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.007360 ms
embedding                      mean time: 0.002197 ms
embedding_fp32x4               mean time: 0.002472 ms
embedding_fp32x4_packed        mean time: 0.002295 ms
torch                          mean time: 0.007321 ms
embedding_half                 mean time: 0.002199 ms
embedding_fp16x8               mean time: 0.002755 ms
embedding_fp16x8_packed        mean time: 0.002251 ms
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 4096
torch                          mean time: 0.007226 ms
embedding                      mean time: 0.003909 ms
embedding_fp32x4               mean time: 0.004149 ms
embedding_fp32x4_packed        mean time: 0.003833 ms
torch                          mean time: 0.007271 ms
embedding_half                 mean time: 0.003868 ms
embedding_fp16x8               mean time: 0.005227 ms
embedding_fp16x8_packed        mean time: 0.003853 ms
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 1
torch                          mean time: 0.005518 ms
embedding                      mean time: 0.002100 ms
embedding_fp32x4               mean time: 0.002122 ms
embedding_fp32x4_packed        mean time: 0.002124 ms
torch                          mean time: 0.005547 ms
embedding_half                 mean time: 0.002083 ms
embedding_fp16x8               mean time: 0.002134 ms
embedding_fp16x8_packed        mean time: 0.002092 ms
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007352 ms
embedding                      mean time: 0.002124 ms
embedding_fp32x4               mean time: 0.002146 ms
embedding_fp32x4_packed        mean time: 0.002151 ms
torch                          mean time: 0.007288 ms
embedding_half                 mean time: 0.002116 ms
embedding_fp16x8               mean time: 0.002173 ms
embedding_fp16x8_packed        mean time: 0.002124 ms
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007336 ms
embedding                      mean time: 0.002158 ms
embedding_fp32x4               mean time: 0.002412 ms
embedding_fp32x4_packed        mean time: 0.002188 ms
torch                          mean time: 0.007357 ms
embedding_half                 mean time: 0.002155 ms
embedding_fp16x8               mean time: 0.002656 ms
embedding_fp16x8_packed        mean time: 0.002165 ms
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.007364 ms
embedding                      mean time: 0.003260 ms
embedding_fp32x4               mean time: 0.005414 ms
embedding_fp32x4_packed        mean time: 0.002600 ms
torch                          mean time: 0.007321 ms
embedding_half                 mean time: 0.003189 ms
embedding_fp16x8               mean time: 0.005707 ms
embedding_fp16x8_packed        mean time: 0.002310 ms
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 4096
torch                          mean time: 0.007297 ms
embedding                      mean time: 0.007767 ms
embedding_fp32x4               mean time: 0.016618 ms
embedding_fp32x4_packed        mean time: 0.004954 ms
torch                          mean time: 0.007178 ms
embedding_half                 mean time: 0.007586 ms
embedding_fp16x8               mean time: 0.016973 ms
embedding_fp16x8_packed        mean time: 0.003854 ms
####################################################################################################
emd_size: 32, emd_dim: 1024, seq_len: 1
torch                          mean time: 0.005561 ms
embedding                      mean time: 0.002123 ms
embedding_fp32x4               mean time: 0.002157 ms
embedding_fp32x4_packed        mean time: 0.002143 ms
torch                          mean time: 0.005548 ms
embedding_half                 mean time: 0.002128 ms
embedding_fp16x8               mean time: 0.002142 ms
embedding_fp16x8_packed        mean time: 0.002142 ms
####################################################################################################
emd_size: 32, emd_dim: 1024, seq_len: 64
torch                          mean time: 0.007368 ms
embedding                      mean time: 0.002143 ms
embedding_fp32x4               mean time: 0.002115 ms
embedding_fp32x4_packed        mean time: 0.002170 ms
torch                          mean time: 0.007312 ms
embedding_half                 mean time: 0.002140 ms
embedding_fp16x8               mean time: 0.002264 ms
embedding_fp16x8_packed        mean time: 0.002152 ms
####################################################################################################
emd_size: 32, emd_dim: 1024, seq_len: 256
torch                          mean time: 0.007345 ms
embedding                      mean time: 0.002181 ms
embedding_fp32x4               mean time: 0.003123 ms
embedding_fp32x4_packed        mean time: 0.002188 ms
torch                          mean time: 0.007310 ms
embedding_half                 mean time: 0.002087 ms
embedding_fp16x8               mean time: 0.003839 ms
embedding_fp16x8_packed        mean time: 0.002108 ms
####################################################################################################
emd_size: 32, emd_dim: 1024, seq_len: 1024
torch                          mean time: 0.007310 ms
embedding                      mean time: 0.003997 ms
embedding_fp32x4               mean time: 0.009455 ms
embedding_fp32x4_packed        mean time: 0.003478 ms
torch                          mean time: 0.007325 ms
embedding_half                 mean time: 0.003741 ms
embedding_fp16x8               mean time: 0.009868 ms
embedding_fp16x8_packed        mean time: 0.002706 ms
####################################################################################################
emd_size: 32, emd_dim: 1024, seq_len: 4096
torch                          mean time: 0.008184 ms
embedding                      mean time: 0.010132 ms
embedding_fp32x4               mean time: 0.032380 ms
embedding_fp32x4_packed        mean time: 0.008150 ms
torch                          mean time: 0.007287 ms
embedding_half                 mean time: 0.009071 ms
embedding_fp16x8               mean time: 0.032947 ms
embedding_fp16x8_packed        mean time: 0.005040 ms
####################################################################################################
emd_size: 32, emd_dim: 2048, seq_len: 1
torch                          mean time: 0.005525 ms
embedding                      mean time: 0.002115 ms
embedding_fp32x4               mean time: 0.002368 ms
embedding_fp32x4_packed        mean time: 0.002128 ms
torch                          mean time: 0.005515 ms
embedding_half                 mean time: 0.002050 ms
embedding_fp16x8               mean time: 0.002588 ms
embedding_fp16x8_packed        mean time: 0.002111 ms
####################################################################################################
emd_size: 32, emd_dim: 2048, seq_len: 64
torch                          mean time: 0.007339 ms
embedding                      mean time: 0.002203 ms
embedding_fp32x4               mean time: 0.002629 ms
embedding_fp32x4_packed        mean time: 0.002263 ms
torch                          mean time: 0.007185 ms
embedding_half                 mean time: 0.002143 ms
embedding_fp16x8               mean time: 0.002930 ms
embedding_fp16x8_packed        mean time: 0.002234 ms
####################################################################################################
emd_size: 32, emd_dim: 2048, seq_len: 256
torch                          mean time: 0.007258 ms
embedding                      mean time: 0.002763 ms
embedding_fp32x4               mean time: 0.004813 ms
embedding_fp32x4_packed        mean time: 0.002712 ms
torch                          mean time: 0.007297 ms
embedding_half                 mean time: 0.002342 ms
embedding_fp16x8               mean time: 0.005431 ms
embedding_fp16x8_packed        mean time: 0.002360 ms
####################################################################################################
emd_size: 32, emd_dim: 2048, seq_len: 1024
torch                          mean time: 0.007363 ms
embedding                      mean time: 0.005772 ms
embedding_fp32x4               mean time: 0.018550 ms
embedding_fp32x4_packed        mean time: 0.005209 ms
torch                          mean time: 0.007320 ms
embedding_half                 mean time: 0.004913 ms
embedding_fp16x8               mean time: 0.018234 ms
embedding_fp16x8_packed        mean time: 0.003593 ms
####################################################################################################
emd_size: 32, emd_dim: 2048, seq_len: 4096
torch                          mean time: 0.014480 ms
embedding                      mean time: 0.016397 ms
embedding_fp32x4               mean time: 0.065291 ms
embedding_fp32x4_packed        mean time: 0.014925 ms
torch                          mean time: 0.008192 ms
embedding_half                 mean time: 0.013115 ms
embedding_fp16x8               mean time: 0.065761 ms
embedding_fp16x8_packed        mean time: 0.008340 ms
####################################################################################################
emd_size: 32, emd_dim: 4096, seq_len: 1
torch                          mean time: 0.005553 ms
embedding                      mean time: 0.002512 ms
embedding_fp32x4               mean time: 0.003263 ms
embedding_fp32x4_packed        mean time: 0.002661 ms
torch                          mean time: 0.005536 ms
embedding_half                 mean time: 0.002496 ms
embedding_fp16x8               mean time: 0.003818 ms
embedding_fp16x8_packed        mean time: 0.002603 ms
####################################################################################################
emd_size: 32, emd_dim: 4096, seq_len: 64
torch                          mean time: 0.007301 ms
embedding                      mean time: 0.002795 ms
embedding_fp32x4               mean time: 0.003825 ms
embedding_fp32x4_packed        mean time: 0.002916 ms
torch                          mean time: 0.007203 ms
embedding_half                 mean time: 0.002684 ms
embedding_fp16x8               mean time: 0.004416 ms
embedding_fp16x8_packed        mean time: 0.002864 ms
####################################################################################################
emd_size: 32, emd_dim: 4096, seq_len: 256
torch                          mean time: 0.007370 ms
embedding                      mean time: 0.003696 ms
embedding_fp32x4               mean time: 0.007075 ms
embedding_fp32x4_packed        mean time: 0.003614 ms
torch                          mean time: 0.007305 ms
embedding_half                 mean time: 0.003056 ms
embedding_fp16x8               mean time: 0.009483 ms
embedding_fp16x8_packed        mean time: 0.003088 ms
####################################################################################################
emd_size: 32, emd_dim: 4096, seq_len: 1024
torch                          mean time: 0.008514 ms
embedding                      mean time: 0.009330 ms
embedding_fp32x4               mean time: 0.035024 ms
embedding_fp32x4_packed        mean time: 0.008970 ms
torch                          mean time: 0.007283 ms
embedding_half                 mean time: 0.007342 ms
embedding_fp16x8               mean time: 0.036423 ms
embedding_fp16x8_packed        mean time: 0.005282 ms
####################################################################################################
emd_size: 32, emd_dim: 4096, seq_len: 4096
torch                          mean time: 0.029403 ms
embedding                      mean time: 0.029985 ms
embedding_fp32x4               mean time: 0.132187 ms
embedding_fp32x4_packed        mean time: 0.029630 ms
torch                          mean time: 0.014513 ms
embedding_half                 mean time: 0.021969 ms
embedding_fp16x8               mean time: 0.135369 ms
embedding_fp16x8_packed        mean time: 0.015337 ms
####################################################################################################
emd_size: 128, emd_dim: 8, seq_len: 1
torch                          mean time: 0.005536 ms
embedding                      mean time: 0.002100 ms
embedding_fp32x4               mean time: 0.002104 ms
embedding_fp32x4_packed        mean time: 0.002110 ms
torch                          mean time: 0.005533 ms
embedding_half                 mean time: 0.002087 ms
embedding_fp16x8               mean time: 0.002135 ms
embedding_fp16x8_packed        mean time: 0.002093 ms
####################################################################################################
emd_size: 128, emd_dim: 8, seq_len: 64
torch                          mean time: 0.007316 ms
embedding                      mean time: 0.002143 ms
embedding_fp32x4               mean time: 0.002134 ms
embedding_fp32x4_packed        mean time: 0.002135 ms
torch                          mean time: 0.007298 ms
embedding_half                 mean time: 0.002108 ms
embedding_fp16x8               mean time: 0.002168 ms
embedding_fp16x8_packed        mean time: 0.002117 ms
####################################################################################################
emd_size: 128, emd_dim: 8, seq_len: 256
torch                          mean time: 0.007349 ms
embedding                      mean time: 0.002156 ms
embedding_fp32x4               mean time: 0.002151 ms
embedding_fp32x4_packed        mean time: 0.002166 ms
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.002133 ms
embedding_fp16x8               mean time: 0.002178 ms
embedding_fp16x8_packed        mean time: 0.002149 ms
####################################################################################################
emd_size: 128, emd_dim: 8, seq_len: 1024
torch                          mean time: 0.007342 ms
embedding                      mean time: 0.002234 ms
embedding_fp32x4               mean time: 0.002338 ms
embedding_fp32x4_packed        mean time: 0.002242 ms
torch                          mean time: 0.007303 ms
embedding_half                 mean time: 0.002223 ms
embedding_fp16x8               mean time: 0.002459 ms
embedding_fp16x8_packed        mean time: 0.002231 ms
####################################################################################################
emd_size: 128, emd_dim: 8, seq_len: 4096
torch                          mean time: 0.007307 ms
embedding                      mean time: 0.003843 ms
embedding_fp32x4               mean time: 0.003912 ms
embedding_fp32x4_packed        mean time: 0.003839 ms
torch                          mean time: 0.007223 ms
embedding_half                 mean time: 0.003827 ms
embedding_fp16x8               mean time: 0.004009 ms
embedding_fp16x8_packed        mean time: 0.003830 ms
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 1
torch                          mean time: 0.005529 ms
embedding                      mean time: 0.002108 ms
embedding_fp32x4               mean time: 0.002110 ms
embedding_fp32x4_packed        mean time: 0.002107 ms
torch                          mean time: 0.005516 ms
embedding_half                 mean time: 0.002084 ms
embedding_fp16x8               mean time: 0.002120 ms
embedding_fp16x8_packed        mean time: 0.002087 ms
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007253 ms
embedding                      mean time: 0.002126 ms
embedding_fp32x4               mean time: 0.002138 ms
embedding_fp32x4_packed        mean time: 0.002136 ms
torch                          mean time: 0.007315 ms
embedding_half                 mean time: 0.002105 ms
embedding_fp16x8               mean time: 0.002161 ms
embedding_fp16x8_packed        mean time: 0.002128 ms
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007309 ms
embedding                      mean time: 0.002170 ms
embedding_fp32x4               mean time: 0.002163 ms
embedding_fp32x4_packed        mean time: 0.002165 ms
torch                          mean time: 0.007335 ms
embedding_half                 mean time: 0.002150 ms
embedding_fp16x8               mean time: 0.002162 ms
embedding_fp16x8_packed        mean time: 0.002150 ms
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.007330 ms
embedding                      mean time: 0.002228 ms
embedding_fp32x4               mean time: 0.002324 ms
embedding_fp32x4_packed        mean time: 0.002239 ms
torch                          mean time: 0.007306 ms
embedding_half                 mean time: 0.002213 ms
embedding_fp16x8               mean time: 0.002436 ms
embedding_fp16x8_packed        mean time: 0.002220 ms
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 4096
torch                          mean time: 0.007365 ms
embedding                      mean time: 0.003839 ms
embedding_fp32x4               mean time: 0.003907 ms
embedding_fp32x4_packed        mean time: 0.003847 ms
torch                          mean time: 0.007234 ms
embedding_half                 mean time: 0.003843 ms
embedding_fp16x8               mean time: 0.004011 ms
embedding_fp16x8_packed        mean time: 0.003847 ms
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 1
torch                          mean time: 0.005534 ms
embedding                      mean time: 0.002100 ms
embedding_fp32x4               mean time: 0.002126 ms
embedding_fp32x4_packed        mean time: 0.002110 ms
torch                          mean time: 0.005510 ms
embedding_half                 mean time: 0.002097 ms
embedding_fp16x8               mean time: 0.002129 ms
embedding_fp16x8_packed        mean time: 0.002100 ms
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007229 ms
embedding                      mean time: 0.002123 ms
embedding_fp32x4               mean time: 0.002132 ms
embedding_fp32x4_packed        mean time: 0.002143 ms
torch                          mean time: 0.007313 ms
embedding_half                 mean time: 0.002109 ms
embedding_fp16x8               mean time: 0.002169 ms
embedding_fp16x8_packed        mean time: 0.002109 ms
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007363 ms
embedding                      mean time: 0.002148 ms
embedding_fp32x4               mean time: 0.002118 ms
embedding_fp32x4_packed        mean time: 0.002195 ms
torch                          mean time: 0.007316 ms
embedding_half                 mean time: 0.002132 ms
embedding_fp16x8               mean time: 0.002168 ms
embedding_fp16x8_packed        mean time: 0.002160 ms
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.007349 ms
embedding                      mean time: 0.002199 ms
embedding_fp32x4               mean time: 0.002520 ms
embedding_fp32x4_packed        mean time: 0.002269 ms
torch                          mean time: 0.007294 ms
embedding_half                 mean time: 0.002178 ms
embedding_fp16x8               mean time: 0.002722 ms
embedding_fp16x8_packed        mean time: 0.002246 ms
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 4096
torch                          mean time: 0.007364 ms
embedding                      mean time: 0.003938 ms
embedding_fp32x4               mean time: 0.004226 ms
embedding_fp32x4_packed        mean time: 0.003850 ms
torch                          mean time: 0.007302 ms
embedding_half                 mean time: 0.003894 ms
embedding_fp16x8               mean time: 0.005245 ms
embedding_fp16x8_packed        mean time: 0.003846 ms
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 1
torch                          mean time: 0.005535 ms
embedding                      mean time: 0.002083 ms
embedding_fp32x4               mean time: 0.002108 ms
embedding_fp32x4_packed        mean time: 0.002097 ms
torch                          mean time: 0.005519 ms
embedding_half                 mean time: 0.002086 ms
embedding_fp16x8               mean time: 0.002153 ms
embedding_fp16x8_packed        mean time: 0.002115 ms
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007298 ms
embedding                      mean time: 0.002126 ms
embedding_fp32x4               mean time: 0.002144 ms
embedding_fp32x4_packed        mean time: 0.002146 ms
torch                          mean time: 0.007318 ms
embedding_half                 mean time: 0.002104 ms
embedding_fp16x8               mean time: 0.002174 ms
embedding_fp16x8_packed        mean time: 0.002135 ms
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007332 ms
embedding                      mean time: 0.002152 ms
embedding_fp32x4               mean time: 0.002503 ms
embedding_fp32x4_packed        mean time: 0.002193 ms
torch                          mean time: 0.007282 ms
embedding_half                 mean time: 0.002148 ms
embedding_fp16x8               mean time: 0.002621 ms
embedding_fp16x8_packed        mean time: 0.002154 ms
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.007362 ms
embedding                      mean time: 0.003280 ms
embedding_fp32x4               mean time: 0.005728 ms
embedding_fp32x4_packed        mean time: 0.002570 ms
torch                          mean time: 0.007292 ms
embedding_half                 mean time: 0.003227 ms
embedding_fp16x8               mean time: 0.005538 ms
embedding_fp16x8_packed        mean time: 0.002275 ms
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 4096
torch                          mean time: 0.007382 ms
embedding                      mean time: 0.008059 ms
embedding_fp32x4               mean time: 0.016696 ms
embedding_fp32x4_packed        mean time: 0.005041 ms
torch                          mean time: 0.007311 ms
embedding_half                 mean time: 0.007780 ms
embedding_fp16x8               mean time: 0.017054 ms
embedding_fp16x8_packed        mean time: 0.003855 ms
####################################################################################################
emd_size: 128, emd_dim: 1024, seq_len: 1
torch                          mean time: 0.005537 ms
embedding                      mean time: 0.002122 ms
embedding_fp32x4               mean time: 0.002123 ms
embedding_fp32x4_packed        mean time: 0.002143 ms
torch                          mean time: 0.005533 ms
embedding_half                 mean time: 0.002123 ms
embedding_fp16x8               mean time: 0.002145 ms
embedding_fp16x8_packed        mean time: 0.002133 ms
####################################################################################################
emd_size: 128, emd_dim: 1024, seq_len: 64
torch                          mean time: 0.007344 ms
embedding                      mean time: 0.002132 ms
embedding_fp32x4               mean time: 0.002107 ms
embedding_fp32x4_packed        mean time: 0.002168 ms
torch                          mean time: 0.007298 ms
embedding_half                 mean time: 0.002159 ms
embedding_fp16x8               mean time: 0.002230 ms
embedding_fp16x8_packed        mean time: 0.002146 ms
####################################################################################################
emd_size: 128, emd_dim: 1024, seq_len: 256
torch                          mean time: 0.007368 ms
embedding                      mean time: 0.002168 ms
embedding_fp32x4               mean time: 0.003139 ms
embedding_fp32x4_packed        mean time: 0.002182 ms
torch                          mean time: 0.007295 ms
embedding_half                 mean time: 0.002109 ms
embedding_fp16x8               mean time: 0.003584 ms
embedding_fp16x8_packed        mean time: 0.002114 ms
####################################################################################################
emd_size: 128, emd_dim: 1024, seq_len: 1024
torch                          mean time: 0.007409 ms
embedding                      mean time: 0.004025 ms
embedding_fp32x4               mean time: 0.009681 ms
embedding_fp32x4_packed        mean time: 0.003473 ms
torch                          mean time: 0.007331 ms
embedding_half                 mean time: 0.003791 ms
embedding_fp16x8               mean time: 0.009837 ms
embedding_fp16x8_packed        mean time: 0.002803 ms
####################################################################################################
emd_size: 128, emd_dim: 1024, seq_len: 4096
torch                          mean time: 0.008364 ms
embedding                      mean time: 0.010727 ms
embedding_fp32x4               mean time: 0.032456 ms
embedding_fp32x4_packed        mean time: 0.008303 ms
torch                          mean time: 0.007324 ms
embedding_half                 mean time: 0.009789 ms
embedding_fp16x8               mean time: 0.032942 ms
embedding_fp16x8_packed        mean time: 0.005102 ms
####################################################################################################
emd_size: 128, emd_dim: 2048, seq_len: 1
torch                          mean time: 0.005537 ms
embedding                      mean time: 0.002114 ms
embedding_fp32x4               mean time: 0.002389 ms
embedding_fp32x4_packed        mean time: 0.002135 ms
torch                          mean time: 0.005540 ms
embedding_half                 mean time: 0.002111 ms
embedding_fp16x8               mean time: 0.002631 ms
embedding_fp16x8_packed        mean time: 0.002116 ms
####################################################################################################
emd_size: 128, emd_dim: 2048, seq_len: 64
torch                          mean time: 0.007297 ms
embedding                      mean time: 0.002217 ms
embedding_fp32x4               mean time: 0.002643 ms
embedding_fp32x4_packed        mean time: 0.002267 ms
torch                          mean time: 0.007305 ms
embedding_half                 mean time: 0.002150 ms
embedding_fp16x8               mean time: 0.002902 ms
embedding_fp16x8_packed        mean time: 0.002203 ms
####################################################################################################
emd_size: 128, emd_dim: 2048, seq_len: 256
torch                          mean time: 0.007374 ms
embedding                      mean time: 0.002738 ms
embedding_fp32x4               mean time: 0.004804 ms
embedding_fp32x4_packed        mean time: 0.002679 ms
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.002323 ms
embedding_fp16x8               mean time: 0.005380 ms
embedding_fp16x8_packed        mean time: 0.002357 ms
####################################################################################################
emd_size: 128, emd_dim: 2048, seq_len: 1024
torch                          mean time: 0.007336 ms
embedding                      mean time: 0.005814 ms
embedding_fp32x4               mean time: 0.018305 ms
embedding_fp32x4_packed        mean time: 0.005210 ms
torch                          mean time: 0.007300 ms
embedding_half                 mean time: 0.004997 ms
embedding_fp16x8               mean time: 0.018808 ms
embedding_fp16x8_packed        mean time: 0.003551 ms
####################################################################################################
emd_size: 128, emd_dim: 2048, seq_len: 4096
torch                          mean time: 0.014793 ms
embedding                      mean time: 0.017040 ms
embedding_fp32x4               mean time: 0.065301 ms
embedding_fp32x4_packed        mean time: 0.015100 ms
torch                          mean time: 0.008377 ms
embedding_half                 mean time: 0.014129 ms
embedding_fp16x8               mean time: 0.065848 ms
embedding_fp16x8_packed        mean time: 0.008486 ms
####################################################################################################
emd_size: 128, emd_dim: 4096, seq_len: 1
torch                          mean time: 0.005536 ms
embedding                      mean time: 0.002512 ms
embedding_fp32x4               mean time: 0.003257 ms
embedding_fp32x4_packed        mean time: 0.002669 ms
torch                          mean time: 0.005548 ms
embedding_half                 mean time: 0.002498 ms
embedding_fp16x8               mean time: 0.003819 ms
embedding_fp16x8_packed        mean time: 0.002594 ms
####################################################################################################
emd_size: 128, emd_dim: 4096, seq_len: 64
torch                          mean time: 0.007339 ms
embedding                      mean time: 0.002773 ms
embedding_fp32x4               mean time: 0.003736 ms
embedding_fp32x4_packed        mean time: 0.002911 ms
torch                          mean time: 0.007293 ms
embedding_half                 mean time: 0.002688 ms
embedding_fp16x8               mean time: 0.004415 ms
embedding_fp16x8_packed        mean time: 0.002814 ms
####################################################################################################
emd_size: 128, emd_dim: 4096, seq_len: 256
torch                          mean time: 0.007351 ms
embedding                      mean time: 0.003704 ms
embedding_fp32x4               mean time: 0.007326 ms
embedding_fp32x4_packed        mean time: 0.003610 ms
torch                          mean time: 0.007193 ms
embedding_half                 mean time: 0.002997 ms
embedding_fp16x8               mean time: 0.009611 ms
embedding_fp16x8_packed        mean time: 0.003061 ms
####################################################################################################
emd_size: 128, emd_dim: 4096, seq_len: 1024
torch                          mean time: 0.008578 ms
embedding                      mean time: 0.009410 ms
embedding_fp32x4               mean time: 0.034660 ms
embedding_fp32x4_packed        mean time: 0.008932 ms
torch                          mean time: 0.007297 ms
embedding_half                 mean time: 0.007376 ms
embedding_fp16x8               mean time: 0.036916 ms
embedding_fp16x8_packed        mean time: 0.005326 ms
####################################################################################################
emd_size: 128, emd_dim: 4096, seq_len: 4096
torch                          mean time: 0.029872 ms
embedding                      mean time: 0.030663 ms
embedding_fp32x4               mean time: 0.132196 ms
embedding_fp32x4_packed        mean time: 0.029909 ms
torch                          mean time: 0.014811 ms
embedding_half                 mean time: 0.023130 ms
embedding_fp16x8               mean time: 0.134951 ms
embedding_fp16x8_packed        mean time: 0.015670 ms
####################################################################################################
emd_size: 1024, emd_dim: 8, seq_len: 1
torch                          mean time: 0.005537 ms
embedding                      mean time: 0.002092 ms
embedding_fp32x4               mean time: 0.002100 ms
embedding_fp32x4_packed        mean time: 0.002108 ms
torch                          mean time: 0.005524 ms
embedding_half                 mean time: 0.002088 ms
embedding_fp16x8               mean time: 0.002132 ms
embedding_fp16x8_packed        mean time: 0.002095 ms
####################################################################################################
emd_size: 1024, emd_dim: 8, seq_len: 64
torch                          mean time: 0.007319 ms
embedding                      mean time: 0.002122 ms
embedding_fp32x4               mean time: 0.002125 ms
embedding_fp32x4_packed        mean time: 0.002145 ms
torch                          mean time: 0.007292 ms
embedding_half                 mean time: 0.002108 ms
embedding_fp16x8               mean time: 0.002156 ms
embedding_fp16x8_packed        mean time: 0.002111 ms
####################################################################################################
emd_size: 1024, emd_dim: 8, seq_len: 256
torch                          mean time: 0.007336 ms
embedding                      mean time: 0.002156 ms
embedding_fp32x4               mean time: 0.002157 ms
embedding_fp32x4_packed        mean time: 0.002170 ms
torch                          mean time: 0.007295 ms
embedding_half                 mean time: 0.002148 ms
embedding_fp16x8               mean time: 0.002170 ms
embedding_fp16x8_packed        mean time: 0.002147 ms
####################################################################################################
emd_size: 1024, emd_dim: 8, seq_len: 1024
torch                          mean time: 0.007314 ms
embedding                      mean time: 0.002222 ms
embedding_fp32x4               mean time: 0.002347 ms
embedding_fp32x4_packed        mean time: 0.002234 ms
torch                          mean time: 0.007294 ms
embedding_half                 mean time: 0.002212 ms
embedding_fp16x8               mean time: 0.002500 ms
embedding_fp16x8_packed        mean time: 0.002223 ms
####################################################################################################
emd_size: 1024, emd_dim: 8, seq_len: 4096
torch                          mean time: 0.007310 ms
embedding                      mean time: 0.003881 ms
embedding_fp32x4               mean time: 0.003961 ms
embedding_fp32x4_packed        mean time: 0.003887 ms
torch                          mean time: 0.007256 ms
embedding_half                 mean time: 0.003864 ms
embedding_fp16x8               mean time: 0.004001 ms
embedding_fp16x8_packed        mean time: 0.003875 ms
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 1
torch                          mean time: 0.005539 ms
embedding                      mean time: 0.002088 ms
embedding_fp32x4               mean time: 0.002102 ms
embedding_fp32x4_packed        mean time: 0.002101 ms
torch                          mean time: 0.005530 ms
embedding_half                 mean time: 0.002092 ms
embedding_fp16x8               mean time: 0.002132 ms
embedding_fp16x8_packed        mean time: 0.002103 ms
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007330 ms
embedding                      mean time: 0.002121 ms
embedding_fp32x4               mean time: 0.002124 ms
embedding_fp32x4_packed        mean time: 0.002131 ms
torch                          mean time: 0.007276 ms
embedding_half                 mean time: 0.002107 ms
embedding_fp16x8               mean time: 0.002169 ms
embedding_fp16x8_packed        mean time: 0.002122 ms
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007331 ms
embedding                      mean time: 0.002162 ms
embedding_fp32x4               mean time: 0.002180 ms
embedding_fp32x4_packed        mean time: 0.002177 ms
torch                          mean time: 0.007314 ms
embedding_half                 mean time: 0.002147 ms
embedding_fp16x8               mean time: 0.002165 ms
embedding_fp16x8_packed        mean time: 0.002154 ms
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.007331 ms
embedding                      mean time: 0.002218 ms
embedding_fp32x4               mean time: 0.002319 ms
embedding_fp32x4_packed        mean time: 0.002232 ms
torch                          mean time: 0.007275 ms
embedding_half                 mean time: 0.002225 ms
embedding_fp16x8               mean time: 0.002460 ms
embedding_fp16x8_packed        mean time: 0.002236 ms
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 4096
torch                          mean time: 0.007308 ms
embedding                      mean time: 0.003881 ms
embedding_fp32x4               mean time: 0.003949 ms
embedding_fp32x4_packed        mean time: 0.003889 ms
torch                          mean time: 0.007275 ms
embedding_half                 mean time: 0.003870 ms
embedding_fp16x8               mean time: 0.004012 ms
embedding_fp16x8_packed        mean time: 0.003883 ms
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 1
torch                          mean time: 0.005561 ms
embedding                      mean time: 0.002107 ms
embedding_fp32x4               mean time: 0.002115 ms
embedding_fp32x4_packed        mean time: 0.002152 ms
torch                          mean time: 0.005527 ms
embedding_half                 mean time: 0.002093 ms
embedding_fp16x8               mean time: 0.002134 ms
embedding_fp16x8_packed        mean time: 0.002109 ms
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007311 ms
embedding                      mean time: 0.002121 ms
embedding_fp32x4               mean time: 0.002141 ms
embedding_fp32x4_packed        mean time: 0.002136 ms
torch                          mean time: 0.007312 ms
embedding_half                 mean time: 0.002104 ms
embedding_fp16x8               mean time: 0.002162 ms
embedding_fp16x8_packed        mean time: 0.002111 ms
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007310 ms
embedding                      mean time: 0.002144 ms
embedding_fp32x4               mean time: 0.002133 ms
embedding_fp32x4_packed        mean time: 0.002176 ms
torch                          mean time: 0.007272 ms
embedding_half                 mean time: 0.002140 ms
embedding_fp16x8               mean time: 0.002162 ms
embedding_fp16x8_packed        mean time: 0.002162 ms
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.007318 ms
embedding                      mean time: 0.002196 ms
embedding_fp32x4               mean time: 0.002552 ms
embedding_fp32x4_packed        mean time: 0.002276 ms
torch                          mean time: 0.007294 ms
embedding_half                 mean time: 0.002182 ms
embedding_fp16x8               mean time: 0.002680 ms
embedding_fp16x8_packed        mean time: 0.002236 ms
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 4096
torch                          mean time: 0.007328 ms
embedding                      mean time: 0.003942 ms
embedding_fp32x4               mean time: 0.004245 ms
embedding_fp32x4_packed        mean time: 0.003869 ms
torch                          mean time: 0.007287 ms
embedding_half                 mean time: 0.003903 ms
embedding_fp16x8               mean time: 0.005012 ms
embedding_fp16x8_packed        mean time: 0.003872 ms
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 1
torch                          mean time: 0.005531 ms
embedding                      mean time: 0.002090 ms
embedding_fp32x4               mean time: 0.002114 ms
embedding_fp32x4_packed        mean time: 0.002111 ms
torch                          mean time: 0.005555 ms
embedding_half                 mean time: 0.002092 ms
embedding_fp16x8               mean time: 0.002147 ms
embedding_fp16x8_packed        mean time: 0.002109 ms
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007401 ms
embedding                      mean time: 0.002131 ms
embedding_fp32x4               mean time: 0.002151 ms
embedding_fp32x4_packed        mean time: 0.002145 ms
torch                          mean time: 0.007278 ms
embedding_half                 mean time: 0.002128 ms
embedding_fp16x8               mean time: 0.002172 ms
embedding_fp16x8_packed        mean time: 0.002130 ms
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007328 ms
embedding                      mean time: 0.002153 ms
embedding_fp32x4               mean time: 0.002408 ms
embedding_fp32x4_packed        mean time: 0.002178 ms
torch                          mean time: 0.007288 ms
embedding_half                 mean time: 0.002146 ms
embedding_fp16x8               mean time: 0.002573 ms
embedding_fp16x8_packed        mean time: 0.002162 ms
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.007364 ms
embedding                      mean time: 0.003268 ms
embedding_fp32x4               mean time: 0.005471 ms
embedding_fp32x4_packed        mean time: 0.002599 ms
torch                          mean time: 0.007274 ms
embedding_half                 mean time: 0.003225 ms
embedding_fp16x8               mean time: 0.006019 ms
embedding_fp16x8_packed        mean time: 0.002302 ms
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 4096
torch                          mean time: 0.007342 ms
embedding                      mean time: 0.008184 ms
embedding_fp32x4               mean time: 0.016731 ms
embedding_fp32x4_packed        mean time: 0.005049 ms
torch                          mean time: 0.007286 ms
embedding_half                 mean time: 0.007902 ms
embedding_fp16x8               mean time: 0.017106 ms
embedding_fp16x8_packed        mean time: 0.003859 ms
####################################################################################################
emd_size: 1024, emd_dim: 1024, seq_len: 1
torch                          mean time: 0.005572 ms
embedding                      mean time: 0.002127 ms
embedding_fp32x4               mean time: 0.002129 ms
embedding_fp32x4_packed        mean time: 0.002146 ms
torch                          mean time: 0.005543 ms
embedding_half                 mean time: 0.002131 ms
embedding_fp16x8               mean time: 0.002158 ms
embedding_fp16x8_packed        mean time: 0.002143 ms
####################################################################################################
emd_size: 1024, emd_dim: 1024, seq_len: 64
torch                          mean time: 0.007300 ms
embedding                      mean time: 0.002153 ms
embedding_fp32x4               mean time: 0.002120 ms
embedding_fp32x4_packed        mean time: 0.002176 ms
torch                          mean time: 0.007303 ms
embedding_half                 mean time: 0.002141 ms
embedding_fp16x8               mean time: 0.002208 ms
embedding_fp16x8_packed        mean time: 0.002155 ms
####################################################################################################
emd_size: 1024, emd_dim: 1024, seq_len: 256
torch                          mean time: 0.007353 ms
embedding                      mean time: 0.002166 ms
embedding_fp32x4               mean time: 0.003159 ms
embedding_fp32x4_packed        mean time: 0.002166 ms
torch                          mean time: 0.007302 ms
embedding_half                 mean time: 0.002119 ms
embedding_fp16x8               mean time: 0.003598 ms
embedding_fp16x8_packed        mean time: 0.002117 ms
####################################################################################################
emd_size: 1024, emd_dim: 1024, seq_len: 1024
torch                          mean time: 0.007350 ms
embedding                      mean time: 0.004050 ms
embedding_fp32x4               mean time: 0.009522 ms
embedding_fp32x4_packed        mean time: 0.003442 ms
torch                          mean time: 0.007207 ms
embedding_half                 mean time: 0.003819 ms
embedding_fp16x8               mean time: 0.010093 ms
embedding_fp16x8_packed        mean time: 0.002740 ms
####################################################################################################
emd_size: 1024, emd_dim: 1024, seq_len: 4096
torch                          mean time: 0.008411 ms
embedding                      mean time: 0.010918 ms
embedding_fp32x4               mean time: 0.032458 ms
embedding_fp32x4_packed        mean time: 0.008349 ms
torch                          mean time: 0.007295 ms
embedding_half                 mean time: 0.009985 ms
embedding_fp16x8               mean time: 0.032921 ms
embedding_fp16x8_packed        mean time: 0.005131 ms
####################################################################################################
emd_size: 1024, emd_dim: 2048, seq_len: 1
torch                          mean time: 0.005541 ms
embedding                      mean time: 0.002110 ms
embedding_fp32x4               mean time: 0.002367 ms
embedding_fp32x4_packed        mean time: 0.002138 ms
torch                          mean time: 0.005534 ms
embedding_half                 mean time: 0.002114 ms
embedding_fp16x8               mean time: 0.002618 ms
embedding_fp16x8_packed        mean time: 0.002134 ms
####################################################################################################
emd_size: 1024, emd_dim: 2048, seq_len: 64
torch                          mean time: 0.007334 ms
embedding                      mean time: 0.002182 ms
embedding_fp32x4               mean time: 0.002629 ms
embedding_fp32x4_packed        mean time: 0.002249 ms
torch                          mean time: 0.007285 ms
embedding_half                 mean time: 0.002141 ms
embedding_fp16x8               mean time: 0.002919 ms
embedding_fp16x8_packed        mean time: 0.002207 ms
####################################################################################################
emd_size: 1024, emd_dim: 2048, seq_len: 256
torch                          mean time: 0.007349 ms
embedding                      mean time: 0.002636 ms
embedding_fp32x4               mean time: 0.004564 ms
embedding_fp32x4_packed        mean time: 0.002654 ms
torch                          mean time: 0.007341 ms
embedding_half                 mean time: 0.002318 ms
embedding_fp16x8               mean time: 0.005341 ms
embedding_fp16x8_packed        mean time: 0.002361 ms
####################################################################################################
emd_size: 1024, emd_dim: 2048, seq_len: 1024
torch                          mean time: 0.007397 ms
embedding                      mean time: 0.005869 ms
embedding_fp32x4               mean time: 0.018234 ms
embedding_fp32x4_packed        mean time: 0.005240 ms
torch                          mean time: 0.007332 ms
embedding_half                 mean time: 0.005003 ms
embedding_fp16x8               mean time: 0.018260 ms
embedding_fp16x8_packed        mean time: 0.003592 ms
####################################################################################################
emd_size: 1024, emd_dim: 2048, seq_len: 4096
torch                          mean time: 0.014915 ms
embedding                      mean time: 0.017182 ms
embedding_fp32x4               mean time: 0.065422 ms
embedding_fp32x4_packed        mean time: 0.015155 ms
torch                          mean time: 0.008427 ms
embedding_half                 mean time: 0.014462 ms
embedding_fp16x8               mean time: 0.066138 ms
embedding_fp16x8_packed        mean time: 0.008604 ms
####################################################################################################
emd_size: 1024, emd_dim: 4096, seq_len: 1
torch                          mean time: 0.005538 ms
embedding                      mean time: 0.002510 ms
embedding_fp32x4               mean time: 0.003264 ms
embedding_fp32x4_packed        mean time: 0.002670 ms
torch                          mean time: 0.005568 ms
embedding_half                 mean time: 0.002489 ms
embedding_fp16x8               mean time: 0.003812 ms
embedding_fp16x8_packed        mean time: 0.002576 ms
####################################################################################################
emd_size: 1024, emd_dim: 4096, seq_len: 64
torch                          mean time: 0.007309 ms
embedding                      mean time: 0.002791 ms
embedding_fp32x4               mean time: 0.003748 ms
embedding_fp32x4_packed        mean time: 0.002925 ms
torch                          mean time: 0.007313 ms
embedding_half                 mean time: 0.002682 ms
embedding_fp16x8               mean time: 0.004409 ms
embedding_fp16x8_packed        mean time: 0.002819 ms
####################################################################################################
emd_size: 1024, emd_dim: 4096, seq_len: 256
torch                          mean time: 0.007337 ms
embedding                      mean time: 0.003619 ms
embedding_fp32x4               mean time: 0.007713 ms
embedding_fp32x4_packed        mean time: 0.003552 ms
torch                          mean time: 0.007308 ms
embedding_half                 mean time: 0.002986 ms
embedding_fp16x8               mean time: 0.009411 ms
embedding_fp16x8_packed        mean time: 0.003066 ms
####################################################################################################
emd_size: 1024, emd_dim: 4096, seq_len: 1024
torch                          mean time: 0.008607 ms
embedding                      mean time: 0.009460 ms
embedding_fp32x4               mean time: 0.036653 ms
embedding_fp32x4_packed        mean time: 0.009009 ms
torch                          mean time: 0.007209 ms
embedding_half                 mean time: 0.007401 ms
embedding_fp16x8               mean time: 0.036070 ms
embedding_fp16x8_packed        mean time: 0.005334 ms
####################################################################################################
emd_size: 1024, emd_dim: 4096, seq_len: 4096
torch                          mean time: 0.030085 ms
embedding                      mean time: 0.030773 ms
embedding_fp32x4               mean time: 0.131821 ms
embedding_fp32x4_packed        mean time: 0.030031 ms
torch                          mean time: 0.014900 ms
embedding_half                 mean time: 0.023441 ms
embedding_fp16x8               mean time: 0.135252 ms
embedding_fp16x8_packed        mean time: 0.015543 ms
####################################################################################################
emd_size: 16384, emd_dim: 8, seq_len: 1
torch                          mean time: 0.005541 ms
embedding                      mean time: 0.002103 ms
embedding_fp32x4               mean time: 0.002100 ms
embedding_fp32x4_packed        mean time: 0.002099 ms
torch                          mean time: 0.005541 ms
embedding_half                 mean time: 0.002104 ms
embedding_fp16x8               mean time: 0.002144 ms
embedding_fp16x8_packed        mean time: 0.002099 ms
####################################################################################################
emd_size: 16384, emd_dim: 8, seq_len: 64
torch                          mean time: 0.007340 ms
embedding                      mean time: 0.002114 ms
embedding_fp32x4               mean time: 0.002131 ms
embedding_fp32x4_packed        mean time: 0.002127 ms
torch                          mean time: 0.007317 ms
embedding_half                 mean time: 0.002110 ms
embedding_fp16x8               mean time: 0.002171 ms
embedding_fp16x8_packed        mean time: 0.002110 ms
####################################################################################################
emd_size: 16384, emd_dim: 8, seq_len: 256
torch                          mean time: 0.007333 ms
embedding                      mean time: 0.002155 ms
embedding_fp32x4               mean time: 0.002151 ms
embedding_fp32x4_packed        mean time: 0.002165 ms
torch                          mean time: 0.007303 ms
embedding_half                 mean time: 0.002161 ms
embedding_fp16x8               mean time: 0.002160 ms
embedding_fp16x8_packed        mean time: 0.002157 ms
####################################################################################################
emd_size: 16384, emd_dim: 8, seq_len: 1024
torch                          mean time: 0.007387 ms
embedding                      mean time: 0.002221 ms
embedding_fp32x4               mean time: 0.002333 ms
embedding_fp32x4_packed        mean time: 0.002235 ms
torch                          mean time: 0.007307 ms
embedding_half                 mean time: 0.002220 ms
embedding_fp16x8               mean time: 0.002457 ms
embedding_fp16x8_packed        mean time: 0.002224 ms
####################################################################################################
emd_size: 16384, emd_dim: 8, seq_len: 4096
torch                          mean time: 0.007323 ms
embedding                      mean time: 0.003884 ms
embedding_fp32x4               mean time: 0.003968 ms
embedding_fp32x4_packed        mean time: 0.003902 ms
torch                          mean time: 0.007296 ms
embedding_half                 mean time: 0.003883 ms
embedding_fp16x8               mean time: 0.004014 ms
embedding_fp16x8_packed        mean time: 0.003891 ms
####################################################################################################
emd_size: 16384, emd_dim: 32, seq_len: 1
torch                          mean time: 0.005523 ms
embedding                      mean time: 0.002096 ms
embedding_fp32x4               mean time: 0.002113 ms
embedding_fp32x4_packed        mean time: 0.002113 ms
torch                          mean time: 0.005517 ms
embedding_half                 mean time: 0.002083 ms
embedding_fp16x8               mean time: 0.002133 ms
embedding_fp16x8_packed        mean time: 0.002096 ms
####################################################################################################
emd_size: 16384, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007332 ms
embedding                      mean time: 0.002128 ms
embedding_fp32x4               mean time: 0.002135 ms
embedding_fp32x4_packed        mean time: 0.002129 ms
torch                          mean time: 0.007315 ms
embedding_half                 mean time: 0.002111 ms
embedding_fp16x8               mean time: 0.002167 ms
embedding_fp16x8_packed        mean time: 0.002116 ms
####################################################################################################
emd_size: 16384, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007363 ms
embedding                      mean time: 0.002147 ms
embedding_fp32x4               mean time: 0.002133 ms
embedding_fp32x4_packed        mean time: 0.002173 ms
torch                          mean time: 0.007293 ms
embedding_half                 mean time: 0.002146 ms
embedding_fp16x8               mean time: 0.002154 ms
embedding_fp16x8_packed        mean time: 0.002143 ms
####################################################################################################
emd_size: 16384, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.007278 ms
embedding                      mean time: 0.002236 ms
embedding_fp32x4               mean time: 0.002329 ms
embedding_fp32x4_packed        mean time: 0.002231 ms
torch                          mean time: 0.007298 ms
embedding_half                 mean time: 0.002226 ms
embedding_fp16x8               mean time: 0.002436 ms
embedding_fp16x8_packed        mean time: 0.002235 ms
####################################################################################################
emd_size: 16384, emd_dim: 32, seq_len: 4096
torch                          mean time: 0.007284 ms
embedding                      mean time: 0.003892 ms
embedding_fp32x4               mean time: 0.003949 ms
embedding_fp32x4_packed        mean time: 0.003904 ms
torch                          mean time: 0.007222 ms
embedding_half                 mean time: 0.003885 ms
embedding_fp16x8               mean time: 0.004017 ms
embedding_fp16x8_packed        mean time: 0.003889 ms
####################################################################################################
emd_size: 16384, emd_dim: 128, seq_len: 1
torch                          mean time: 0.005543 ms
embedding                      mean time: 0.002097 ms
embedding_fp32x4               mean time: 0.002115 ms
embedding_fp32x4_packed        mean time: 0.002115 ms
torch                          mean time: 0.005527 ms
embedding_half                 mean time: 0.002090 ms
embedding_fp16x8               mean time: 0.002136 ms
embedding_fp16x8_packed        mean time: 0.002103 ms
####################################################################################################
emd_size: 16384, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007369 ms
embedding                      mean time: 0.002129 ms
embedding_fp32x4               mean time: 0.002139 ms
embedding_fp32x4_packed        mean time: 0.002134 ms
torch                          mean time: 0.007303 ms
embedding_half                 mean time: 0.002129 ms
embedding_fp16x8               mean time: 0.002181 ms
embedding_fp16x8_packed        mean time: 0.002120 ms
####################################################################################################
emd_size: 16384, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007405 ms
embedding                      mean time: 0.002142 ms
embedding_fp32x4               mean time: 0.002131 ms
embedding_fp32x4_packed        mean time: 0.002172 ms
torch                          mean time: 0.007350 ms
embedding_half                 mean time: 0.002132 ms
embedding_fp16x8               mean time: 0.002139 ms
embedding_fp16x8_packed        mean time: 0.002151 ms
####################################################################################################
emd_size: 16384, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.007326 ms
embedding                      mean time: 0.002202 ms
embedding_fp32x4               mean time: 0.002554 ms
embedding_fp32x4_packed        mean time: 0.002272 ms
torch                          mean time: 0.007278 ms
embedding_half                 mean time: 0.002183 ms
embedding_fp16x8               mean time: 0.002710 ms
embedding_fp16x8_packed        mean time: 0.002251 ms
####################################################################################################
emd_size: 16384, emd_dim: 128, seq_len: 4096
torch                          mean time: 0.007349 ms
embedding                      mean time: 0.003942 ms
embedding_fp32x4               mean time: 0.004279 ms
embedding_fp32x4_packed        mean time: 0.003868 ms
torch                          mean time: 0.007314 ms
embedding_half                 mean time: 0.003898 ms
embedding_fp16x8               mean time: 0.005210 ms
embedding_fp16x8_packed        mean time: 0.003877 ms
####################################################################################################
emd_size: 16384, emd_dim: 512, seq_len: 1
torch                          mean time: 0.005563 ms
embedding                      mean time: 0.002111 ms
embedding_fp32x4               mean time: 0.002125 ms
embedding_fp32x4_packed        mean time: 0.002124 ms
torch                          mean time: 0.005557 ms
embedding_half                 mean time: 0.002089 ms
embedding_fp16x8               mean time: 0.002141 ms
embedding_fp16x8_packed        mean time: 0.002099 ms
####################################################################################################
emd_size: 16384, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007338 ms
embedding                      mean time: 0.002130 ms
embedding_fp32x4               mean time: 0.002151 ms
embedding_fp32x4_packed        mean time: 0.002157 ms
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.002110 ms
embedding_fp16x8               mean time: 0.002175 ms
embedding_fp16x8_packed        mean time: 0.002122 ms
####################################################################################################
emd_size: 16384, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007365 ms
embedding                      mean time: 0.002147 ms
embedding_fp32x4               mean time: 0.002467 ms
embedding_fp32x4_packed        mean time: 0.002191 ms
torch                          mean time: 0.007339 ms
embedding_half                 mean time: 0.002148 ms
embedding_fp16x8               mean time: 0.002560 ms
embedding_fp16x8_packed        mean time: 0.002167 ms
####################################################################################################
emd_size: 16384, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.007374 ms
embedding                      mean time: 0.003282 ms
embedding_fp32x4               mean time: 0.005516 ms
embedding_fp32x4_packed        mean time: 0.002573 ms
torch                          mean time: 0.007238 ms
embedding_half                 mean time: 0.003229 ms
embedding_fp16x8               mean time: 0.006021 ms
embedding_fp16x8_packed        mean time: 0.002322 ms
####################################################################################################
emd_size: 16384, emd_dim: 512, seq_len: 4096
torch                          mean time: 0.007337 ms
embedding                      mean time: 0.008203 ms
embedding_fp32x4               mean time: 0.016742 ms
embedding_fp32x4_packed        mean time: 0.005046 ms
torch                          mean time: 0.007278 ms
embedding_half                 mean time: 0.007902 ms
embedding_fp16x8               mean time: 0.017082 ms
embedding_fp16x8_packed        mean time: 0.003861 ms
####################################################################################################
emd_size: 16384, emd_dim: 1024, seq_len: 1
torch                          mean time: 0.005538 ms
embedding                      mean time: 0.002121 ms
embedding_fp32x4               mean time: 0.002139 ms
embedding_fp32x4_packed        mean time: 0.002141 ms
torch                          mean time: 0.005552 ms
embedding_half                 mean time: 0.002132 ms
embedding_fp16x8               mean time: 0.002136 ms
embedding_fp16x8_packed        mean time: 0.002141 ms
####################################################################################################
emd_size: 16384, emd_dim: 1024, seq_len: 64
torch                          mean time: 0.007334 ms
embedding                      mean time: 0.002147 ms
embedding_fp32x4               mean time: 0.002112 ms
embedding_fp32x4_packed        mean time: 0.002200 ms
torch                          mean time: 0.007306 ms
embedding_half                 mean time: 0.002149 ms
embedding_fp16x8               mean time: 0.002215 ms
embedding_fp16x8_packed        mean time: 0.002159 ms
####################################################################################################
emd_size: 16384, emd_dim: 1024, seq_len: 256
torch                          mean time: 0.007309 ms
embedding                      mean time: 0.002174 ms
embedding_fp32x4               mean time: 0.003151 ms
embedding_fp32x4_packed        mean time: 0.002177 ms
torch                          mean time: 0.007283 ms
embedding_half                 mean time: 0.002099 ms
embedding_fp16x8               mean time: 0.003543 ms
embedding_fp16x8_packed        mean time: 0.002116 ms
####################################################################################################
emd_size: 16384, emd_dim: 1024, seq_len: 1024
torch                          mean time: 0.007362 ms
embedding                      mean time: 0.004063 ms
embedding_fp32x4               mean time: 0.009474 ms
embedding_fp32x4_packed        mean time: 0.003475 ms
torch                          mean time: 0.007294 ms
embedding_half                 mean time: 0.003820 ms
embedding_fp16x8               mean time: 0.009964 ms
embedding_fp16x8_packed        mean time: 0.002714 ms
####################################################################################################
emd_size: 16384, emd_dim: 1024, seq_len: 4096
torch                          mean time: 0.008436 ms
embedding                      mean time: 0.010999 ms
embedding_fp32x4               mean time: 0.032482 ms
embedding_fp32x4_packed        mean time: 0.008358 ms
torch                          mean time: 0.007291 ms
embedding_half                 mean time: 0.010069 ms
embedding_fp16x8               mean time: 0.032966 ms
embedding_fp16x8_packed        mean time: 0.005146 ms
####################################################################################################
emd_size: 16384, emd_dim: 2048, seq_len: 1
torch                          mean time: 0.005531 ms
embedding                      mean time: 0.002089 ms
embedding_fp32x4               mean time: 0.002369 ms
embedding_fp32x4_packed        mean time: 0.002099 ms
torch                          mean time: 0.005513 ms
embedding_half                 mean time: 0.002063 ms
embedding_fp16x8               mean time: 0.002642 ms
embedding_fp16x8_packed        mean time: 0.002071 ms
####################################################################################################
emd_size: 16384, emd_dim: 2048, seq_len: 64
torch                          mean time: 0.007315 ms
embedding                      mean time: 0.002199 ms
embedding_fp32x4               mean time: 0.002639 ms
embedding_fp32x4_packed        mean time: 0.002257 ms
torch                          mean time: 0.007273 ms
embedding_half                 mean time: 0.002148 ms
embedding_fp16x8               mean time: 0.002986 ms
embedding_fp16x8_packed        mean time: 0.002219 ms
####################################################################################################
emd_size: 16384, emd_dim: 2048, seq_len: 256
torch                          mean time: 0.007328 ms
embedding                      mean time: 0.002667 ms
embedding_fp32x4               mean time: 0.004676 ms
embedding_fp32x4_packed        mean time: 0.002665 ms
torch                          mean time: 0.007297 ms
embedding_half                 mean time: 0.002311 ms
embedding_fp16x8               mean time: 0.005506 ms
embedding_fp16x8_packed        mean time: 0.002364 ms
####################################################################################################
emd_size: 16384, emd_dim: 2048, seq_len: 1024
torch                          mean time: 0.007320 ms
embedding                      mean time: 0.005881 ms
embedding_fp32x4               mean time: 0.018549 ms
embedding_fp32x4_packed        mean time: 0.005207 ms
torch                          mean time: 0.007315 ms
embedding_half                 mean time: 0.005007 ms
embedding_fp16x8               mean time: 0.018323 ms
embedding_fp16x8_packed        mean time: 0.003624 ms
####################################################################################################
emd_size: 16384, emd_dim: 2048, seq_len: 4096
torch                          mean time: 0.014959 ms
embedding                      mean time: 0.017337 ms
embedding_fp32x4               mean time: 0.065351 ms
embedding_fp32x4_packed        mean time: 0.015193 ms
torch                          mean time: 0.008443 ms
embedding_half                 mean time: 0.014501 ms
embedding_fp16x8               mean time: 0.065950 ms
embedding_fp16x8_packed        mean time: 0.008563 ms
####################################################################################################
emd_size: 16384, emd_dim: 4096, seq_len: 1
torch                          mean time: 0.005504 ms
embedding                      mean time: 0.002511 ms
embedding_fp32x4               mean time: 0.003253 ms
embedding_fp32x4_packed        mean time: 0.002665 ms
torch                          mean time: 0.005519 ms
embedding_half                 mean time: 0.002485 ms
embedding_fp16x8               mean time: 0.003817 ms
embedding_fp16x8_packed        mean time: 0.002592 ms
####################################################################################################
emd_size: 16384, emd_dim: 4096, seq_len: 64
torch                          mean time: 0.007316 ms
embedding                      mean time: 0.002798 ms
embedding_fp32x4               mean time: 0.003824 ms
embedding_fp32x4_packed        mean time: 0.002918 ms
torch                          mean time: 0.007265 ms
embedding_half                 mean time: 0.002680 ms
embedding_fp16x8               mean time: 0.004303 ms
embedding_fp16x8_packed        mean time: 0.002804 ms
####################################################################################################
emd_size: 16384, emd_dim: 4096, seq_len: 256
torch                          mean time: 0.007369 ms
embedding                      mean time: 0.003756 ms
embedding_fp32x4               mean time: 0.007318 ms
embedding_fp32x4_packed        mean time: 0.003599 ms
torch                          mean time: 0.007257 ms
embedding_half                 mean time: 0.002969 ms
embedding_fp16x8               mean time: 0.009413 ms
embedding_fp16x8_packed        mean time: 0.003068 ms
####################################################################################################
emd_size: 16384, emd_dim: 4096, seq_len: 1024
torch                          mean time: 0.008574 ms
embedding                      mean time: 0.009442 ms
embedding_fp32x4               mean time: 0.035094 ms
embedding_fp32x4_packed        mean time: 0.009118 ms
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.007456 ms
embedding_fp16x8               mean time: 0.036843 ms
embedding_fp16x8_packed        mean time: 0.005329 ms
####################################################################################################
emd_size: 16384, emd_dim: 4096, seq_len: 4096
torch                          mean time: 0.154803 ms
embedding                      mean time: 0.161634 ms
embedding_fp32x4               mean time: 0.174209 ms
embedding_fp32x4_packed        mean time: 0.156330 ms
torch                          mean time: 0.014941 ms
embedding_half                 mean time: 0.023512 ms
embedding_fp16x8               mean time: 0.135364 ms
embedding_fp16x8_packed        mean time: 0.015608 ms
####################################################################################################
emd_size: 10240, emd_dim: 8, seq_len: 1
torch                          mean time: 0.005541 ms
embedding                      mean time: 0.002075 ms
embedding_fp32x4               mean time: 0.002090 ms
embedding_fp32x4_packed        mean time: 0.002087 ms
torch                          mean time: 0.005514 ms
embedding_half                 mean time: 0.002076 ms
embedding_fp16x8               mean time: 0.002120 ms
embedding_fp16x8_packed        mean time: 0.002090 ms
####################################################################################################
emd_size: 10240, emd_dim: 8, seq_len: 64
torch                          mean time: 0.007287 ms
embedding                      mean time: 0.002115 ms
embedding_fp32x4               mean time: 0.002120 ms
embedding_fp32x4_packed        mean time: 0.002123 ms
torch                          mean time: 0.007292 ms
embedding_half                 mean time: 0.002109 ms
embedding_fp16x8               mean time: 0.002146 ms
embedding_fp16x8_packed        mean time: 0.002101 ms
####################################################################################################
emd_size: 10240, emd_dim: 8, seq_len: 256
torch                          mean time: 0.007320 ms
embedding                      mean time: 0.002129 ms
embedding_fp32x4               mean time: 0.002101 ms
embedding_fp32x4_packed        mean time: 0.002154 ms
torch                          mean time: 0.007296 ms
embedding_half                 mean time: 0.002111 ms
embedding_fp16x8               mean time: 0.002161 ms
embedding_fp16x8_packed        mean time: 0.002132 ms
####################################################################################################
emd_size: 10240, emd_dim: 8, seq_len: 1024
torch                          mean time: 0.007303 ms
embedding                      mean time: 0.002216 ms
embedding_fp32x4               mean time: 0.002321 ms
embedding_fp32x4_packed        mean time: 0.002226 ms
torch                          mean time: 0.007335 ms
embedding_half                 mean time: 0.002202 ms
embedding_fp16x8               mean time: 0.002431 ms
embedding_fp16x8_packed        mean time: 0.002214 ms
####################################################################################################
emd_size: 10240, emd_dim: 8, seq_len: 4096
torch                          mean time: 0.007297 ms
embedding                      mean time: 0.003879 ms
embedding_fp32x4               mean time: 0.003966 ms
embedding_fp32x4_packed        mean time: 0.003888 ms
torch                          mean time: 0.007288 ms
embedding_half                 mean time: 0.003881 ms
embedding_fp16x8               mean time: 0.004015 ms
embedding_fp16x8_packed        mean time: 0.003881 ms
####################################################################################################
emd_size: 10240, emd_dim: 32, seq_len: 1
torch                          mean time: 0.005507 ms
embedding                      mean time: 0.002083 ms
embedding_fp32x4               mean time: 0.002094 ms
embedding_fp32x4_packed        mean time: 0.002099 ms
torch                          mean time: 0.005498 ms
embedding_half                 mean time: 0.002068 ms
embedding_fp16x8               mean time: 0.002112 ms
embedding_fp16x8_packed        mean time: 0.002076 ms
####################################################################################################
emd_size: 10240, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007320 ms
embedding                      mean time: 0.002111 ms
embedding_fp32x4               mean time: 0.002119 ms
embedding_fp32x4_packed        mean time: 0.002124 ms
torch                          mean time: 0.007310 ms
embedding_half                 mean time: 0.002103 ms
embedding_fp16x8               mean time: 0.002153 ms
embedding_fp16x8_packed        mean time: 0.002112 ms
####################################################################################################
emd_size: 10240, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007286 ms
embedding                      mean time: 0.002142 ms
embedding_fp32x4               mean time: 0.002131 ms
embedding_fp32x4_packed        mean time: 0.002150 ms
torch                          mean time: 0.007268 ms
embedding_half                 mean time: 0.002118 ms
embedding_fp16x8               mean time: 0.002115 ms
embedding_fp16x8_packed        mean time: 0.002141 ms
####################################################################################################
emd_size: 10240, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.007321 ms
embedding                      mean time: 0.002227 ms
embedding_fp32x4               mean time: 0.002312 ms
embedding_fp32x4_packed        mean time: 0.002236 ms
torch                          mean time: 0.007248 ms
embedding_half                 mean time: 0.002214 ms
embedding_fp16x8               mean time: 0.002450 ms
embedding_fp16x8_packed        mean time: 0.002220 ms
####################################################################################################
emd_size: 10240, emd_dim: 32, seq_len: 4096
torch                          mean time: 0.007314 ms
embedding                      mean time: 0.003869 ms
embedding_fp32x4               mean time: 0.003937 ms
embedding_fp32x4_packed        mean time: 0.003879 ms
torch                          mean time: 0.007322 ms
embedding_half                 mean time: 0.003868 ms
embedding_fp16x8               mean time: 0.004004 ms
embedding_fp16x8_packed        mean time: 0.003881 ms
####################################################################################################
emd_size: 10240, emd_dim: 128, seq_len: 1
torch                          mean time: 0.005530 ms
embedding                      mean time: 0.002091 ms
embedding_fp32x4               mean time: 0.002110 ms
embedding_fp32x4_packed        mean time: 0.002101 ms
torch                          mean time: 0.005396 ms
embedding_half                 mean time: 0.002078 ms
embedding_fp16x8               mean time: 0.002127 ms
embedding_fp16x8_packed        mean time: 0.002090 ms
####################################################################################################
emd_size: 10240, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007286 ms
embedding                      mean time: 0.002106 ms
embedding_fp32x4               mean time: 0.002123 ms
embedding_fp32x4_packed        mean time: 0.002119 ms
torch                          mean time: 0.007257 ms
embedding_half                 mean time: 0.002100 ms
embedding_fp16x8               mean time: 0.002148 ms
embedding_fp16x8_packed        mean time: 0.002117 ms
####################################################################################################
emd_size: 10240, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007329 ms
embedding                      mean time: 0.002125 ms
embedding_fp32x4               mean time: 0.002087 ms
embedding_fp32x4_packed        mean time: 0.002163 ms
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.002126 ms
embedding_fp16x8               mean time: 0.002124 ms
embedding_fp16x8_packed        mean time: 0.002141 ms
####################################################################################################
emd_size: 10240, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.007317 ms
embedding                      mean time: 0.002190 ms
embedding_fp32x4               mean time: 0.002573 ms
embedding_fp32x4_packed        mean time: 0.002276 ms
torch                          mean time: 0.007308 ms
embedding_half                 mean time: 0.002171 ms
embedding_fp16x8               mean time: 0.002690 ms
embedding_fp16x8_packed        mean time: 0.002233 ms
####################################################################################################
emd_size: 10240, emd_dim: 128, seq_len: 4096
torch                          mean time: 0.007291 ms
embedding                      mean time: 0.003932 ms
embedding_fp32x4               mean time: 0.004277 ms
embedding_fp32x4_packed        mean time: 0.003862 ms
torch                          mean time: 0.007242 ms
embedding_half                 mean time: 0.003889 ms
embedding_fp16x8               mean time: 0.005054 ms
embedding_fp16x8_packed        mean time: 0.003865 ms
####################################################################################################
emd_size: 10240, emd_dim: 512, seq_len: 1
torch                          mean time: 0.005524 ms
embedding                      mean time: 0.002072 ms
embedding_fp32x4               mean time: 0.002102 ms
embedding_fp32x4_packed        mean time: 0.002092 ms
torch                          mean time: 0.005516 ms
embedding_half                 mean time: 0.002072 ms
embedding_fp16x8               mean time: 0.002141 ms
embedding_fp16x8_packed        mean time: 0.002100 ms
####################################################################################################
emd_size: 10240, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007310 ms
embedding                      mean time: 0.002116 ms
embedding_fp32x4               mean time: 0.002141 ms
embedding_fp32x4_packed        mean time: 0.002133 ms
torch                          mean time: 0.007296 ms
embedding_half                 mean time: 0.002100 ms
embedding_fp16x8               mean time: 0.002151 ms
embedding_fp16x8_packed        mean time: 0.002118 ms
####################################################################################################
emd_size: 10240, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007293 ms
embedding                      mean time: 0.002124 ms
embedding_fp32x4               mean time: 0.002459 ms
embedding_fp32x4_packed        mean time: 0.002151 ms
torch                          mean time: 0.007320 ms
embedding_half                 mean time: 0.002129 ms
embedding_fp16x8               mean time: 0.002557 ms
embedding_fp16x8_packed        mean time: 0.002144 ms
####################################################################################################
emd_size: 10240, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.007377 ms
embedding                      mean time: 0.003274 ms
embedding_fp32x4               mean time: 0.005466 ms
embedding_fp32x4_packed        mean time: 0.002610 ms
torch                          mean time: 0.007292 ms
embedding_half                 mean time: 0.003226 ms
embedding_fp16x8               mean time: 0.005661 ms
embedding_fp16x8_packed        mean time: 0.002295 ms
####################################################################################################
emd_size: 10240, emd_dim: 512, seq_len: 4096
torch                          mean time: 0.007333 ms
embedding                      mean time: 0.008201 ms
embedding_fp32x4               mean time: 0.016729 ms
embedding_fp32x4_packed        mean time: 0.005039 ms
torch                          mean time: 0.007288 ms
embedding_half                 mean time: 0.007904 ms
embedding_fp16x8               mean time: 0.017146 ms
embedding_fp16x8_packed        mean time: 0.003858 ms
####################################################################################################
emd_size: 10240, emd_dim: 1024, seq_len: 1
torch                          mean time: 0.005531 ms
embedding                      mean time: 0.002127 ms
embedding_fp32x4               mean time: 0.002124 ms
embedding_fp32x4_packed        mean time: 0.002137 ms
torch                          mean time: 0.005517 ms
embedding_half                 mean time: 0.002105 ms
embedding_fp16x8               mean time: 0.002088 ms
embedding_fp16x8_packed        mean time: 0.002123 ms
####################################################################################################
emd_size: 10240, emd_dim: 1024, seq_len: 64
torch                          mean time: 0.007297 ms
embedding                      mean time: 0.002114 ms
embedding_fp32x4               mean time: 0.002091 ms
embedding_fp32x4_packed        mean time: 0.002145 ms
torch                          mean time: 0.007336 ms
embedding_half                 mean time: 0.002136 ms
embedding_fp16x8               mean time: 0.002197 ms
embedding_fp16x8_packed        mean time: 0.002155 ms
####################################################################################################
emd_size: 10240, emd_dim: 1024, seq_len: 256
torch                          mean time: 0.007378 ms
embedding                      mean time: 0.002174 ms
embedding_fp32x4               mean time: 0.003144 ms
embedding_fp32x4_packed        mean time: 0.002182 ms
torch                          mean time: 0.007322 ms
embedding_half                 mean time: 0.002101 ms
embedding_fp16x8               mean time: 0.003614 ms
embedding_fp16x8_packed        mean time: 0.002121 ms
####################################################################################################
emd_size: 10240, emd_dim: 1024, seq_len: 1024
torch                          mean time: 0.007372 ms
embedding                      mean time: 0.004071 ms
embedding_fp32x4               mean time: 0.009458 ms
embedding_fp32x4_packed        mean time: 0.003467 ms
torch                          mean time: 0.007362 ms
embedding_half                 mean time: 0.003814 ms
embedding_fp16x8               mean time: 0.009940 ms
embedding_fp16x8_packed        mean time: 0.002799 ms
####################################################################################################
emd_size: 10240, emd_dim: 1024, seq_len: 4096
torch                          mean time: 0.008427 ms
embedding                      mean time: 0.010980 ms
embedding_fp32x4               mean time: 0.032485 ms
embedding_fp32x4_packed        mean time: 0.008358 ms
torch                          mean time: 0.007327 ms
embedding_half                 mean time: 0.010053 ms
embedding_fp16x8               mean time: 0.032999 ms
embedding_fp16x8_packed        mean time: 0.005144 ms
####################################################################################################
emd_size: 10240, emd_dim: 2048, seq_len: 1
torch                          mean time: 0.005542 ms
embedding                      mean time: 0.002115 ms
embedding_fp32x4               mean time: 0.002374 ms
embedding_fp32x4_packed        mean time: 0.002125 ms
torch                          mean time: 0.005560 ms
embedding_half                 mean time: 0.002081 ms
embedding_fp16x8               mean time: 0.002648 ms
embedding_fp16x8_packed        mean time: 0.002103 ms
####################################################################################################
emd_size: 10240, emd_dim: 2048, seq_len: 64
torch                          mean time: 0.007322 ms
embedding                      mean time: 0.002193 ms
embedding_fp32x4               mean time: 0.002620 ms
embedding_fp32x4_packed        mean time: 0.002253 ms
torch                          mean time: 0.007343 ms
embedding_half                 mean time: 0.002116 ms
embedding_fp16x8               mean time: 0.002950 ms
embedding_fp16x8_packed        mean time: 0.002185 ms
####################################################################################################
emd_size: 10240, emd_dim: 2048, seq_len: 256
torch                          mean time: 0.007393 ms
embedding                      mean time: 0.002669 ms
embedding_fp32x4               mean time: 0.004688 ms
embedding_fp32x4_packed        mean time: 0.002646 ms
torch                          mean time: 0.007340 ms
embedding_half                 mean time: 0.002318 ms
embedding_fp16x8               mean time: 0.005356 ms
embedding_fp16x8_packed        mean time: 0.002359 ms
####################################################################################################
emd_size: 10240, emd_dim: 2048, seq_len: 1024
torch                          mean time: 0.007364 ms
embedding                      mean time: 0.005862 ms
embedding_fp32x4               mean time: 0.018603 ms
embedding_fp32x4_packed        mean time: 0.005209 ms
torch                          mean time: 0.007327 ms
embedding_half                 mean time: 0.005005 ms
embedding_fp16x8               mean time: 0.018943 ms
embedding_fp16x8_packed        mean time: 0.003589 ms
####################################################################################################
emd_size: 10240, emd_dim: 2048, seq_len: 4096
torch                          mean time: 0.014934 ms
embedding                      mean time: 0.017178 ms
embedding_fp32x4               mean time: 0.065522 ms
embedding_fp32x4_packed        mean time: 0.015120 ms
torch                          mean time: 0.008420 ms
embedding_half                 mean time: 0.014487 ms
embedding_fp16x8               mean time: 0.065873 ms
embedding_fp16x8_packed        mean time: 0.008531 ms
####################################################################################################
emd_size: 10240, emd_dim: 4096, seq_len: 1
torch                          mean time: 0.005562 ms
embedding                      mean time: 0.002505 ms
embedding_fp32x4               mean time: 0.003251 ms
embedding_fp32x4_packed        mean time: 0.002654 ms
torch                          mean time: 0.005535 ms
embedding_half                 mean time: 0.002492 ms
embedding_fp16x8               mean time: 0.003811 ms
embedding_fp16x8_packed        mean time: 0.002597 ms
####################################################################################################
emd_size: 10240, emd_dim: 4096, seq_len: 64
torch                          mean time: 0.007352 ms
embedding                      mean time: 0.002789 ms
embedding_fp32x4               mean time: 0.003817 ms
embedding_fp32x4_packed        mean time: 0.002932 ms
torch                          mean time: 0.007295 ms
embedding_half                 mean time: 0.002686 ms
embedding_fp16x8               mean time: 0.004395 ms
embedding_fp16x8_packed        mean time: 0.002816 ms
####################################################################################################
emd_size: 10240, emd_dim: 4096, seq_len: 256
torch                          mean time: 0.007368 ms
embedding                      mean time: 0.003683 ms
embedding_fp32x4               mean time: 0.007163 ms
embedding_fp32x4_packed        mean time: 0.003579 ms
torch                          mean time: 0.007301 ms
embedding_half                 mean time: 0.002981 ms
embedding_fp16x8               mean time: 0.009635 ms
embedding_fp16x8_packed        mean time: 0.003067 ms
####################################################################################################
emd_size: 10240, emd_dim: 4096, seq_len: 1024
torch                          mean time: 0.008543 ms
embedding                      mean time: 0.009442 ms
embedding_fp32x4               mean time: 0.034997 ms
embedding_fp32x4_packed        mean time: 0.008959 ms
torch                          mean time: 0.007344 ms
embedding_half                 mean time: 0.007409 ms
embedding_fp16x8               mean time: 0.036779 ms
embedding_fp16x8_packed        mean time: 0.005412 ms
####################################################################################################
emd_size: 10240, emd_dim: 4096, seq_len: 4096
torch                          mean time: 0.144760 ms
embedding                      mean time: 0.150352 ms
embedding_fp32x4               mean time: 0.163800 ms
embedding_fp32x4_packed        mean time: 0.145120 ms
torch                          mean time: 0.014945 ms
embedding_half                 mean time: 0.023509 ms
embedding_fp16x8               mean time: 0.135245 ms
embedding_fp16x8_packed        mean time: 0.015645 ms
####################################################################################################
emd_size: 102400, emd_dim: 8, seq_len: 1
torch                          mean time: 0.005528 ms
embedding                      mean time: 0.002090 ms
embedding_fp32x4               mean time: 0.002110 ms
embedding_fp32x4_packed        mean time: 0.002112 ms
torch                          mean time: 0.005514 ms
embedding_half                 mean time: 0.002088 ms
embedding_fp16x8               mean time: 0.002131 ms
embedding_fp16x8_packed        mean time: 0.002106 ms
####################################################################################################
emd_size: 102400, emd_dim: 8, seq_len: 64
torch                          mean time: 0.007373 ms
embedding                      mean time: 0.002125 ms
embedding_fp32x4               mean time: 0.002138 ms
embedding_fp32x4_packed        mean time: 0.002142 ms
torch                          mean time: 0.007243 ms
embedding_half                 mean time: 0.002114 ms
embedding_fp16x8               mean time: 0.002167 ms
embedding_fp16x8_packed        mean time: 0.002120 ms
####################################################################################################
emd_size: 102400, emd_dim: 8, seq_len: 256
torch                          mean time: 0.007473 ms
embedding                      mean time: 0.002178 ms
embedding_fp32x4               mean time: 0.002208 ms
embedding_fp32x4_packed        mean time: 0.002176 ms
torch                          mean time: 0.007333 ms
embedding_half                 mean time: 0.002154 ms
embedding_fp16x8               mean time: 0.002182 ms
embedding_fp16x8_packed        mean time: 0.002161 ms
####################################################################################################
emd_size: 102400, emd_dim: 8, seq_len: 1024
torch                          mean time: 0.007391 ms
embedding                      mean time: 0.002232 ms
embedding_fp32x4               mean time: 0.002346 ms
embedding_fp32x4_packed        mean time: 0.002257 ms
torch                          mean time: 0.007308 ms
embedding_half                 mean time: 0.002228 ms
embedding_fp16x8               mean time: 0.002477 ms
embedding_fp16x8_packed        mean time: 0.002234 ms
####################################################################################################
emd_size: 102400, emd_dim: 8, seq_len: 4096
torch                          mean time: 0.007345 ms
embedding                      mean time: 0.003873 ms
embedding_fp32x4               mean time: 0.003954 ms
embedding_fp32x4_packed        mean time: 0.003875 ms
torch                          mean time: 0.007325 ms
embedding_half                 mean time: 0.003866 ms
embedding_fp16x8               mean time: 0.004057 ms
embedding_fp16x8_packed        mean time: 0.003876 ms
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 1
torch                          mean time: 0.005568 ms
embedding                      mean time: 0.002085 ms
embedding_fp32x4               mean time: 0.002100 ms
embedding_fp32x4_packed        mean time: 0.002104 ms
torch                          mean time: 0.005507 ms
embedding_half                 mean time: 0.002094 ms
embedding_fp16x8               mean time: 0.002151 ms
embedding_fp16x8_packed        mean time: 0.002108 ms
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007355 ms
embedding                      mean time: 0.002124 ms
embedding_fp32x4               mean time: 0.002138 ms
embedding_fp32x4_packed        mean time: 0.002140 ms
torch                          mean time: 0.007315 ms
embedding_half                 mean time: 0.002120 ms
embedding_fp16x8               mean time: 0.002157 ms
embedding_fp16x8_packed        mean time: 0.002122 ms
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007378 ms
embedding                      mean time: 0.002157 ms
embedding_fp32x4               mean time: 0.002152 ms
embedding_fp32x4_packed        mean time: 0.002201 ms
torch                          mean time: 0.007318 ms
embedding_half                 mean time: 0.002166 ms
embedding_fp16x8               mean time: 0.002173 ms
embedding_fp16x8_packed        mean time: 0.002166 ms
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.007357 ms
embedding                      mean time: 0.002223 ms
embedding_fp32x4               mean time: 0.002316 ms
embedding_fp32x4_packed        mean time: 0.002236 ms
torch                          mean time: 0.007298 ms
embedding_half                 mean time: 0.002223 ms
embedding_fp16x8               mean time: 0.002453 ms
embedding_fp16x8_packed        mean time: 0.002234 ms
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 4096
torch                          mean time: 0.007333 ms
embedding                      mean time: 0.003864 ms
embedding_fp32x4               mean time: 0.003929 ms
embedding_fp32x4_packed        mean time: 0.003872 ms
torch                          mean time: 0.007282 ms
embedding_half                 mean time: 0.003864 ms
embedding_fp16x8               mean time: 0.004007 ms
embedding_fp16x8_packed        mean time: 0.003875 ms
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 1
torch                          mean time: 0.005530 ms
embedding                      mean time: 0.002104 ms
embedding_fp32x4               mean time: 0.002126 ms
embedding_fp32x4_packed        mean time: 0.002125 ms
torch                          mean time: 0.005546 ms
embedding_half                 mean time: 0.002105 ms
embedding_fp16x8               mean time: 0.002144 ms
embedding_fp16x8_packed        mean time: 0.002106 ms
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007330 ms
embedding                      mean time: 0.002142 ms
embedding_fp32x4               mean time: 0.002143 ms
embedding_fp32x4_packed        mean time: 0.002142 ms
torch                          mean time: 0.007313 ms
embedding_half                 mean time: 0.002115 ms
embedding_fp16x8               mean time: 0.002163 ms
embedding_fp16x8_packed        mean time: 0.002122 ms
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007346 ms
embedding                      mean time: 0.002149 ms
embedding_fp32x4               mean time: 0.002130 ms
embedding_fp32x4_packed        mean time: 0.002180 ms
torch                          mean time: 0.007268 ms
embedding_half                 mean time: 0.002129 ms
embedding_fp16x8               mean time: 0.002167 ms
embedding_fp16x8_packed        mean time: 0.002164 ms
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.007358 ms
embedding                      mean time: 0.002203 ms
embedding_fp32x4               mean time: 0.002531 ms
embedding_fp32x4_packed        mean time: 0.002274 ms
torch                          mean time: 0.007305 ms
embedding_half                 mean time: 0.002169 ms
embedding_fp16x8               mean time: 0.002696 ms
embedding_fp16x8_packed        mean time: 0.002237 ms
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 4096
torch                          mean time: 0.007343 ms
embedding                      mean time: 0.003960 ms
embedding_fp32x4               mean time: 0.004286 ms
embedding_fp32x4_packed        mean time: 0.003861 ms
torch                          mean time: 0.007307 ms
embedding_half                 mean time: 0.003911 ms
embedding_fp16x8               mean time: 0.005122 ms
embedding_fp16x8_packed        mean time: 0.003876 ms
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 1
torch                          mean time: 0.005583 ms
embedding                      mean time: 0.002110 ms
embedding_fp32x4               mean time: 0.002125 ms
embedding_fp32x4_packed        mean time: 0.002114 ms
torch                          mean time: 0.005545 ms
embedding_half                 mean time: 0.002103 ms
embedding_fp16x8               mean time: 0.002145 ms
embedding_fp16x8_packed        mean time: 0.002098 ms
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007342 ms
embedding                      mean time: 0.002136 ms
embedding_fp32x4               mean time: 0.002157 ms
embedding_fp32x4_packed        mean time: 0.002151 ms
torch                          mean time: 0.007338 ms
embedding_half                 mean time: 0.002126 ms
embedding_fp16x8               mean time: 0.002158 ms
embedding_fp16x8_packed        mean time: 0.002133 ms
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007314 ms
embedding                      mean time: 0.002118 ms
embedding_fp32x4               mean time: 0.002416 ms
embedding_fp32x4_packed        mean time: 0.002166 ms
torch                          mean time: 0.007318 ms
embedding_half                 mean time: 0.002135 ms
embedding_fp16x8               mean time: 0.002532 ms
embedding_fp16x8_packed        mean time: 0.002165 ms
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.007376 ms
embedding                      mean time: 0.003300 ms
embedding_fp32x4               mean time: 0.005724 ms
embedding_fp32x4_packed        mean time: 0.002639 ms
torch                          mean time: 0.007321 ms
embedding_half                 mean time: 0.003229 ms
embedding_fp16x8               mean time: 0.005936 ms
embedding_fp16x8_packed        mean time: 0.002329 ms
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 4096
torch                          mean time: 0.007385 ms
embedding                      mean time: 0.008203 ms
embedding_fp32x4               mean time: 0.016742 ms
embedding_fp32x4_packed        mean time: 0.005046 ms
torch                          mean time: 0.007294 ms
embedding_half                 mean time: 0.007901 ms
embedding_fp16x8               mean time: 0.017061 ms
embedding_fp16x8_packed        mean time: 0.003865 ms
####################################################################################################
emd_size: 102400, emd_dim: 1024, seq_len: 1
torch                          mean time: 0.005558 ms
embedding                      mean time: 0.002128 ms
embedding_fp32x4               mean time: 0.002182 ms
embedding_fp32x4_packed        mean time: 0.002150 ms
torch                          mean time: 0.005544 ms
embedding_half                 mean time: 0.002117 ms
embedding_fp16x8               mean time: 0.002148 ms
embedding_fp16x8_packed        mean time: 0.002130 ms
####################################################################################################
emd_size: 102400, emd_dim: 1024, seq_len: 64
torch                          mean time: 0.007341 ms
embedding                      mean time: 0.002135 ms
embedding_fp32x4               mean time: 0.002102 ms
embedding_fp32x4_packed        mean time: 0.002180 ms
torch                          mean time: 0.007330 ms
embedding_half                 mean time: 0.002143 ms
embedding_fp16x8               mean time: 0.002238 ms
embedding_fp16x8_packed        mean time: 0.002148 ms
####################################################################################################
emd_size: 102400, emd_dim: 1024, seq_len: 256
torch                          mean time: 0.007354 ms
embedding                      mean time: 0.002174 ms
embedding_fp32x4               mean time: 0.003199 ms
embedding_fp32x4_packed        mean time: 0.002166 ms
torch                          mean time: 0.007322 ms
embedding_half                 mean time: 0.002106 ms
embedding_fp16x8               mean time: 0.003561 ms
embedding_fp16x8_packed        mean time: 0.002113 ms
####################################################################################################
emd_size: 102400, emd_dim: 1024, seq_len: 1024
torch                          mean time: 0.007410 ms
embedding                      mean time: 0.004098 ms
embedding_fp32x4               mean time: 0.009514 ms
embedding_fp32x4_packed        mean time: 0.003491 ms
torch                          mean time: 0.007370 ms
embedding_half                 mean time: 0.003824 ms
embedding_fp16x8               mean time: 0.009883 ms
embedding_fp16x8_packed        mean time: 0.002779 ms
####################################################################################################
emd_size: 102400, emd_dim: 1024, seq_len: 4096
torch                          mean time: 0.008451 ms
embedding                      mean time: 0.011005 ms
embedding_fp32x4               mean time: 0.032443 ms
embedding_fp32x4_packed        mean time: 0.008337 ms
torch                          mean time: 0.007377 ms
embedding_half                 mean time: 0.010079 ms
embedding_fp16x8               mean time: 0.033050 ms
embedding_fp16x8_packed        mean time: 0.005169 ms
####################################################################################################
emd_size: 102400, emd_dim: 2048, seq_len: 1
torch                          mean time: 0.005569 ms
embedding                      mean time: 0.002109 ms
embedding_fp32x4               mean time: 0.002369 ms
embedding_fp32x4_packed        mean time: 0.002130 ms
torch                          mean time: 0.005548 ms
embedding_half                 mean time: 0.002090 ms
embedding_fp16x8               mean time: 0.002625 ms
embedding_fp16x8_packed        mean time: 0.002115 ms
####################################################################################################
emd_size: 102400, emd_dim: 2048, seq_len: 64
torch                          mean time: 0.007330 ms
embedding                      mean time: 0.002196 ms
embedding_fp32x4               mean time: 0.002629 ms
embedding_fp32x4_packed        mean time: 0.002272 ms
torch                          mean time: 0.007282 ms
embedding_half                 mean time: 0.002141 ms
embedding_fp16x8               mean time: 0.002887 ms
embedding_fp16x8_packed        mean time: 0.002198 ms
####################################################################################################
emd_size: 102400, emd_dim: 2048, seq_len: 256
torch                          mean time: 0.007363 ms
embedding                      mean time: 0.002641 ms
embedding_fp32x4               mean time: 0.004659 ms
embedding_fp32x4_packed        mean time: 0.002642 ms
torch                          mean time: 0.007318 ms
embedding_half                 mean time: 0.002321 ms
embedding_fp16x8               mean time: 0.005377 ms
embedding_fp16x8_packed        mean time: 0.002362 ms
####################################################################################################
emd_size: 102400, emd_dim: 2048, seq_len: 1024
torch                          mean time: 0.007354 ms
embedding                      mean time: 0.005899 ms
embedding_fp32x4               mean time: 0.018481 ms
embedding_fp32x4_packed        mean time: 0.005215 ms
torch                          mean time: 0.007350 ms
embedding_half                 mean time: 0.005003 ms
embedding_fp16x8               mean time: 0.018194 ms
embedding_fp16x8_packed        mean time: 0.003583 ms
####################################################################################################
emd_size: 102400, emd_dim: 2048, seq_len: 4096
torch                          mean time: 0.014979 ms
embedding                      mean time: 0.017408 ms
embedding_fp32x4               mean time: 0.065329 ms
embedding_fp32x4_packed        mean time: 0.015269 ms
torch                          mean time: 0.008436 ms
embedding_half                 mean time: 0.014494 ms
embedding_fp16x8               mean time: 0.065899 ms
embedding_fp16x8_packed        mean time: 0.008574 ms
####################################################################################################
emd_size: 102400, emd_dim: 4096, seq_len: 1
torch                          mean time: 0.005559 ms
embedding                      mean time: 0.002513 ms
embedding_fp32x4               mean time: 0.003252 ms
embedding_fp32x4_packed        mean time: 0.002660 ms
torch                          mean time: 0.005539 ms
embedding_half                 mean time: 0.002495 ms
embedding_fp16x8               mean time: 0.003823 ms
embedding_fp16x8_packed        mean time: 0.002609 ms
####################################################################################################
emd_size: 102400, emd_dim: 4096, seq_len: 64
torch                          mean time: 0.007369 ms
embedding                      mean time: 0.002785 ms
embedding_fp32x4               mean time: 0.003768 ms
embedding_fp32x4_packed        mean time: 0.002927 ms
torch                          mean time: 0.007314 ms
embedding_half                 mean time: 0.002684 ms
embedding_fp16x8               mean time: 0.004385 ms
embedding_fp16x8_packed        mean time: 0.002816 ms
####################################################################################################
emd_size: 102400, emd_dim: 4096, seq_len: 256
torch                          mean time: 0.007430 ms
embedding                      mean time: 0.003706 ms
embedding_fp32x4               mean time: 0.007089 ms
embedding_fp32x4_packed        mean time: 0.003579 ms
torch                          mean time: 0.007361 ms
embedding_half                 mean time: 0.002987 ms
embedding_fp16x8               mean time: 0.009436 ms
embedding_fp16x8_packed        mean time: 0.003057 ms
####################################################################################################
emd_size: 102400, emd_dim: 4096, seq_len: 1024
torch                          mean time: 0.008581 ms
embedding                      mean time: 0.009449 ms
embedding_fp32x4               mean time: 0.034888 ms
embedding_fp32x4_packed        mean time: 0.009133 ms
torch                          mean time: 0.007318 ms
embedding_half                 mean time: 0.007450 ms
embedding_fp16x8               mean time: 0.036198 ms
embedding_fp16x8_packed        mean time: 0.005342 ms
####################################################################################################
emd_size: 102400, emd_dim: 4096, seq_len: 4096
torch                          mean time: 0.167067 ms
embedding                      mean time: 0.174205 ms
embedding_fp32x4               mean time: 0.186684 ms
embedding_fp32x4_packed        mean time: 0.169705 ms
torch                          mean time: 0.014992 ms
embedding_half                 mean time: 0.024386 ms
embedding_fp16x8               mean time: 0.135323 ms
embedding_fp16x8_packed        mean time: 0.015799 ms
```
