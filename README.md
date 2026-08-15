# TaperHashTable C++ Standalone Benchmark

从 OmniOperator 核心抽离的独立可编译 TaperHashTable benchmark，用于与 Rust 版 (`rust-taper-hashmap/`) 进行性能对比。

## 设计对应关系

| C++ (本项目) | Rust (rust-taper-hashmap) | OmniOperator 原始 |
|---|---|---|
| `taper_hashtable.h` | `taper_hashmap.rs` | `taper_hashtable.h` |
| `row_container.h` | `row_container.rs` | `row_container.h` |
| `column_marshaller.h` | `column_marshaller.rs` | `column_marshaller.h` |
| `taper_bench.cpp` | `benches/hashmap_bench.rs` | N/A |

## 核心算法一致性

- 128 字节对齐 Chunk, 8 slots/chunk
- SWAR bitmask tag matching (PHBitMask)
- 6 字节压缩指针 (48-bit addressing)
- 0.9 load factor threshold
- Prefetch-driven batch emplace with collision iteration
- Merged varchar arena storage (多 varchar 列合并到一个 arena block)
- 5-step emplace pipeline: hash → batch emplace → store → batch compare → scalar fallback

## 构建

```bash
cd cpp-taper-bench
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 运行测试

```bash
./taper_test
```

## 运行 Benchmark

```bash
# 运行全部
./taper_bench

# 只跑 4str_0int, ht=65536 的配置
./taper_bench --benchmark_filter=".*4str_0int.*ht=65536.*"

# 只跑 taper (不跑 baseline)
./taper_bench --benchmark_filter="taper/.*"

# JSON 输出 (方便对比)
./taper_bench --benchmark_format=json --benchmark_out=cpp_results.json
```

## 与 Rust 版对比

```bash
# Rust 端
cd ../rust-taper-hashmap
cargo bench -- "4str_0int.*ht=65536"

# C++ 端
cd ../cpp-taper-bench/build
./taper_bench --benchmark_filter=".*4str_0int.*ht=65536.*" --benchmark_format=json
```

## Benchmark 参数

- **Key types**: 4str_0int, 3str_1int, 2str_2int, 1str_3int, 0str_4int
- **HT size**: 16384, 65536, 262144
- **Load factor**: 0.5, 0.75
- **Selectivity**: 0.1, 0.5, 0.9
- **Probe rows**: 1,000,000

## 与原始 OmniOperator 的差异

1. **无向量抽象**: 直接使用 decoded column data (raw pointers)，跳过 VectorBatch/DecodedVector
2. **无 LLVM**: 不依赖 LLVM-15 JIT
3. **无 CRC32 硬件指令**: hash 统一使用 xxHash (与 Rust 端一致，保证公平对比)
4. **Header-only**: 所有核心逻辑在头文件中，便于编译器内联优化
5. **跨平台**: 支持 x86_64 和 aarch64 (prefetch 指令自动适配)
