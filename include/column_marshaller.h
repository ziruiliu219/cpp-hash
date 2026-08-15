/**
 * Standalone TaperColumnSerializeHandler — naming aligned with OmniOperator.
 * Target: Linux aarch64.
 */
#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>
#include <memory>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include "taper_hashtable.h"
#include "row_container.h"
#include "simple_arena_allocator.h"

namespace taper {

enum class ColumnDesc { Int64, Varchar };

struct ColumnInput {
    ColumnDesc type;
    union { const int64_t* int64Data; struct { const uint8_t* const* ptrs; const size_t* lens; } vc; };
    static ColumnInput MakeInt64(const int64_t* d) { ColumnInput c; c.type=ColumnDesc::Int64; c.int64Data=d; return c; }
    static ColumnInput MakeVarchar(const uint8_t* const* p, const size_t* l) { ColumnInput c; c.type=ColumnDesc::Varchar; c.vc.ptrs=p; c.vc.lens=l; return c; }
};

// ─── Varchar helpers (same as OmniOperator) ─────────────────────────

inline uint8_t ComputeRowLenSize(size_t len) { return len<=0xFF?1:len<=0xFFFF?2:4; }

inline size_t SerializeVarcharToBuffer(uint8_t* writePos, const uint8_t* data, size_t len) {
    uint8_t rowLenSize = ComputeRowLenSize(len); *writePos = rowLenSize;
    uint32_t l32 = static_cast<uint32_t>(len); memcpy(writePos+1, &l32, rowLenSize);
    if (len) memcpy(writePos+1+rowLenSize, data, len);
    return 1+rowLenSize+len;
}

inline size_t ComputeVarCharSerializedSize(const uint8_t* data) {
    uint8_t rowLenSize = *data; if (!rowLenSize) return 1;
    size_t stringLen=0;
    switch(rowLenSize){case 1:stringLen=*(data+1);break;case 2:{uint16_t v;memcpy(&v,data+1,2);stringLen=v;break;}default:{uint32_t v;memcpy(&v,data+1,4);stringLen=v;}}
    return 1+rowLenSize+stringLen;
}

inline bool CompareVarcharFromRow(const uint8_t* rowData, const uint8_t* input, size_t inputLen) {
    uint8_t rowLenSize = *rowData; if(!rowLenSize) return false;
    size_t stringLen=0;
    switch(rowLenSize){case 1:stringLen=*(rowData+1);break;case 2:{uint16_t v;memcpy(&v,rowData+1,2);stringLen=v;break;}default:{uint32_t v;memcpy(&v,rowData+1,4);stringLen=v;}}
    if (stringLen!=inputLen) return false;
    if (stringLen==0) return true;
    const uint8_t* ptr = rowData+1+rowLenSize;
#if defined(__aarch64__)
    size_t i=0;
    for (; i+16<=stringLen; i+=16) {
        uint8x16_t lhs = vld1q_u8(ptr+i);
        uint8x16_t rhs = vld1q_u8(input+i);
        uint8x16_t cmp = vceqq_u8(lhs, rhs);
        uint64x2_t wide = vreinterpretq_u64_u8(cmp);
        if (vgetq_lane_u64(wide,0)!=~0ULL || vgetq_lane_u64(wide,1)!=~0ULL) return false;
    }
    for (; i<stringLen; i++) { if (ptr[i]!=input[i]) return false; }
    return true;
#else
    return memcmp(ptr, input, stringLen)==0;
#endif
}

// ─── SetRowPtr / GetRowPtr — same as OmniOperator ──────────────────

static inline void SetRowPtr(char* buf, uint8_t* ptr) {
    uint64_t val = reinterpret_cast<uint64_t>(ptr);
    memcpy(buf, &val, ROW_PTR_SIZE);
}

static inline uint8_t* GetRowPtr(const char* buf) {
    uint64_t val = 0;
    memcpy(&val, buf, ROW_PTR_SIZE);
    return reinterpret_cast<uint8_t*>(val);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TaperColumnSerializeHandler — naming matches OmniOperator
// ═══════════════════════════════════════════════════════════════════════════════

class TaperColumnSerializeHandler {
public:
    using HashTable = TaperFlatHashTable;

    int32_t totalAggValueSize = 0;
    int32_t totalAggStatesSize = 0;
    std::unique_ptr<HashTable> table;
    std::unique_ptr<RowContainer> aggRows;
    std::vector<int64_t> workingHashVals;
    std::vector<int32_t> workingUpdateIndices;
    int32_t workingUpdateCount = 0;
    std::vector<int32_t> keyTypeSizes;
    std::vector<int32_t> varcharColIndices;
    int32_t varcharSlotColIdx = -1;
    std::vector<const uint8_t*> mergedVarcharCache_;
    int32_t mergedVarcharCacheCount_ = 0;
    std::vector<uint8_t*> groups;

    TaperColumnSerializeHandler(SimpleArenaAllocator& pool, int32_t aggStatesSize,
                                const std::vector<ColumnDesc>& colDescs, size_t initCap)
        : totalAggStatesSize(aggStatesSize), colDescs_(colDescs)
    {
        totalAggValueSize = aggStatesSize + static_cast<int32_t>(sizeof(size_t));
        table = std::make_unique<HashTable>(initCap);

        std::vector<size_t> keySizes;
        std::vector<ColumnKind> kinds;
        for (size_t i = 0; i < colDescs.size(); i++) {
            if (colDescs[i] == ColumnDesc::Int64) {
                keySizes.push_back(8); kinds.push_back(ColumnKind::Fixed);
                keyTypeSizes.push_back(8);
            } else {
                keySizes.push_back(0); kinds.push_back(ColumnKind::Varchar);
                keyTypeSizes.push_back(0);
                varcharColIndices.push_back(static_cast<int32_t>(i));
            }
        }
        if (varcharColIndices.size() > 1) {
            varcharSlotColIdx = varcharColIndices[0];
        }
        aggRows = std::make_unique<RowContainer>(keySizes, kinds, aggStatesSize, pool);
    }

    int32_t AggStateOffset() const { return aggRows->AggStateOffset(); }
    size_t NumGroups() const { return aggRows->NumRows(); }

    /// EmplaceTableWithDecode — exact 5-step pipeline matching OmniOperator.
    void EmplaceTableWithDecode(const int64_t* hashes, int32_t rowsNum,
        const std::vector<ColumnInput>& columns, const int64_t* aggValues)
    {
        if (rowsNum <= 0) return;

        // Ensure hash table capacity >= numRows before EmplaceBatch
        // (matches OmniOperator production: table is pre-sized for expected cardinality)
        while (table->Capacity() < static_cast<size_t>(rowsNum)) {
            // Force expand by inserting nothing — just trigger the resize
            // Simpler: just recreate with larger capacity
            size_t newCap = table->Capacity() * 2 / 8; // chunks needed
            if (newCap == 0) newCap = 1;
            while (newCap * 8 < static_cast<size_t>(rowsNum)) newCap *= 2;
            table = std::make_unique<HashTable>(newCap);
        }

        int32_t groupColNum = static_cast<int32_t>(colDescs_.size());

        groups.resize(rowsNum, nullptr);
        std::vector<uint8_t*> newGroups;
        newGroups.reserve(rowsNum / 4);
        std::vector<uint32_t> newGroupRowIndices(rowsNum);
        int32_t newGroupCount = 0;

        workingUpdateIndices.resize(rowsNum);
        workingUpdateCount = 0;

        // ─── Step 2: EmplaceBatch ───────────────────────────────
        auto initRow = [&](uint32_t rowIdx, char* data) -> char* {
            auto* row = aggRows->NewRow();
            SetRowPtr(data, reinterpret_cast<uint8_t*>(row));
            newGroups.push_back(GetRowPtr(data));
            newGroupRowIndices[newGroupCount++] = rowIdx;
            return row;
        };

        table->EmplaceBatch(hashes, rowsNum,
            [](int32_t) { return false; },
            [&](uint32_t rowIdx, char* data) { initRow(rowIdx, data); },
            [&](uint32_t rowIdx, char* data, bool initFlag) {
                groups[rowIdx] = GetRowPtr(data);
                if (!initFlag) {
                    workingUpdateIndices[workingUpdateCount++] = rowIdx;
                }
            }
        );

        // ─── Step 3: Store keys for new groups ──────────────────
        // Same structure as OmniOperator:
        // 1. BatchStoreMergedVarcharColumns (all new groups, merged varchar)
        // 2. Per-column loop for remaining types
        int32_t aggOffset = AggStateOffset();
        size_t newGroupsStartIdx = 0;

        if (newGroupCount > 0) {
            // Store merged VARCHAR columns first (all in one contiguous block per row)
            if (varcharColIndices.size() > 1) {
                BatchStoreMergedVarcharColumns(columns,
                    newGroups.data() + newGroupsStartIdx,
                    newGroupRowIndices.data(), newGroupCount);
            }

            for (int32_t colIdx = 0; colIdx < groupColNum; colIdx++) {
                auto col = aggRows->ColumnAt(colIdx);
                int32_t offset = col.Offset();
                uint32_t nullByte = col.NullByte();
                uint8_t nullMask = col.NullMask();

                if (colDescs_[colIdx] == ColumnDesc::Varchar) {
                    if (varcharColIndices.size() > 1) {
                        continue; // handled by BatchStoreMergedVarcharColumns
                    }
                    BatchStoreKeyColumnVarcharTyped(colIdx, offset, nullByte, nullMask,
                        newGroups.data() + newGroupsStartIdx, newGroupRowIndices.data(), newGroupCount, columns);
                } else {
                    BatchStoreKeyColumnTyped(colIdx, offset, nullByte, nullMask,
                        newGroups.data() + newGroupsStartIdx, newGroupRowIndices.data(), newGroupCount, columns);
                }
            }

            // Store agg values
            for (int32_t i = 0; i < newGroupCount; i++) {
                char* row = reinterpret_cast<char*>(newGroups[newGroupsStartIdx + i]);
                uint32_t rowIdx = newGroupRowIndices[i];
                RowContainer::StoreValue<int64_t>(row, aggOffset, aggValues[rowIdx]);
            }
        }

        if (workingUpdateCount == 0) return;
        int32_t count = workingUpdateCount;

        // ─── Step 4: GetUnequalsNumWithDecode ───────────────────
        int32_t unequalsNum = GetUnequalsNumWithDecode(count, groupColNum, columns);

        // ─── Step 5: Scalar fallback (unequal rows) ─────────────
        for (int32_t ui = 0; ui < unequalsNum; ui++) {
            int32_t rowIdx = workingUpdateIndices[ui];
            int64_t hash = hashes[rowIdx];
            table->Emplace(hash,
                [&](const char* valBuf) -> bool {
                    return CompareKeysWithDecode(
                        reinterpret_cast<const char*>(GetRowPtr(valBuf)),
                        rowIdx, columns, groupColNum);
                },
                [&](char* data) {
                    auto* row = aggRows->NewRow();
                    SetRowPtr(data, reinterpret_cast<uint8_t*>(row));
                    StoreKeyOneRowFromDecode(row, rowIdx, columns, groupColNum);
                    RowContainer::StoreValue<int64_t>(row, aggOffset, aggValues[rowIdx]);
                },
                [&](char* data, bool isNew) {
                    if (!isNew) {
                        auto* rp = GetRowPtr(data);
                        *reinterpret_cast<int64_t*>(rp + aggOffset) += aggValues[rowIdx];
                    }
                    groups[rowIdx] = GetRowPtr(data);
                }
            );
        }

        // Accumulate for confirmed equal rows
        for (int32_t ui = unequalsNum; ui < count; ui++) {
            int32_t rowIdx = workingUpdateIndices[ui];
            auto* rp = groups[rowIdx];
            *reinterpret_cast<int64_t*>(rp + aggOffset) += aggValues[rowIdx];
        }
    }

private:
    std::vector<ColumnDesc> colDescs_;

    /// GetUnequalsNumWithDecode — batch compare existing groups against input (Step 4).
    /// Returns number of unequal rows (swapped to front of workingUpdateIndices).
    int32_t GetUnequalsNumWithDecode(int32_t count, int32_t groupColNum,
                                     const std::vector<ColumnInput>& columns)
    {
        // Build merged varchar cache
        mergedVarcharCache_.clear();
        mergedVarcharCacheCount_ = 0;
        if (varcharColIndices.size() > 1) {
            mergedVarcharCacheCount_ = static_cast<int32_t>(varcharColIndices.size());
            int32_t maxIdx = 0;
            for (int32_t w = 0; w < count; w++)
                maxIdx = std::max(maxIdx, workingUpdateIndices[w]);
            mergedVarcharCache_.resize((maxIdx + 1) * mergedVarcharCacheCount_, nullptr);
            for (int32_t w = 0; w < count; w++) {
                int32_t idx = workingUpdateIndices[w];
                GetAllMergedVarcharPtrs(
                    reinterpret_cast<const char*>(groups[idx]),
                    &mergedVarcharCache_[idx * mergedVarcharCacheCount_],
                    mergedVarcharCacheCount_);
            }
        }

        int32_t idxFrom = 0;
        for (int32_t groupColIdx = 0; groupColIdx < groupColNum && idxFrom < count; groupColIdx++) {
            auto col = aggRows->ColumnAt(groupColIdx);
            int32_t offset = col.Offset();

            if (colDescs_[groupColIdx] == ColumnDesc::Int64) {
                const int64_t* inputValues = columns[groupColIdx].int64Data;
#if defined(__aarch64__)
                {
                    int32_t i = idxFrom;
                    while (i + 2 <= count) {
                        int32_t idx0 = workingUpdateIndices[i], idx1 = workingUpdateIndices[i+1];
                        int64_t s0 = RowContainer::ReadValue<int64_t>(reinterpret_cast<const char*>(groups[idx0]), offset);
                        int64_t s1 = RowContainer::ReadValue<int64_t>(reinterpret_cast<const char*>(groups[idx1]), offset);
                        int64x2_t vStored = vcombine_s64(vcreate_s64(static_cast<uint64_t>(s0)), vcreate_s64(static_cast<uint64_t>(s1)));
                        int64x2_t vInput = vcombine_s64(vcreate_s64(static_cast<uint64_t>(inputValues[idx0])), vcreate_s64(static_cast<uint64_t>(inputValues[idx1])));
                        uint64x2_t cmp = vceqq_s64(vStored, vInput);
                        if (vgetq_lane_u64(cmp, 0) == 0) { std::swap(workingUpdateIndices[i], workingUpdateIndices[idxFrom]); idxFrom++; }
                        if (vgetq_lane_u64(cmp, 1) == 0) { std::swap(workingUpdateIndices[i+1], workingUpdateIndices[idxFrom]); idxFrom++; }
                        i += 2;
                    }
                    for (; i < count; i++) {
                        int32_t idx = workingUpdateIndices[i];
                        if (RowContainer::ReadValue<int64_t>(reinterpret_cast<const char*>(groups[idx]), offset) != inputValues[idx])
                            { std::swap(workingUpdateIndices[i], workingUpdateIndices[idxFrom]); idxFrom++; }
                    }
                }
#else
                for (int32_t i = idxFrom; i < count; i++) {
                    int32_t idx = workingUpdateIndices[i];
                    if (RowContainer::ReadValue<int64_t>(reinterpret_cast<const char*>(groups[idx]), offset) != inputValues[idx])
                        { std::swap(workingUpdateIndices[i], workingUpdateIndices[idxFrom]); idxFrom++; }
                }
#endif
            } else {
                // VARCHAR column
                auto* inputPtrs = columns[groupColIdx].vc.ptrs;
                auto* inputLens = columns[groupColIdx].vc.lens;
                if (varcharColIndices.size() > 1) {
                    int32_t vcPos = 0;
                    for (int32_t v = 0; v < static_cast<int32_t>(varcharColIndices.size()); v++)
                        if (varcharColIndices[v] == groupColIdx) { vcPos = v; break; }
                    for (int32_t i = idxFrom; i < count; i++) {
                        int32_t idx = workingUpdateIndices[i];
                        auto* arenaPtr = mergedVarcharCache_[idx * mergedVarcharCacheCount_ + vcPos];
                        if (!arenaPtr || !CompareVarcharFromRow(arenaPtr, inputPtrs[idx], inputLens[idx]))
                            { std::swap(workingUpdateIndices[i], workingUpdateIndices[idxFrom]); idxFrom++; }
                    }
                } else {
                    for (int32_t i = idxFrom; i < count; i++) {
                        int32_t idx = workingUpdateIndices[i];
                        const uint8_t* arenaPtr;
                        memcpy(&arenaPtr, reinterpret_cast<const char*>(groups[idx]) + offset, sizeof(arenaPtr));
                        if (!arenaPtr || !CompareVarcharFromRow(arenaPtr, inputPtrs[idx], inputLens[idx]))
                            { std::swap(workingUpdateIndices[i], workingUpdateIndices[idxFrom]); idxFrom++; }
                    }
                }
            }
        }
        return idxFrom;
    }

    /// BatchStoreMergedVarcharColumns — batch store merged varchar for all new groups.
    /// Same as OmniOperator: iterates new groups, serializes all varchar columns into one arena block per row.
    void BatchStoreMergedVarcharColumns(const std::vector<ColumnInput>& columns,
                                        uint8_t** rows, uint32_t* rowIndices, int32_t rowCount)
    {
        for (int32_t i = 0; i < rowCount; i++) {
            char* row = reinterpret_cast<char*>(rows[i]);
            uint32_t rowIdx = rowIndices[i];
            size_t totalSize = 0;
            for (auto vcIdx : varcharColIndices) {
                totalSize += 1 + ComputeRowLenSize(columns[vcIdx].vc.lens[rowIdx]) + columns[vcIdx].vc.lens[rowIdx];
            }
            uint8_t* blockStart = aggRows->ArenaAlloc(totalSize);
            uint8_t* writePos = blockStart;
            for (auto vcIdx : varcharColIndices) {
                auto col = aggRows->ColumnAt(vcIdx);
                RowContainer::ClearNullAt(row, col.NullByte(), col.NullMask());
                writePos += SerializeVarcharToBuffer(writePos, columns[vcIdx].vc.ptrs[rowIdx], columns[vcIdx].vc.lens[rowIdx]);
            }
            auto slotCol = aggRows->ColumnAt(varcharSlotColIdx);
            memcpy(row + slotCol.Offset(), &blockStart, sizeof(blockStart));
        }
    }

    /// BatchStoreKeyColumnVarcharTyped — batch store single varchar column for new groups.
    void BatchStoreKeyColumnVarcharTyped(int32_t colIdx, int32_t offset, uint32_t nullByte, uint8_t nullMask,
                                         uint8_t** rows, uint32_t* rowIndices, int32_t rowCount,
                                         const std::vector<ColumnInput>& columns)
    {
        for (int32_t i = 0; i < rowCount; i++) {
            char* row = reinterpret_cast<char*>(rows[i]);
            uint32_t rowIdx = rowIndices[i];
            RowContainer::ClearNullAt(row, nullByte, nullMask);
            size_t len = columns[colIdx].vc.lens[rowIdx];
            size_t totalSize = 1 + ComputeRowLenSize(len) + len;
            uint8_t* arenaPtr = aggRows->ArenaAlloc(totalSize);
            SerializeVarcharToBuffer(arenaPtr, columns[colIdx].vc.ptrs[rowIdx], len);
            memcpy(row + offset, &arenaPtr, sizeof(arenaPtr));
        }
    }

    /// BatchStoreKeyColumnTyped — batch store fixed-width (int64) column for new groups.
    void BatchStoreKeyColumnTyped(int32_t colIdx, int32_t offset, uint32_t nullByte, uint8_t nullMask,
                                  uint8_t** rows, uint32_t* rowIndices, int32_t rowCount,
                                  const std::vector<ColumnInput>& columns)
    {
        for (int32_t i = 0; i < rowCount; i++) {
            char* row = reinterpret_cast<char*>(rows[i]);
            uint32_t rowIdx = rowIndices[i];
            RowContainer::ClearNullAt(row, nullByte, nullMask);
            RowContainer::StoreValue<int64_t>(row, offset, columns[colIdx].int64Data[rowIdx]);
        }
    }

    /// StoreKeyOneRowFromDecode — store all key columns for one row (used by Step 5 scalar fallback).
    void StoreKeyOneRowFromDecode(char* row, int32_t rowIdx,
        const std::vector<ColumnInput>& columns, int32_t groupColNum)
    {
        if (varcharColIndices.size() > 1) {
            // Merged varchar: serialize all varchar columns into one arena block
            size_t totalSize = 0;
            for (auto vcIdx : varcharColIndices)
                totalSize += 1 + ComputeRowLenSize(columns[vcIdx].vc.lens[rowIdx]) + columns[vcIdx].vc.lens[rowIdx];
            uint8_t* blockStart = aggRows->ArenaAlloc(totalSize);
            uint8_t* writePos = blockStart;
            for (auto vcIdx : varcharColIndices) {
                auto col = aggRows->ColumnAt(vcIdx);
                RowContainer::ClearNullAt(row, col.NullByte(), col.NullMask());
                writePos += SerializeVarcharToBuffer(writePos, columns[vcIdx].vc.ptrs[rowIdx], columns[vcIdx].vc.lens[rowIdx]);
            }
            // Store merged block pointer in slot column
            auto slotCol = aggRows->ColumnAt(varcharSlotColIdx);
            memcpy(row + slotCol.Offset(), &blockStart, sizeof(blockStart));
        } else if (varcharColIndices.size() == 1) {
            int32_t vcIdx = varcharColIndices[0];
            auto col = aggRows->ColumnAt(vcIdx);
            RowContainer::ClearNullAt(row, col.NullByte(), col.NullMask());
            size_t len = columns[vcIdx].vc.lens[rowIdx];
            size_t totalSize = 1 + ComputeRowLenSize(len) + len;
            uint8_t* arenaPtr = aggRows->ArenaAlloc(totalSize);
            SerializeVarcharToBuffer(arenaPtr, columns[vcIdx].vc.ptrs[rowIdx], len);
            memcpy(row + col.Offset(), &arenaPtr, sizeof(arenaPtr));
        }
        // Store int64 columns
        for (int32_t colIdx = 0; colIdx < groupColNum; colIdx++) {
            if (colDescs_[colIdx] == ColumnDesc::Int64) {
                auto col = aggRows->ColumnAt(colIdx);
                RowContainer::ClearNullAt(row, col.NullByte(), col.NullMask());
                RowContainer::StoreValue<int64_t>(row, col.Offset(), columns[colIdx].int64Data[rowIdx]);
            }
        }
    }

    /// GetAllMergedVarcharPtrs — fill pointers to each varchar in the merged block.
    void GetAllMergedVarcharPtrs(const char* row, const uint8_t** outPtrs, int32_t maxCount) const {
        auto slotCol = aggRows->ColumnAt(varcharSlotColIdx);
        const uint8_t* blockPtr;
        memcpy(&blockPtr, row + slotCol.Offset(), sizeof(blockPtr));
        if (!blockPtr) { memset(outPtrs, 0, maxCount * sizeof(uint8_t*)); return; }
        const uint8_t* pos = blockPtr;
        for (int32_t i = 0; i < maxCount; i++) {
            auto vcIdx = varcharColIndices[i];
            auto col = aggRows->ColumnAt(vcIdx);
            if (RowContainer::IsNullAt(row, col.NullByte(), col.NullMask())) {
                outPtrs[i] = nullptr; pos += 1;
            } else {
                outPtrs[i] = pos; pos += ComputeVarCharSerializedSize(pos);
            }
        }
    }

    /// CompareKeysWithDecode — compare all key columns in a row against input.
    bool CompareKeysWithDecode(const char* row, int32_t rowIdx,
        const std::vector<ColumnInput>& columns, int32_t groupColNum) const
    {
        for (int32_t colIdx = 0; colIdx < groupColNum; colIdx++) {
            auto col = aggRows->ColumnAt(colIdx);
            if (colDescs_[colIdx] == ColumnDesc::Int64) {
                if (RowContainer::ReadValue<int64_t>(row, col.Offset()) != columns[colIdx].int64Data[rowIdx])
                    return false;
            } else {
                const uint8_t* arenaPtr;
                memcpy(&arenaPtr, row + col.Offset(), sizeof(arenaPtr));
                if (!arenaPtr || !CompareVarcharFromRow(arenaPtr, columns[colIdx].vc.ptrs[rowIdx], columns[colIdx].vc.lens[rowIdx]))
                    return false;
            }
        }
        return true;
    }
};

} // namespace taper
