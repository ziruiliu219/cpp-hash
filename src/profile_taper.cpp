/// Quick profile: 1str_3int, ht=65536, 20 iterations.
/// varchar col: 1, int64 cols: 3
/// int64 columns exercise SVE SveBatchCompareInt64NoNull on aarch64+SVE machines.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "column_marshaller.h"

static inline uint64_t HBS(const uint8_t* d, size_t l, uint64_t s) { return XXH3_64bits_withSeed(d,l,s); }
static inline uint64_t HBI(int64_t v, uint64_t s) { return XXH3_64bits_withSeed(&v,8,s); }

int main() {
    constexpr size_t NS=1, NI=3, HT=65536, NK=(size_t)(HT*0.5), NP=1000000;
    constexpr double SEL=0.5;
    std::mt19937_64 rng(42);

    // ── Build phase: NK unique keys ──────────────────────────────────────────
    // 1 varchar column
    std::vector<std::vector<uint8_t>> bstr(NK);
    for(size_t i=0;i<NK;i++){auto s="key_"+std::to_string(i);bstr[i].assign(s.begin(),s.end());}
    // 3 int64 columns
    std::vector<std::vector<int64_t>> bint(NI);
    for(size_t c=0;c<NI;c++){bint[c].resize(NK);for(size_t i=0;i<NK;i++)bint[c][i]=int64_t(i)*(97+int64_t(c)*31)+1;}
    // combined hashes
    std::vector<int64_t> bh(NK);
    for(size_t i=0;i<NK;i++){
        uint64_t h=HBS(bstr[i].data(),bstr[i].size(),0);
        for(size_t c=0;c<NI;c++) h=HBI(bint[c][i],h);
        bh[i]=static_cast<int64_t>(h);
    }
    std::vector<int64_t> bv(NK); for(size_t i=0;i<NK;i++)bv[i]=i%1000;

    // ── Probe phase: NP rows (SEL fraction are hits) ─────────────────────────
    size_t nH=(size_t)(NP*SEL), nM=NP-nH;
    std::vector<std::vector<uint8_t>> pstr; pstr.reserve(NP);
    std::vector<std::vector<int64_t>> pint(NI); for(auto&v:pint)v.reserve(NP);
    std::vector<int64_t> ph; ph.reserve(NP);
    std::uniform_int_distribution<size_t> kd(0,NK-1);
    for(size_t i=0;i<nH;i++){
        size_t idx=kd(rng);
        pstr.push_back(bstr[idx]);
        for(size_t c=0;c<NI;c++) pint[c].push_back(bint[c][idx]);
        ph.push_back(bh[idx]);
    }
    for(size_t i=0;i<nM;i++){
        auto s="miss_"+std::to_string(i); std::vector<uint8_t> b(s.begin(),s.end());
        uint64_t h=HBS(b.data(),b.size(),0);
        for(size_t c=0;c<NI;c++){int64_t v=int64_t(NK+1)*200+int64_t(i)*31+int64_t(c);h=HBI(v,h);pint[c].push_back(v);}
        pstr.push_back(std::move(b));
        ph.push_back(static_cast<int64_t>(h));
    }
    // shuffle
    std::vector<size_t> ord(NP); std::iota(ord.begin(),ord.end(),0);
    for(size_t i=NP-1;i>0;i--){std::uniform_int_distribution<size_t> d(0,i);std::swap(ord[i],ord[d(rng)]);}
    std::vector<std::vector<uint8_t>> sp(NP); for(size_t i=0;i<NP;i++)sp[i]=std::move(pstr[ord[i]]);
    std::vector<std::vector<int64_t>> si(NI); for(size_t c=0;c<NI;c++){si[c].resize(NP);for(size_t i=0;i<NP;i++)si[c][i]=pint[c][ord[i]];}
    std::vector<int64_t> sh(NP); for(size_t i=0;i<NP;i++)sh[i]=ph[ord[i]];
    std::vector<int64_t> pv(NP); for(size_t i=0;i<NP;i++)pv[i]=i%1000;

    // ── Combine build + probe ────────────────────────────────────────────────
    size_t TR=NK+NP;
    std::vector<std::vector<uint8_t>> allstr=bstr;
    for(auto&v:sp) allstr.push_back(std::move(v));
    std::vector<std::vector<int64_t>> allint(NI);
    for(size_t c=0;c<NI;c++){allint[c]=bint[c];allint[c].insert(allint[c].end(),si[c].begin(),si[c].end());}
    std::vector<int64_t> ah=bh; ah.insert(ah.end(),sh.begin(),sh.end());
    std::vector<int64_t> av=bv; av.insert(av.end(),pv.begin(),pv.end());

    // pointer/len arrays for varchar
    std::vector<const uint8_t*> aptrs(TR); std::vector<size_t> alens(TR);
    for(size_t i=0;i<TR;i++){aptrs[i]=allstr[i].data();alens[i]=allstr[i].size();}

    // ── Profile loop ─────────────────────────────────────────────────────────
    size_t lastG=0;
    for(int iter=0;iter<20;iter++){
        taper::SimpleArenaAllocator pool;
        // column layout: [varchar, int64, int64, int64]
        std::vector<taper::ColumnDesc> cd;
        cd.push_back(taper::ColumnDesc::Varchar);
        for(size_t c=0;c<NI;c++) cd.push_back(taper::ColumnDesc::Int64);
        // Use (TR+7)/8 chunks so initial capacity >= TR, avoiding any rehash/rebuild
        taper::TaperColumnSerializeHandler t(pool, 8, cd, (TR + 7) / 8);
        std::vector<taper::ColumnInput> cols;
        cols.push_back(taper::ColumnInput::MakeVarchar(aptrs.data(), alens.data()));
        for(size_t c=0;c<NI;c++) cols.push_back(taper::ColumnInput::MakeInt64(allint[c].data()));
        t.EmplaceTableWithDecode(ah.data(), static_cast<int32_t>(TR), cols, av.data());
        lastG=t.NumGroups();
    }
    printf("Done. %zu groups created per iteration.\n", lastG);
}
