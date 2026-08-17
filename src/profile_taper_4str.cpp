/// Profile target: 4str_0int, ht=65536, 200 iterations (~15-20s on Kunpeng).
/// Use with: perf record -F 9999 -g --call-graph fp -- ./build-sve/profile_taper_4str
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

int main() {
    constexpr size_t NS=4, HT=65536, NK=(size_t)(HT*0.5), NP=1000000;
    constexpr double SEL=0.5;
    std::mt19937_64 rng(42);

    // ── Build phase: NK unique keys ──────────────────────────────────────────
    std::vector<std::vector<std::vector<uint8_t>>> bs(NS);
    for(size_t c=0;c<NS;c++){bs[c].resize(NK);for(size_t i=0;i<NK;i++){auto s="key_"+std::to_string(i)+"_c"+std::to_string(c);bs[c][i].assign(s.begin(),s.end());}}
    std::vector<int64_t> bh(NK);
    for(size_t i=0;i<NK;i++){uint64_t h=0;for(size_t c=0;c<NS;c++)h=HBS(bs[c][i].data(),bs[c][i].size(),h);bh[i]=static_cast<int64_t>(h);}
    std::vector<int64_t> bv(NK); for(size_t i=0;i<NK;i++)bv[i]=i%1000;

    // ── Probe phase: NP rows ─────────────────────────────────────────────────
    size_t nH=(size_t)(NP*SEL), nM=NP-nH;
    std::vector<std::vector<std::vector<uint8_t>>> ps(NS); for(auto&v:ps)v.reserve(NP);
    std::vector<int64_t> ph; ph.reserve(NP);
    std::uniform_int_distribution<size_t> kd(0,NK-1);
    for(size_t i=0;i<nH;i++){size_t idx=kd(rng);for(size_t c=0;c<NS;c++)ps[c].push_back(bs[c][idx]);ph.push_back(bh[idx]);}
    for(size_t i=0;i<nM;i++){uint64_t h=0;for(size_t c=0;c<NS;c++){auto s="miss_"+std::to_string(i)+"_"+std::to_string(c);std::vector<uint8_t> b(s.begin(),s.end());h=HBS(b.data(),b.size(),h);ps[c].push_back(std::move(b));}ph.push_back(static_cast<int64_t>(h));}
    // shuffle
    std::vector<size_t> ord(NP); std::iota(ord.begin(),ord.end(),0);
    for(size_t i=NP-1;i>0;i--){std::uniform_int_distribution<size_t> d(0,i);std::swap(ord[i],ord[d(rng)]);}
    std::vector<std::vector<std::vector<uint8_t>>> sp(NS);
    for(size_t c=0;c<NS;c++){sp[c].resize(NP);for(size_t i=0;i<NP;i++)sp[c][i]=std::move(ps[c][ord[i]]);}
    std::vector<int64_t> sh(NP); for(size_t i=0;i<NP;i++)sh[i]=ph[ord[i]];
    std::vector<int64_t> pv(NP); for(size_t i=0;i<NP;i++)pv[i]=i%1000;

    // ── Combine build + probe ────────────────────────────────────────────────
    size_t TR=NK+NP;
    std::vector<std::vector<std::vector<uint8_t>>> all(NS);
    for(size_t c=0;c<NS;c++){all[c]=bs[c];for(auto&v:sp[c])all[c].push_back(std::move(v));}
    std::vector<int64_t> ah=bh; ah.insert(ah.end(),sh.begin(),sh.end());
    std::vector<int64_t> av=bv; av.insert(av.end(),pv.begin(),pv.end());
    std::vector<std::vector<const uint8_t*>> aptrs(NS); std::vector<std::vector<size_t>> alens(NS);
    for(size_t c=0;c<NS;c++){aptrs[c].resize(TR);alens[c].resize(TR);for(size_t i=0;i<TR;i++){aptrs[c][i]=all[c][i].data();alens[c][i]=all[c][i].size();}}

    // ── Profile loop (200 iters for ~15-20s of sampling) ─────────────────────
    size_t lastG=0;
    for(int iter=0;iter<200;iter++){
        taper::SimpleArenaAllocator pool;
        std::vector<taper::ColumnDesc> cd(NS, taper::ColumnDesc::Varchar);
        taper::TaperColumnSerializeHandler t(pool, 8, cd, HT);
        std::vector<taper::ColumnInput> cols;
        for(size_t c=0;c<NS;c++) cols.push_back(taper::ColumnInput::MakeVarchar(aptrs[c].data(),alens[c].data()));
        t.EmplaceTableWithDecode(ah.data(),static_cast<int32_t>(TR),cols,av.data());
        lastG=t.NumGroups();
    }
    printf("Done. %zu groups created per iteration.\n", lastG);
}
