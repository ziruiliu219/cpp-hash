/// Google Benchmark: TaperHashTable vs baseline. Matches Rust criterion bench.
#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "column_marshaller.h"

static inline uint64_t HB(const uint8_t* d,size_t l,uint64_t s){return XXH3_64bits_withSeed(d,l,s);}
static inline uint64_t HC(uint64_t s,int64_t v){return XXH3_64bits_withSeed(&v,8,s);}

struct BenchData {
    std::vector<std::vector<std::vector<uint8_t>>> strCols;
    std::vector<std::vector<int64_t>> intCols;
    std::vector<int64_t> hashes, values;
    std::vector<std::vector<const uint8_t*>> strPtrs;
    std::vector<std::vector<size_t>> strLens;
    size_t nStr,nInt,totalRows;
};

BenchData GenData(size_t nStr,size_t nInt,size_t nKeys,size_t nProbe,double sel,uint64_t seed=42){
    std::mt19937_64 rng(seed); BenchData d; d.nStr=nStr; d.nInt=nInt;
    d.strCols.resize(nStr); for(size_t c=0;c<nStr;c++){d.strCols[c].resize(nKeys);for(size_t i=0;i<nKeys;i++){auto s="key_"+std::to_string(i)+"_c"+std::to_string(c);d.strCols[c][i].assign(s.begin(),s.end());}}
    d.intCols.resize(nInt); for(size_t c=0;c<nInt;c++){d.intCols[c].resize(nKeys);for(size_t i=0;i<nKeys;i++)d.intCols[c][i]=int64_t(i)*(97+c*31)+1;}
    std::vector<int64_t> bh(nKeys); for(size_t i=0;i<nKeys;i++){uint64_t h=0;for(size_t c=0;c<nStr;c++)h=HB(d.strCols[c][i].data(),d.strCols[c][i].size(),h);for(size_t c=0;c<nInt;c++)h=HC(h,d.intCols[c][i]);bh[i]=int64_t(h);}
    std::vector<int64_t> bv(nKeys); for(size_t i=0;i<nKeys;i++)bv[i]=i%1000;
    size_t nH=size_t(nProbe*sel),nM=nProbe-nH;
    std::vector<std::vector<std::vector<uint8_t>>> ps(nStr); std::vector<std::vector<int64_t>> pi(nInt);
    for(auto&v:ps)v.reserve(nProbe); for(auto&v:pi)v.reserve(nProbe);
    std::vector<int64_t> ph; ph.reserve(nProbe);
    std::uniform_int_distribution<size_t> kd(0,nKeys-1);
    for(size_t i=0;i<nH;i++){size_t idx=rng()%nKeys;for(size_t c=0;c<nStr;c++)ps[c].push_back(d.strCols[c][idx]);for(size_t c=0;c<nInt;c++)pi[c].push_back(d.intCols[c][idx]);ph.push_back(bh[idx]);}
    for(size_t i=0;i<nM;i++){uint64_t h=0;for(size_t c=0;c<nStr;c++){auto s="miss_"+std::to_string(i)+"_"+std::to_string(c);std::vector<uint8_t>b(s.begin(),s.end());h=HB(b.data(),b.size(),h);ps[c].push_back(std::move(b));}for(size_t c=0;c<nInt;c++){int64_t v=int64_t(nKeys+1)*200+i*31+c;h=HC(h,v);pi[c].push_back(v);}ph.push_back(int64_t(h));}
    std::vector<size_t> ord(nProbe); std::iota(ord.begin(),ord.end(),0);
    for(size_t i=nProbe-1;i>0;i--){std::swap(ord[i],ord[rng()%(i+1)]);}
    for(size_t c=0;c<nStr;c++){auto tmp=std::move(ps[c]);ps[c].resize(nProbe);for(size_t i=0;i<nProbe;i++)ps[c][i]=std::move(tmp[ord[i]]);}
    for(size_t c=0;c<nInt;c++){auto tmp=pi[c];pi[c].resize(nProbe);for(size_t i=0;i<nProbe;i++)pi[c][i]=tmp[ord[i]];}
    {auto tmp=ph;for(size_t i=0;i<nProbe;i++)ph[i]=tmp[ord[i]];}
    std::vector<int64_t> pv(nProbe); for(size_t i=0;i<nProbe;i++)pv[i]=i%1000;
    d.totalRows=nKeys+nProbe;
    for(size_t c=0;c<nStr;c++){d.strCols[c].reserve(d.totalRows);for(auto&v:ps[c])d.strCols[c].push_back(std::move(v));}
    for(size_t c=0;c<nInt;c++){d.intCols[c].insert(d.intCols[c].end(),pi[c].begin(),pi[c].end());}
    d.hashes=bh; d.hashes.insert(d.hashes.end(),ph.begin(),ph.end());
    d.values=bv; d.values.insert(d.values.end(),pv.begin(),pv.end());
    d.strPtrs.resize(nStr); d.strLens.resize(nStr);
    for(size_t c=0;c<nStr;c++){d.strPtrs[c].resize(d.totalRows);d.strLens[c].resize(d.totalRows);for(size_t i=0;i<d.totalRows;i++){d.strPtrs[c][i]=d.strCols[c][i].data();d.strLens[c][i]=d.strCols[c][i].size();}}
    return d;
}

static void RunTaper(const BenchData& d, size_t numChunks) {
    constexpr size_t BATCH_SIZE = 410;
    taper::SimpleArenaAllocator pool;
    std::vector<taper::ColumnDesc> cd; for(size_t c=0;c<d.nStr;c++)cd.push_back(taper::ColumnDesc::Varchar); for(size_t c=0;c<d.nInt;c++)cd.push_back(taper::ColumnDesc::Int64);
    taper::TaperColumnSerializeHandler t(pool, 8, cd, numChunks);

    size_t totalRows = d.totalRows;
    size_t numBatches = (totalRows + BATCH_SIZE - 1) / BATCH_SIZE;

    for (size_t batch = 0; batch < numBatches; batch++) {
        size_t start = batch * BATCH_SIZE;
        size_t end = std::min(start + BATCH_SIZE, totalRows);
        int32_t batchLen = static_cast<int32_t>(end - start);

        std::vector<taper::ColumnInput> cols;
        for (size_t c = 0; c < d.nStr; c++)
            cols.push_back(taper::ColumnInput::MakeVarchar(d.strPtrs[c].data() + start, d.strLens[c].data() + start));
        for (size_t c = 0; c < d.nInt; c++)
            cols.push_back(taper::ColumnInput::MakeInt64(d.intCols[c].data() + start));

        t.EmplaceTableWithDecode(d.hashes.data() + start, batchLen, cols, d.values.data() + start);
    }
    benchmark::DoNotOptimize(t.NumGroups());
}

struct Cfg{const char*n;size_t ns,ni,ht;double lf,sel;};
static std::vector<Cfg> MkCfg(){
    std::vector<Cfg> r;
    struct KT{const char*n;size_t s,i;};
    KT kts[]={{"4str_0int",4,0},{"3str_1int",3,1},{"2str_2int",2,2},{"1str_3int",1,3},{"0str_4int",0,4}};
    for(auto&kt:kts)for(auto ht:{16384,65536,262144,1048576})for(auto lf:{0.5,0.75})for(auto sel:{0.1,0.3,0.5,0.7,0.9})
        r.push_back({kt.n,kt.s,kt.i,(size_t)ht,lf,sel});
    return r;
}

static std::unordered_map<size_t,BenchData> dataCache;
static const BenchData& GetData(size_t idx){
    auto it=dataCache.find(idx);if(it!=dataCache.end())return it->second;
    auto cfgs=MkCfg();auto&c=cfgs[idx];
    dataCache[idx]=GenData(c.ns,c.ni,size_t(c.ht*c.lf),1000000,c.sel);
    return dataCache[idx];
}

static void BM_Taper(benchmark::State& st){
    auto cfgs=MkCfg();auto&c=cfgs[st.range(0)];auto&d=GetData(st.range(0));
    // Compute num_chunks same as Rust: distinct_keys / 0.85 / 8, power-of-2
    size_t numKeys = static_cast<size_t>(c.ht * c.lf);
    size_t numMisses = 1000000 - static_cast<size_t>(1000000 * c.sel);
    size_t distinctKeys = numKeys + numMisses;
    size_t minSlots = std::max(static_cast<size_t>(distinctKeys / 0.85), size_t(8));
    size_t numChunks = 1; while(numChunks * 8 < minSlots) numChunks *= 2;
    for(auto _:st)RunTaper(d,numChunks);
    st.SetItemsProcessed(st.iterations()*d.totalRows);
}

int main(int argc,char**argv){
    auto cfgs=MkCfg();
    for(size_t i=0;i<cfgs.size();i++){
        auto&c=cfgs[i];
        std::string name=std::string("taper/")+c.n+"/ht="+std::to_string(c.ht)+"/lf="+std::to_string(c.lf).substr(0,4)+"/sel="+std::to_string(c.sel).substr(0,3);
        benchmark::RegisterBenchmark(name.c_str(),BM_Taper)->Arg(i)->Iterations(10);
    }
    benchmark::Initialize(&argc,argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
