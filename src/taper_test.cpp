/// Quick sanity test — run on target aarch64 machine.
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#define XXH_INLINE_ALL
#include "xxhash.h"
#include "column_marshaller.h"

static int64_t HB(const uint8_t* d, size_t l, uint64_t s) { return static_cast<int64_t>(XXH3_64bits_withSeed(d,l,s)); }
static int64_t HC(int64_t seed, int64_t v) { return static_cast<int64_t>(XXH3_64bits_withSeed(&v,8,seed)); }

void TestInt() {
    printf("TestInt... ");
    taper::SimpleArenaAllocator pool;
    std::vector<taper::ColumnDesc> cd={taper::ColumnDesc::Int64, taper::ColumnDesc::Int64};
    taper::TaperColumnSerializeHandler t(pool, 8, cd, 64);
    int64_t c0[]={1,2,3,1,2}, c1[]={10,20,30,10,20}, v[]={100,200,300,400,500};
    int64_t h[5]; for(int i=0;i<5;i++) h[i]=HC(HC(0,c0[i]),c1[i]);
    std::vector<taper::ColumnInput> cols={taper::ColumnInput::MakeInt64(c0),taper::ColumnInput::MakeInt64(c1)};
    t.EmplaceTableWithDecode(h,5,cols,v);
    assert(t.NumGroups()==3); printf("OK\n");
}

void TestMixed() {
    printf("TestMixed... ");
    taper::SimpleArenaAllocator pool;
    std::vector<taper::ColumnDesc> cd={taper::ColumnDesc::Varchar, taper::ColumnDesc::Int64};
    taper::TaperColumnSerializeHandler t(pool, 8, cd, 64);
    std::string ss[]={"alpha","beta","gamma","alpha","beta","delta"};
    int64_t ints[]={1,2,3,1,2,4}, vals[]={10,20,30,40,50,60};
    std::vector<const uint8_t*> sp(6); std::vector<size_t> sl(6);
    for(int i=0;i<6;i++){sp[i]=reinterpret_cast<const uint8_t*>(ss[i].data());sl[i]=ss[i].size();}
    int64_t h[6]; for(int i=0;i<6;i++) h[i]=HC(HB(sp[i],sl[i],0),ints[i]);
    std::vector<taper::ColumnInput> cols={taper::ColumnInput::MakeVarchar(sp.data(),sl.data()),taper::ColumnInput::MakeInt64(ints)};
    t.EmplaceTableWithDecode(h,6,cols,vals);
    assert(t.NumGroups()==4); printf("OK (4 groups)\n");
}

void TestMultiVarchar() {
    printf("TestMultiVarchar... ");
    taper::SimpleArenaAllocator pool;
    std::vector<taper::ColumnDesc> cd={taper::ColumnDesc::Varchar, taper::ColumnDesc::Varchar};
    taper::TaperColumnSerializeHandler t(pool, 8, cd, 64);
    std::string s0[]={"foo","bar","foo","baz"}, s1[]={"X","Y","X","Z"};
    int64_t vals[]={1,2,3,4};
    std::vector<const uint8_t*> p0(4),p1(4); std::vector<size_t> l0(4),l1(4);
    for(int i=0;i<4;i++){p0[i]=(const uint8_t*)s0[i].data();l0[i]=s0[i].size();p1[i]=(const uint8_t*)s1[i].data();l1[i]=s1[i].size();}
    int64_t h[4]; for(int i=0;i<4;i++) h[i]=HB(p1[i],l1[i],HB(p0[i],l0[i],0));
    std::vector<taper::ColumnInput> cols={taper::ColumnInput::MakeVarchar(p0.data(),l0.data()),taper::ColumnInput::MakeVarchar(p1.data(),l1.data())};
    t.EmplaceTableWithDecode(h,4,cols,vals);
    assert(t.NumGroups()==3); printf("OK (3 groups)\n");
}

int main() {
    printf("═══ TaperHashTable Tests ═══\n");
    TestInt(); TestMixed(); TestMultiVarchar();
    printf("═══ All PASSED ═══\n");
}
