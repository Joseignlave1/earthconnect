// Research adapter; the codec and its index are the authors' unmodified library.
#include <k2_raster.hpp>
#include <k2_raster_heuristic.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
using Pair = std::pair<uint64_t, uint64_t>;
struct Query { uint64_t r0, r1, c0, c1; int lo, hi; };
static volatile size_t sink = 0;

template<class F> double measure(F f) {
    std::vector<double> times;
    for (int b=0; b<7; ++b) {
        auto start=Clock::now();
        for (int i=0; i<200; ++i) sink += f();
        times.push_back(std::chrono::duration<double, std::micro>(Clock::now()-start).count()/200);
    }
    std::sort(times.begin(),times.end());
    return times[3];
}

template<class Raster> void run(const std::vector<int>& raw, size_t rows, size_t cols,
                               const std::string& output, const std::vector<Query>& qs) {
    auto start=Clock::now();
    // k=2 at every level. Heuristic variant has one vocabulary/plain level.
    Raster built(raw, rows, cols, 2, 2, 0, 1);
    double build_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    if (!sdsl::store_to_file(built,output)) throw std::runtime_error("serialize failed");
    Raster tree;
    if (!sdsl::load_from_file(tree,output)) throw std::runtime_error("reload failed");
    for (size_t r=0;r<rows;++r) for(size_t c=0;c<cols;++c)
        if (tree.get_cell(r,c)!=raw[r*cols+c]) throw std::runtime_error("pixel mismatch");
    std::vector<int> window;
    tree.get_values_window(7,19,11,30,window);
    for(size_t r=7;r<=19;++r) for(size_t c=11;c<=30;++c)
        if(window[(r-7)*20+c-11]!=raw[r*cols+c]) throw std::runtime_error("window mismatch");
    std::cout << "{\"bytes\":" << sdsl::size_in_bytes(tree)
              << ",\"build_ms\":" << build_ms << ",\"pixels_verified\":" << raw.size()
              << ",\"window_pixels_verified\":" << window.size() << ",\"queries\":[";
    bool first=true;
    for(const auto& q:qs) {
        auto indexed=[&]() {
            std::vector<Pair> hits;
            tree.get_cells_by_value(q.r0,q.r1,q.c0,q.c1,q.lo,q.hi,hits);
            return hits;
        };
        auto scan=[&]() {
            std::vector<Pair> hits;
            for(size_t r=q.r0;r<=q.r1;++r) for(size_t c=q.c0;c<=q.c1;++c)
                if(raw[r*cols+c]>=q.lo && raw[r*cols+c]<=q.hi) hits.emplace_back(r,c);
            return hits;
        };
        auto hits=indexed(); auto expected=scan();
        std::sort(hits.begin(),hits.end());
        if(hits!=expected) throw std::runtime_error("range query mismatch");
        double indexed_us=measure([&](){return indexed().size();});
        double scan_us=measure([&](){return scan().size();});
        if(!first) std::cout << ','; first=false;
        std::cout << "{\"count\":" << hits.size() << ",\"indexed_us\":" << indexed_us
                  << ",\"scan_us\":" << scan_us << ",\"cells\":[";
        for(size_t i=0;i<hits.size();++i) {
            if(i) std::cout << ',';
            std::cout << hits[i].first*cols+hits[i].second;
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
}

int main(int argc,char** argv) {
    try {
        if(argc!=7) throw std::runtime_error("raw32 rows cols output queries type(1|11)");
        size_t rows=std::stoul(argv[2]),cols=std::stoul(argv[3]);
        if(rows<31||cols<31) throw std::runtime_error("demo window requires >=31 rows/cols");
        std::vector<int> raw(rows*cols);
        std::ifstream input(argv[1],std::ios::binary);
        if(!input.read(reinterpret_cast<char*>(raw.data()),raw.size()*sizeof(int)))
            throw std::runtime_error("incomplete input");
        std::vector<Query> qs; Query q; std::ifstream queries(argv[5]);
        if(!queries) throw std::runtime_error("queries missing");
        while(queries>>q.r0>>q.r1>>q.c0>>q.c1>>q.lo>>q.hi) {
            if(q.r0>q.r1||q.r1>=rows||q.c0>q.c1||q.c1>=cols||q.lo>q.hi)
                throw std::runtime_error("invalid query");
            qs.push_back(q);
        }
        if(std::string(argv[6])=="1") run<k2raster::k2_raster<>>(raw,rows,cols,argv[4],qs);
        else if(std::string(argv[6])=="11") run<k2raster::k2_raster_heuristic<>>(raw,rows,cols,argv[4],qs);
        else throw std::runtime_error("unknown codec type");
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
