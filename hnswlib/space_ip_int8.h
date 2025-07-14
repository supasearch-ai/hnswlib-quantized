#pragma once
#include "hnswlib.h"
#include <cstdint>

namespace hnswlib {

static float InnerProductInt8(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const int8_t *v1 = reinterpret_cast<const int8_t *>(pVect1v);
    const int8_t *v2 = reinterpret_cast<const int8_t *>(pVect2v);
    size_t dim = *(reinterpret_cast<const size_t *>(qty_ptr));
    
    // Extract scales stored after int8 arrays
    const float *s1_ptr = reinterpret_cast<const float *>(v1 + dim);
    const float *s2_ptr = reinterpret_cast<const float *>(v2 + dim);
    float s1 = *s1_ptr, s2 = *s2_ptr;
    
    // Compute dot product in int32 to avoid overflow
    int32_t dot = 0;
    for (size_t i = 0; i < dim; ++i) {
        int32_t a = static_cast<int32_t>(v1[i]);
        int32_t b = static_cast<int32_t>(v2[i]);
        dot += a * b;
    }
    
    // Apply scale factors and return distance (1.0 - inner_product)
    float ip = s1 * s2 * dot;
    return 1.0f - ip;
}

class InnerProductSpaceInt8 : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_, dim_;
    
public:
    explicit InnerProductSpaceInt8(size_t dim) : dim_(dim) {
        fstdistfunc_ = InnerProductInt8;
        data_size_ = dim * sizeof(int8_t) + sizeof(float);
    }
    
    size_t get_data_size() override { 
        return data_size_; 
    }
    
    DISTFUNC<float> get_dist_func() override { 
        return fstdistfunc_; 
    }
    
    void *get_dist_func_param() override { 
        return &dim_; 
    }
    
    ~InnerProductSpaceInt8() {}
};

}  // namespace hnswlib