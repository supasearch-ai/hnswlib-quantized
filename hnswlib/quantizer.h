#pragma once
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace hnswlib {

struct QuantizerI8 {
    static inline float encode(const float *src, int8_t *dst, size_t dim) {
        float max_abs = 0.f;
        for (size_t i = 0; i < dim; ++i) {
            float v = std::fabs(src[i]);
            if (v > max_abs) max_abs = v;
        }
        
        if (max_abs == 0.f) {
            std::fill_n(dst, dim, 0);
            return 1.f;
        }
        
        float scale = max_abs / 127.f;
        float inv_scale = 1.f / scale;
        
        for (size_t i = 0; i < dim; ++i) {
            int q = static_cast<int>(std::round(src[i] * inv_scale));
            q = std::max(-128, std::min(127, q));
            dst[i] = static_cast<int8_t>(q);
        }
        
        return scale;
    }
};

}  // namespace hnswlib