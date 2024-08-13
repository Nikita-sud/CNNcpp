#include <metal_stdlib>
using namespace metal;

kernel void convolve(texture2d<float, access::read> input [[texture(0)]],
                     constant float* filter [[buffer(0)]],
                     texture2d<float, access::write> output [[texture(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    float result = 0.0;
    int filterSize = 3; // размер фильтра

    int2 size = int2(input.get_width(), input.get_height());

    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            int2 coord = int2(gid.x + i - 1, gid.y + j - 1);
            if (coord.x >= 0 && coord.y >= 0 && coord.x < size.x && coord.y < size.y) {
                result += input.read(coord).r * filter[i * filterSize + j];
            }
        }
    }

    output.write(float4(result), gid);
}