#pragma once

#include <complex>
#include <cstdint>
#include <cmath>

// 数学常量
constexpr double PI = 3.14159265358979323846;

// 通用常量
constexpr int W_MAX = 256;
constexpr int H_MAX = 256;
constexpr int DEFAULT_W = 128;
constexpr int DEFAULT_H = 128;
constexpr int DEFAULT_SCR_W = 640 * 2;
constexpr int DEFAULT_SCR_H = 640 * 2;

// 坐标结构体
struct coordI {
    int x, y;
    
    coordI(int x = 0, int y = 0) : x(x), y(y) {}
    
    coordI operator+(const coordI& other) const {
        return coordI(x + other.x, y + other.y);
    }
    
    coordI operator-(const coordI& other) const {
        return coordI(x - other.x, y - other.y);
    }
    
    bool operator==(const coordI& other) const {
        return x == other.x && y == other.y;
    }
    
    bool operator!=(const coordI& other) const {
        return !(*this == other);
    }
};

struct coordD {
    double x, y;
    
    coordD(double x = 0.0, double y = 0.0) : x(x), y(y) {}
    
    coordD operator+(const coordD& other) const {
        return coordD(x + other.x, y + other.y);
    }
    
    coordD operator-(const coordD& other) const {
        return coordD(x - other.x, y - other.y);
    }
    
    coordD operator*(double scalar) const {
        return coordD(x * scalar, y * scalar);
    }
    
    coordD operator/(double scalar) const {
        return coordD(x / scalar, y / scalar);
    }
};

// 相机结构体
struct camera {
    double x, y, sk;  // 位置(x,y)和缩放(sk)
    
    camera(double x = 0.5, double y = 0.5, double sk = 1.0) : x(x), y(y), sk(sk) {}
    
    void reset() {
        x = 0.5;
        y = 0.5;
        sk = 1.0;
    }
};

// 像素颜色结构体（用于2d-renderer兼容性）
struct Pixel {
    float r, g, b, a;          // 颜色和透明度
    float lr, lg, lb;          // 发光颜色
    
    Pixel(
        float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 0.0f,
        float lr = 0.0f, float lg = 0.0f, float lb = 0.0f
    ) : r(r), g(g), b(b), a(a), lr(lr), lg(lg), lb(lb) {}
};

// 复数类型别名
using Complex = std::complex<double>;

// 工具函数声明
namespace utils {
    // 坐标转换函数
    coordD mem2abs(coordD p, int W, int H);
    coordD abs2mem(coordD p, int W, int H);
    coordD abs2scr(coordD p, const camera& cam, int scrW, int scrH);
    coordD scr2abs(coordD p, const camera& cam, int scrW, int scrH);
    coordI mem2scr(coordI p, const camera& cam, int W, int H, int scrW, int scrH);
    coordI scr2mem(coordI p, const camera& cam, int W, int H, int scrW, int scrH);
    
    // 数值工具
    inline double clamp(double value, double min, double max) {
        return (value < min) ? min : (value > max) ? max : value;
    }
    
    inline int clamp(int value, int min, int max) {
        return (value < min) ? min : (value > max) ? max : value;
    }
    
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    // 颜色转换：ARGB uint32_t 到 分量
    inline void argbToComponents(uint32_t argb, uint8_t& a, uint8_t& r, uint8_t& g, uint8_t& b) {
        a = (argb >> 24) & 0xFF;
        r = (argb >> 16) & 0xFF;
        g = (argb >> 8) & 0xFF;
        b = argb & 0xFF;
    }
    
    inline uint32_t componentsToArgb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
        return (static_cast<uint32_t>(a) << 24) |
               (static_cast<uint32_t>(r) << 16) |
               (static_cast<uint32_t>(g) << 8) |
               static_cast<uint32_t>(b);
    }
    
    // 颜色转换：ARGB uint32_t 到 normalized float
    inline void argbToFloat(uint32_t argb, float& a, float& r, float& g, float& b) {
        uint8_t a8, r8, g8, b8;
        argbToComponents(argb, a8, r8, g8, b8);
        a = a8 / 255.0f;
        r = r8 / 255.0f;
        g = g8 / 255.0f;
        b = b8 / 255.0f;
    }
}