#pragma once

#include <cstddef>
#include <vector>
#include <string>

enum class ColorFormat {
    RGBA,
    RGB,
    RG,
    GREYSCALE
};

struct Vector2u {
    unsigned int x, y;
};

struct Vector2i {
    int x, y;
};

struct Vector4f {
    float r, g, b, a;

    Vector4f operator+(const Vector4f& other) const { return {r + other.r, g + other.g, b + other.b, a + other.a}; }
    Vector4f operator-(const Vector4f& other) const { return {r - other.r, g - other.g, b - other.b, a - other.a}; }
    Vector4f operator*(float scalar) const { return {r * scalar, g * scalar, b * scalar, a * scalar}; }
};

struct IntRect {
    int x, y, width, height;
};

/**
* @brief Image is similar to Texture, however the data lives on the processor and RAM. This allows
*  for more control over the data. It has no connection to OpenGL which means it can be loaded before
*  the OpenGL context is created.
*/
class Image {
private:
    // Variables
    std::vector<std::byte> data;

    ColorFormat format;
    Vector2u size;

public:
    // Constructors
    Image(const Vector2u& size = {1, 1}, Vector4f color = {1, 1, 1, 1}, ColorFormat format = ColorFormat::RGBA); // Image constructor from no data
    Image(const Vector2u& size, ColorFormat format, const std::byte* pixelDataArr); // Raw data constructor for raw pixel data
    Image(const std::vector<std::byte>& rawData); // Compressed data constructor, stuff like PNG or JPEG
    Image(const std::string& path); // Load from file
    Image(const Image& other) = default; // Copy constructor
    Image(Image&& other) noexcept = default; // Move constructor
    ~Image() = default; // Destructor

    // Operators
    Image& operator=(const Image& other) = default;
    Image& operator=(Image&& other) noexcept = default;

    // State functions
    /**
        * @brief Loads from compressed texture data.
        * @param arr PNG, JPEG, etc
        */
    void load(const std::vector<std::byte>& arr);

    /**
        * @brief Load from a file
        * @param path 
        */
    void load(const std::string& path);

    /**
        * @brief Save to a file
        * @param path 
        */
    void save(const std::string& path) const;

    // Editing functions
    void mask(const Vector4f& color, float tolerance = 0.f, std::byte alpha = std::byte{ 0x00 });
    void copy(const Image& src, const Vector2i& position, IntRect rect = {0, 0, 0, 0}, bool applyAlpha = false);
    void flip_horizontally();
    void flip_vertically();
    void set_pixel(const Vector2u& position, const Vector4f& color);
    void convert(ColorFormat format);
    
    // Getters
    Vector4f get_pixel(const Vector2u& position) const;
    inline ColorFormat get_format() const { return format; }
    inline const Vector2u& get_size() const { return size; }
    inline size_t get_pixel_count() const { return data.size(); }
    inline const std::byte* c_arr() const { return data.data(); }
    inline bool is_transparent() const { return format == ColorFormat::RGBA; }
};
