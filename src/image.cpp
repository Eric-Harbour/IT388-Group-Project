#include "image.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstring>

// Helper functions
unsigned int color_format_to_bytes(ColorFormat format){
    switch(format){
    case ColorFormat::GREYSCALE: return 1;
    case ColorFormat::RG: return 2;
    case ColorFormat::RGB: return 3;
    case ColorFormat::RGBA: return 4;
    }
    return 0;
}

std::byte float_to_byte(float value){
    // Convert the 0-1 to 0-255
    float clamped = std::clamp(value, 0.f, 1.f);
    unsigned char intermediate = static_cast<unsigned char>(clamped * 255.0f);
    return static_cast<std::byte>(intermediate);
}

float byte_to_float(std::byte byte){
    // Convert the 0-255 to 1-0
    unsigned char val = static_cast<unsigned char>(byte);
    return static_cast<float>(val) / 255.f;
};

// Constructors
Image::Image(const Vector2u& size, Vector4f color, ColorFormat format) : format(format), size(size) {
    // Create the array to hold the image data. Pixel count is equal to the area of the image per channel.
    unsigned int stride = color_format_to_bytes(format);
    size_t pixelCount = static_cast<size_t>(size.x) * size.y * stride;
    data.resize(pixelCount);

    // Values to store into the image data. The color is in a float of 0-1 where the image needs a byte
    std::byte avgColor = float_to_byte((color.r + color.g + color.b) / 3.f); // Average here if being stored in greyscale
    std::byte byteValues[] = { float_to_byte(color.r), float_to_byte(color.g), float_to_byte(color.b), float_to_byte(color.a) };

    // Fill the image with the color data
    for(size_t i = 0; i < pixelCount; i += stride){
        if(stride == 1){
            // Set the data to greyscale if there is only one color per pixel
            data[i] = avgColor;
        } else {
            // Otherwise set the data to RGBA respectively, cutting off when maxxing out the stride.
            // i.e. if there's only 3 channels, omit the alpha.
            for(unsigned int k = 0; k < stride; k++){
                data[i + k] = byteValues[k];
            }
        }
    }
}

Image::Image(const Vector2u& size, ColorFormat format, const std::byte* pixelData) : format(format), size(size) {
    // Create data to hold this data, copy all available
    unsigned int stride = color_format_to_bytes(format);
    size_t pixelCount = static_cast<size_t>(size.x) * size.y * stride;
    data.assign(pixelData, pixelData + pixelCount);
}

Image::Image(const std::vector<std::byte>& rawData) {
    load(rawData);
}

Image::Image(const std::string& path) {
    load(path);
}

// State functions
void Image::load(const std::vector<std::byte>& arr){
    // Load the data from a byte array of PNG, JPEG, or similar
    stbi_set_flip_vertically_on_load(false);

    int width, height, channels;
    unsigned char* pixelData = stbi_load_from_memory(reinterpret_cast<const unsigned char*>(arr.data()), static_cast<int>(arr.size()), &width, &height, &channels, 0);

    if (!pixelData) {
        return;
    }

    // Save new parameters from loaded image
    switch(channels){
    case 1:
        format = ColorFormat::GREYSCALE;
        break;

    case 2:
        format = ColorFormat::RG;
        break;

    case 3:
        format = ColorFormat::RGB;
        break;

    case 4:
        format = ColorFormat::RGBA;
        break;

    default:
        assert(false && "Unhandled channel count when loading image");
    }

    size = { static_cast<unsigned int>(width), static_cast<unsigned int>(height) };
    size_t pixelCount = static_cast<size_t>(width) * height * color_format_to_bytes(format);
    data.assign(reinterpret_cast<std::byte*>(pixelData), reinterpret_cast<std::byte*>(pixelData) + pixelCount);

    // Remove the old pixel data to prevent memory leaks
    stbi_image_free(pixelData);
}

void Image::load(const std::string& path){
    int width, height, channels;
    stbi_set_flip_vertically_on_load(false);
    unsigned char* pixelData = stbi_load(path.c_str(), &width, &height, &channels, 0);

    if (!pixelData) {
        return;
    }

    switch(channels){
    case 1: format = ColorFormat::GREYSCALE; break;
    case 2: format = ColorFormat::RG; break;
    case 3: format = ColorFormat::RGB; break;
    case 4: format = ColorFormat::RGBA; break;
    default: assert(false && "Unhandled channel count when loading image");
    }

    size = { static_cast<unsigned int>(width), static_cast<unsigned int>(height) };
    size_t pixelCount = static_cast<size_t>(width) * height * color_format_to_bytes(format);
    data.assign(reinterpret_cast<std::byte*>(pixelData), reinterpret_cast<std::byte*>(pixelData) + pixelCount);

    stbi_image_free(pixelData);
}

void Image::save(const std::string& path) const {
    // Save this image to a text file. STB takes chars instead of bytes so cast it. and save it.
    stbi_write_png(path.c_str(), static_cast<int>(size.x), static_cast<int>(size.y), static_cast<int>(color_format_to_bytes(format)), data.data(), 0);
}

// Manipulation functions
void Image::convert(ColorFormat format){
    // Converts to the specified format, if data is missing just fill it in with 255
    Image target(get_size(), {1, 1, 1, 1}, format);
    target.copy(*this, {0, 0});
    *this = std::move(target);
}

void Image::mask(const Vector4f& color, float tolerance, std::byte alpha){
    // This lambda checks if the area is actually masked by the given color.
    auto is_masked = [color, tolerance](const Vector4f& value){
        Vector4f upperBound = color + color * tolerance;
        Vector4f lowerBound = color - color * tolerance;
        return value.r >= lowerBound.r && value.g >= lowerBound.g && value.b >= lowerBound.b
            && value.r <= upperBound.r && value.g <= upperBound.g && value.b <= upperBound.b;
    };

    // Run the mask algorithm over the bytes
    unsigned int stride = color_format_to_bytes(format);

    for(size_t i = 0; i < data.size(); i += stride){
        // Get the value of the byte and check if it is masked in order to zero out the data if its not
        bool masked = false;

        if(format == ColorFormat::GREYSCALE){
            // Only check one channel, so do it as a greyscale value
            float value = byte_to_float(data[i]);
            masked = is_masked({ value, value, value, 1.f});
        } else {
            // Otherwise do RGBA, omitting extras as white
            float pixelValues[]{1, 1, 1, 1};

            for(unsigned int k = 0; k < stride; k++){
                pixelValues[k] = byte_to_float(data[i + k]);
            }

            masked = is_masked({ pixelValues[0], pixelValues[1], pixelValues[2], pixelValues[3] });
        }

        // Zero out alpha, if it has alpha, if not masked. If theres no alpha then zero out all color.
        if(!masked){
            if(format == ColorFormat::RGBA){
                data[i + 3] = std::byte{0x0};
            } else {
                for(unsigned int k = 0; k < stride; k++){
                    data[i + k] = std::byte{0x0};
                }
            }
        }
    }
}

void Image::copy(const Image& src, const Vector2i& position, IntRect rect, bool applyAlpha){
    // Copies the source image to this image at position
    unsigned int srcStride = color_format_to_bytes(src.format);
    unsigned int destStride = color_format_to_bytes(format);

    // Check if the provided width or height is zero, if not auto-select the whole image. This is for convenience and nothing else
    if(rect.width <= 0) rect.width = static_cast<int>(src.size.x);
    if(rect.height <= 0) rect.height = static_cast<int>(src.size.y);

    Vector2u start{static_cast<unsigned int>(std::max(0, rect.x)), static_cast<unsigned int>(std::max(0, rect.y))};
    Vector2u end{static_cast<unsigned int>(std::min(static_cast<int>(src.size.x), rect.x + rect.width)), 
                 static_cast<unsigned int>(std::min(static_cast<int>(src.size.y), rect.y + rect.height))};

    // Run the algorithm for copying from the src to this data
    for(unsigned int y = start.y; y < end.y; y++){
        for(unsigned int x = start.x; x < end.x; x++){
            int targetX = position.x + static_cast<int>(x) - rect.x;
            int targetY = position.y + static_cast<int>(y) - rect.y;

            // Skip this pixel if its not on the actual image
            if(targetX >= static_cast<int>(size.x) || targetX < 0 || targetY >= static_cast<int>(size.y) || targetY < 0)
                continue;

            // Set each pixel from the src to the destination.
            for(unsigned int i = 0; i < destStride; i++){
                // Obtain the indices for each the source and the target.
                size_t srcIndex = (static_cast<size_t>(x) + static_cast<size_t>(y) * src.size.x) * srcStride + i;
                size_t targetIndex = (static_cast<size_t>(targetX) + static_cast<size_t>(targetY) * size.x) * destStride + i;

                // Make sure the index doesnt over-extend the bytes
                if(i < srcStride){
                    // Final branch to check if it needs to apply alpha values to it or not. Useful for transparent copying
                    if(applyAlpha && src.format == ColorFormat::RGBA && i != 3){
                        float scalar = byte_to_float(src.data[(static_cast<size_t>(x) + static_cast<size_t>(y) * src.size.x) * srcStride + 3]); // Alpha byte to float
                        data[targetIndex] = float_to_byte(byte_to_float(src.data[srcIndex]) * scalar);
                    } else {
                        data[targetIndex] = src.data[srcIndex];
                    }
                } else {
                    data[targetIndex] = std::byte(0xFF);
                }
            }
        }
    }
}

void Image::flip_horizontally(){
    unsigned int stride = color_format_to_bytes(format);
    for(unsigned int y = 0; y < size.y; y++){
        for(unsigned int x = 0; x < size.x / 2; x++){
            size_t index1 = (static_cast<size_t>(x) + static_cast<size_t>(y) * size.x) * stride;
            size_t index2 = (static_cast<size_t>(size.x) - x - 1 + static_cast<size_t>(y) * size.x) * stride;

            for(unsigned int i = 0; i < stride; i++){
                std::swap(data[index1 + i], data[index2 + i]);
            }
        }
    }
}

void Image::flip_vertically(){
    unsigned int stride = color_format_to_bytes(format);
    for(unsigned int x = 0; x < size.x; x++){
        for(unsigned int y = 0; y < size.y / 2; y++){
            size_t index1 = (static_cast<size_t>(x) + static_cast<size_t>(y) * size.x) * stride;
            size_t index2 = (static_cast<size_t>(x) + (static_cast<size_t>(size.y) - y - 1) * size.x) * stride;

            for(unsigned int i = 0; i < stride; i++){
                std::swap(data[index1 + i], data[index2 + i]);
            }
        }
    }
}

void Image::set_pixel(const Vector2u& position, const Vector4f& color){
    unsigned int stride = color_format_to_bytes(format);
    size_t index = (static_cast<size_t>(position.x) + static_cast<size_t>(position.y) * size.x) * stride;

    assert(index < data.size() && "Out of bounds");

    float colorData[4]{color.r, color.g, color.b, color.a};

    if(format == ColorFormat::GREYSCALE){
        data[index] = float_to_byte((color.r + color.g + color.b) / 3.f * color.a);
    } else {
        for(unsigned int k = 0; k < stride; k++){
            data[index + k] = float_to_byte(colorData[k]);
        }
    }
}

Vector4f Image::get_pixel(const Vector2u& position) const {
    unsigned int stride = color_format_to_bytes(format);
    size_t index = (static_cast<size_t>(position.x) + static_cast<size_t>(position.y) * size.x) * stride;

    assert(index < data.size() && "Out of bounds");

    float colorData[4]{1.f, 1.f, 1.f, 1.f};

    if(format == ColorFormat::GREYSCALE){
        float val = byte_to_float(data[index]);
        colorData[0] = val;
        colorData[1] = val;
        colorData[2] = val;
    } else {
        for(unsigned int k = 0; k < stride; k++){
            colorData[k] = byte_to_float(data[index + k]);
        }
    }

    return {colorData[0], colorData[1], colorData[2], colorData[3]};
}
