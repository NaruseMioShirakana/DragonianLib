/*
* file: ImgVideo.hpp
* info: Image data slicing class implementation
*
* Author: Maplespe(mapleshr@icloud.com)
*
* date: 2023-3-4 Create.
*/
#pragma once
#define _D_Dragonian_Lib_Image_Video_Header namespace DragonianLib { namespace ImageVideo {
#define _D_Dragonian_Lib_Image_Video_End } }
#include "MyTemplateLibrary/Vector.h"

_D_Dragonian_Lib_Image_Video_Header

/**
 * @class Image
 * @brief Image data slicing class
 */
class Image
{
public:
    //[0] - width [1] - height [2] - number of row slices
    int shape[3] = { 0,0,4 };
    struct Data
    {
        DragonianLibSTL::Vector<float> rgb;
        DragonianLibSTL::Vector<float> alpha;
    } data;

    Image() = delete; // Disable default constructor
    ~Image() = default;
    Image(const Image&) = delete;
    Image(Image&&) = delete;
    Image& operator=(const Image&) = delete;
    Image& operator=(Image&&) = delete;

    /**
    * @brief Image slicer
    * @param input - Input file path
    * @param width - Slice width
    * @param height - Slice height
    * @param len - Context length
    * @param pad - Padding value for out-of-bounds areas
    * @param line - Display slice grid lines
    * @param debug_dir - Debug directory
    *
    * @return Throws an exception if construction fails, including but not limited to file corruption, non-existence, parameter errors, insufficient memory
    */
    Image(const wchar_t* input, int width, int height, int len, float pad = 0.f, bool line = true, const wchar_t* debug_dir = nullptr);

    Image(unsigned char* input, int src_width, int src_height, int width, int height, int len, float pad = 0.f, bool line = true, const wchar_t* debug_dir = nullptr);

    Image(const wchar_t* input, int interp_mode = 7);

    /**
    * @brief Merge slice data and write out
    * @param path - Save path
    * @param scale - Scaling factor
    * @param quality - Image compression quality (0-100), 100 being the best
    *
    * @return Exceptions: std::bad_alloc, std::runtime_error
    */
    bool MergeWrite(const wchar_t* path, int scale, unsigned int quality = 100) const;

    int GetWidth() const
    {
        return width_;
    }

    int GetHeight() const
    {
        return height_;
    }

    void Transpose(size_t scale = 1);

    void TransposeBGR(size_t scale = 1);

private:
    // Original slice size information
    struct ClipData
    {
        std::pair<int, int> clipSize; // Sliced canvas size
        std::pair<int, int> blockSize; // Slice size
        int clipLength = 0; // Slice offset length
    } m_clip;
    int width_;
    int height_;
    bool T_ = false;
};

/**
 * @brief Initialize GDI+
 */
void GdiInit();

/**
 * @brief Close GDI+
 */
void GdiClose();

_D_Dragonian_Lib_Image_Video_End
