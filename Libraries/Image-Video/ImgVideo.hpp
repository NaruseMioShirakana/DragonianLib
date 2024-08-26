/*
* file: ImgVideo.hpp
* info: 图片数据切片类实现
*
* Author: Maplespe(mapleshr@icloud.com)
*
* date: 2023-3-4 Create.
*/
#pragma once
#define IMAGEVIDEOCLASSHEADER namespace DragonianLib {
#define IMAGEVIDEOCLASSEND }
#include "Vector.h"

IMAGEVIDEOCLASSHEADER

class Image
{
public:
	//[0] - width [1] - height [2] - 行切片数
	int shape[3] = { 0,0,4 };
	struct Data
	{
		DragonianLibSTL::Vector<float> rgb;
		DragonianLibSTL::Vector<float> alpha;
	} data;

	Image() = delete; //禁止默认构造
	~Image() = default;
	Image(const Image&) = delete;
	Image(Image&&) = delete;
	Image& operator=(const Image&) = delete;
	Image& operator=(Image&&) = delete;

	/*图像切片器
	* @param input - 输入文件路径
	* @param width - 切片宽度
	* @param height - 切片高度
	* @param len - 上下文长度
	* @param pad - 超出边界区域的填充值
	* @param line - 显示切片网格线
	*
	* @return 如果构造失败将抛出异常 包括但不限于 文件损坏、不存在、参数错误、内存不足
	*/
	Image(const wchar_t* input, int width, int height, int len, float pad = 0.f, bool line = true, const wchar_t* debug_dir = nullptr);

	Image(unsigned char* input, int src_width, int src_height, int width, int height, int len, float pad = 0.f, bool line = true, const wchar_t* debug_dir = nullptr);

	Image(const wchar_t* input, int interp_mode = 7);

	/*合并切片数据并写出
	* @param path - 保存路径
	* @param scale - 缩放倍数
	* @param quality - 图片压缩质量 (0-100) 100为最好
	*
	* @return 异常: std::bad_alloc,std::runtime_error
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
	//原切片尺寸信息
	struct ClipData
	{
		std::pair<int, int> clipSize; //已切片canvas尺寸
		std::pair<int, int> blockSize;//切片尺寸
		int clipLength = 0;			  //切片offset长度
	} m_clip;
	int width_;
	int height_;
	bool T_ = false;
};

void GdiInit();

void GdiClose();

IMAGEVIDEOCLASSEND