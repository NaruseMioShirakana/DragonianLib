/*
* file: ImgVideo.cpp
* info: 图片数据切片类实现
*
* Author: Maplespe(mapleshr@icloud.com)
*
* date: 2023-3-4 Create.
*/
#include "ImgVideo.hpp"
//Gdiplus
#ifdef _WIN32
#include <comdef.h>
#include <gdiplus.h>
#pragma comment(lib, "Msimg32.lib")
#pragma comment(lib, "gdiplus.lib")  
#endif
#include <stdexcept>

IMAGEVIDEOCLASSHEADER

void GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT num = 0;           // number of image encoders
	UINT size = 0;          // size of the image encoder array in bytes

	Gdiplus::GetImageEncodersSize(&num, &size);
	if (size == 0) {
		return;
	}

	Gdiplus::ImageCodecInfo* pImageCodecInfo = static_cast<Gdiplus::ImageCodecInfo*>(malloc(size));
	if (pImageCodecInfo == nullptr) {
		return;
	}

	Gdiplus::GetImageEncoders(num, size, pImageCodecInfo);
	for (UINT i = 0; i < num; ++i) {
		if (wcscmp(pImageCodecInfo[i].MimeType, format) == 0) {
			*pClsid = pImageCodecInfo[i].Clsid;
			break;
		}
	}

	free(pImageCodecInfo);
}

//quality - 压缩质量 (0-100), 100 表示最高质量
bool SaveBitmapToPNG(Gdiplus::Bitmap* bitmap, const WCHAR* filename, UINT quality = 100)
{
	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	Gdiplus::EncoderParameters encoderParams{};
	encoderParams.Count = 1;
	encoderParams.Parameter[0].Guid = Gdiplus::EncoderQuality;
	encoderParams.Parameter[0].Type = Gdiplus::EncoderParameterValueTypeLong;
	encoderParams.Parameter[0].NumberOfValues = 1;
	encoderParams.Parameter[0].Value = &quality;

	bitmap->Save(filename, &pngClsid, &encoderParams);

	return true;
}

void DrawRectangle(HDC hdc, int x, int y, int width, int height)
{
	MoveToEx(hdc, x, y, nullptr);
	LineTo(hdc, x + width, y);
	LineTo(hdc, x + width, y + height);
	LineTo(hdc, x, y + height);
	LineTo(hdc, x, y);
}

ImageSlicer::ImageSlicer(const wchar_t* input, const int width, const int height,
	const int len, const float pad, bool line) : width_(width), height_(height)
{
	//加载图像
	Gdiplus::Bitmap* bmp = Gdiplus::Bitmap::FromFile(input);
	if (!bmp) throw std::runtime_error("image load failed!");

	shape[0] = (int)bmp->GetWidth();
	shape[1] = (int)bmp->GetHeight();

	//切片数
	int clipCountX = static_cast<int>(ceilf((float)shape[0] / float(width - len * 2)));
	int clipCountY = static_cast<int>(ceilf((float)shape[1] / float(height - len * 2)));
	//切片后的总宽高
	int clipWidth = clipCountX * width;
	int clipHeight = clipCountY * height;
	//offset
	clipCountX = static_cast<int>(ceilf((float)clipWidth / (float)width));
	clipCountY = static_cast<int>(ceilf((float)clipHeight / (float)height));

	shape[2] = clipCountX;

	//创建切片画布
	Gdiplus::Bitmap* canvas = new Gdiplus::Bitmap(clipWidth, clipHeight, PixelFormat32bppARGB);
	Gdiplus::Graphics dw(canvas);

	BYTE p = static_cast<BYTE>(pad * 255.f);
	dw.Clear(Gdiplus::Color(p, p, p, p));

	//转为GDI操作 GDIPlus太太太慢了
	HDC canvasDC = dw.GetHDC();

	HDC compDC = CreateCompatibleDC(canvasDC);
	HBITMAP hbmp = nullptr;
	bmp->GetHBITMAP(Gdiplus::Color::Transparent, &hbmp);
	SelectObject(compDC, hbmp);
	auto DrawImage = [&canvasDC, &compDC, this](int x, int y, int srcx, int srcy, int w, int h)
	{
		//const Gdiplus::Rect dst(x, y, w, h);
		//dw.DrawImage(bmp, dst, srcx, srcy, w, h, Gdiplus::UnitPixel);
		if(srcx + w > shape[0])
			w = shape[0] - srcx;
		if (srcy + h > shape[1])
			h = shape[1] - srcy;

		BLENDFUNCTION blend = { AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };
		AlphaBlend(canvasDC, x, y, w, h, compDC, srcx, srcy, w, h, blend);
	};

	//显示网格线
	HPEN pen = nullptr;
	HPEN dstPen = nullptr;
	if (line)
	{
		pen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));
		dstPen = CreatePen(PS_SOLID, 2, RGB(0, 255, 0));
		SelectObject(canvasDC, pen);
	}

	int dstY = 0;
	int srcY = 0;
	for (int y = 0; y < clipCountY; y++)
	{
		int dstX = 0;
		int srcX = 0;
		for (int x = 0; x < clipCountX; x++)
		{
			if (y != 0 && len != 0)
			{
				//绘制 Y offset 部分
				DrawImage(x != 0 ? dstX + len : dstX, dstY, srcX, srcY - len, width, len);
			}
			if (x != 0 && len != 0)
			{
				//绘制 X offset 部分
				DrawImage(dstX, dstY, srcX - len, y != 0 ? srcY - len : srcY, len, height);

				const int _x = dstX + len;
				const int _y = y != 0 ? dstY + len : dstY;
				const int _w = width - len * 2;
				const int _h = y != 0 ? height - len * 2 : height - len;
				DrawImage(_x,
					_y,
					srcX, srcY,
					width - len,
					y != 0 ? height - len : height
				);
				if (line)
				{
					SelectObject(canvasDC, dstPen);
					DrawRectangle(canvasDC, _x, _y, _w, _h);
				}
				srcX += _w;
			}
			else
			{
				const int _x = len != 0 ? 0 : dstX;
				const int _y = y != 0 ? dstY + len : dstY;
				const int _w = width - len;
				const int _h = y != 0 ? height - len * 2 : height - len;
				DrawImage(_x, _y, srcX, srcY, _w + len, _h + len);
				if (line)
				{
					SelectObject(canvasDC, dstPen);
					DrawRectangle(canvasDC, _x, _y, _w, _h);
				}
				srcX += _w;
			}
			if (line)
			{
				SelectObject(canvasDC, pen);
				DrawRectangle(canvasDC, dstX, dstY, dstX + width, dstY + height);
			}
			dstX += width;
		}
		dstY += height;
		srcY += height - len * 2;
	}
	dw.ReleaseHDC(canvasDC);
	DeleteObject(hbmp);
	DeleteDC(compDC);
	if (pen) DeleteObject(pen);
	if (dstPen) DeleteObject(dstPen);
	delete bmp;

	//读取到vector
	size_t pixelSize = size_t(clipWidth) * clipHeight;
	data.rgb.Reserve(pixelSize * 3);
	data.alpha.Reserve(pixelSize);

	Gdiplus::Rect lockRect(0, 0, clipWidth, clipHeight);
	Gdiplus::BitmapData lockData{};
	if(canvas->LockBits(&lockRect, Gdiplus::ImageLockModeRead, PixelFormat32bppARGB, &lockData) != Gdiplus::Status::Ok)
	{
		delete canvas;
		throw std::runtime_error("LockBits failed!");
	}

	//按切片存储
	auto ReadImage = [&](int x, int y)
	{
		for (int _y = 0; _y < width; _y++)
		{
			for (int _x = 0; _x < height; _x++)
			{
				BYTE* ptr = static_cast<BYTE*>(lockData.Scan0) + ptrdiff_t(lockData.Stride * (y + _y) + (x + _x) * 4);

				data.alpha.EmplaceBack((float)ptr[3] / 255.f);//A
				data.rgb.EmplaceBack((float)ptr[2] / 255.f);  //R
				data.rgb.EmplaceBack((float)ptr[1] / 255.f);  //G
				data.rgb.EmplaceBack((float)ptr[0] / 255.f);  //B
			}
		}
	};

	dstY = 0;
	for (int y = 0; y < clipCountY; y++)
	{
		int dstX = 0;
		for (int x = 0; x < clipCountX; x++)
		{
			ReadImage(dstX, dstY);
			dstX += width;
		}
		dstY += height;
	}

	canvas->UnlockBits(&lockData);

	m_clip.clipSize = std::make_pair(clipWidth, clipHeight);
	m_clip.blockSize = std::make_pair(width, height);
	m_clip.clipLength = len;

	//SaveBitmapToPNG(canvas, L"E:\\testclip.png");

	delete canvas;
}

// 使用压缩比较高的格式从_RGB和_Alpha保存图像
bool ImageSlicer::MergeWrite(const wchar_t* path, int scale, UINT quality) const
{
	//缩放切片
	int newWidth = m_clip.clipSize.first * scale;
	int newHeight = m_clip.clipSize.second * scale;
	int newLength = m_clip.clipLength * scale;
	int newClipW = m_clip.blockSize.first * scale;
	int newClipH = m_clip.blockSize.second * scale;
	int srcWidth = shape[0] * scale;
	int srcHeight = shape[1] * scale;

	int clipCountX = newWidth / newClipW;
	int clipCountY = newHeight / newClipH;

	//检查像素是否匹配
	size_t alphaSize = size_t(newWidth) * newHeight;
	size_t pixelSize = alphaSize * 3;
	if (pixelSize != data.rgb.Size() || alphaSize != data.alpha.Size())
		return false;

	//创建canvas
	Gdiplus::Bitmap* canvas = new Gdiplus::Bitmap(srcWidth, srcHeight, PixelFormat32bppARGB);
	//填充数据
	Gdiplus::Rect lockRect(0, 0, srcWidth, srcHeight);
	Gdiplus::BitmapData lockData{};
	if (canvas->LockBits(&lockRect, Gdiplus::ImageLockModeWrite, PixelFormat32bppARGB, &lockData) != Gdiplus::Status::Ok)
	{
		delete canvas;
		throw std::runtime_error("LockBits failed!");
	}

	auto WriteImage = [&](int x, int y, int srcx, int srcy, int w, int h, int blockX, int blockY)
	{
		if (x + w > srcWidth)
			w = srcWidth - x;
		if (x + h > srcHeight)
			h = srcHeight - h;

		srcx -= blockX * newClipW;
		srcy -= blockY * newClipH;

		//block偏移
		const int blockSizeA = newClipW * newClipH;
		const int blockSizeRGB = blockSizeA * 3;
		const int blockOffsetA = (blockY * clipCountX + blockX) * blockSizeA;
		const int blockOffsetRGB = (blockY * clipCountX + blockX) * blockSizeRGB;

		for (int _y = 0; _y < h; _y++)
		{
			for (int _x = 0; _x < w; _x++)
			{
				auto SrcPixel = [&](int index)
				{
					const int offset = blockOffsetRGB + ((srcy + _y) * newClipW + (srcx + _x)) * 3;
					return static_cast<BYTE>(data.rgb[offset + index] * 255.f);
				};
				if (y + _y >= srcHeight || x + _x >= srcWidth) break;

				BYTE* p = static_cast<BYTE*>(lockData.Scan0) + ptrdiff_t(lockData.Stride * (y + _y) + (x + _x) * 4);

				p[0] = SrcPixel(2);//B
				p[1] = SrcPixel(1);//G
				p[2] = SrcPixel(0);//R

				//alpha
				p[3] = BYTE(data.alpha[blockOffsetA + ((srcy + _y) * newClipW + (srcx + _x))] * 255.f);
			}
		}
	};

	//反推切片即可还原
	int dstY = 0;
	int srcY = 0;
	for (int y = 0; y < clipCountY; y++)
	{
		int dstX = 0;
		int srcX = 0;
		for (int x = 0; x < clipCountX; x++)
		{
			if (x != 0 && newLength != 0)
			{
				const int _w = newClipW - newLength - newLength;
				WriteImage(dstX, dstY,						 //x,y
					srcX + newLength,						 //srcX
					y != 0 ? srcY + newLength : srcY,		 //srcY
					_w,		 //width
					y != 0 ? newClipH - newLength  - newLength : newClipH,//height
					x, y									 //blockX,blockY
				);
				dstX += _w;
			}
			else
			{
				const int _w = newClipW - newLength;
				WriteImage(dstX, dstY,						 //x,y
					newLength != 0 ? 0 : srcX,				 //srcX
					y != 0 ? srcY + newLength : srcY,		 //srcY
					_w, newClipH,						 //width,height
					x, y									 //blockX,blockY
				);
				dstX += _w;
			}
			srcX += newClipW;
		}
		dstY += newClipH - newLength - newLength;
		srcY += newClipH;
	}

	canvas->UnlockBits(&lockData);

	bool ret = SaveBitmapToPNG(canvas, path, quality);
	delete canvas;
	return ret;
}

ULONG_PTR Token = 0;

void GdiInit()
{
	if (Token)
		return;
	Gdiplus::GdiplusStartupInput In;
	Gdiplus::GdiplusStartupOutput Out;
	GdiplusStartup(&Token, &In, &Out);
}

void GdiClose()
{
	if (!Token)
		return;
	Gdiplus::GdiplusShutdown(Token);
}

IMAGEVIDEOCLASSEND