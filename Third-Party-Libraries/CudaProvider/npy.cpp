﻿#include "npy.h"

#include <filesystem>
#include <regex>
#include <stdexcept>

namespace DragonianLib
{
	namespace Util
	{
		const auto NpHeaderRegex = std::regex(R"('descr':[ ]*'<(.*?)',[ ]*'fortran_order':[ ]*(.*?),[ ]*'shape':[ ]*\((.*?)\)\, \})");
		const auto NumberRegex = std::regex(R"(\d+)");

		size_t GetNumpyTypeAligsize(const std::string& _Type)
		{
			if (_Type == "f4" || _Type == "i4" || _Type == "u4")
				return 4;
			if (_Type == "f8" || _Type == "i8" || _Type == "u8")
				return 8;
			if (_Type == "i2" || _Type == "u2")
				return 2;
			if (_Type == "i1" || _Type == "u1")
				return 1;
			throw std::runtime_error("Unsupported numpy type: " + _Type);
		}

		class FileGuard  // NOLINT(cppcoreguidelines-special-member-functions)
		{
		public:
			FileGuard(const std::wstring& _Path, const wchar_t* _Mode)
			{
				FileHandle = _wfopen(_Path.c_str(), _Mode);
			}
			~FileGuard()
			{
				if (FileHandle)
					fclose(FileHandle);
			}

			bool Enabled() const
			{
				return FileHandle != nullptr;
			}

			size_t Read(void* _Buffer, size_t _Size, size_t _Count) const
			{
				if (!FileHandle)
					return 0;
				return fread(_Buffer, _Size, _Count, FileHandle);
			}

		private:
			FILE* FileHandle = nullptr;
		};

		std::pair<std::vector<int64_t>, std::vector<Byte>> LoadNumpyFile(const std::wstring& _Path)
		{
			FileGuard _MyFile(_Path, L"rb");
			if (!_MyFile.Enabled())
				throw std::runtime_error("Failed to open file");
			NumpyHeader Header;
			if (_MyFile.Read(&Header, sizeof(NumpyHeader), 1) != 1)
				throw std::runtime_error("Failed to read header");
			if (memcmp(Header.magic, "\x93NUMPY", 6) != 0)
				throw std::runtime_error("Invalid magic number");
			if (Header.majorVersion != 1 || Header.minorVersion != 0)
				throw std::runtime_error("Unsupported version");
			if (Header.headerLength < 10)
				throw std::runtime_error("Invalid header length");
			if (Header.headerLength == 10)
				return {};

			std::vector<char> HeaderData(Header.headerLength);
			if (_MyFile.Read(HeaderData.data(), HeaderData.size(), 1) != 1)
				throw std::runtime_error("Failed to read header data");
			auto HeaderStr = std::string(HeaderData.data(), HeaderData.size());
			std::smatch Match;
			if (!std::regex_search(HeaderStr, Match, NpHeaderRegex))
				throw std::runtime_error("Invalid header format");
			const auto Type = Match[1].str();
			if (Match[2].str() == "True")
				throw std::runtime_error("Fortran order is not supported");
			const auto Shape = Match[3].str();
			std::vector<int64_t> ShapeVec;
			for (std::sregex_iterator It(Shape.begin(), Shape.end(), NumberRegex), End; It != End; ++It)
				ShapeVec.emplace_back(std::stoll(It->str()));

			auto DataSize = GetNumpyTypeAligsize(Type);
			for (int64_t Vec : ShapeVec)
			{
				if (Vec < 0)
					throw std::runtime_error("Invalid shape");
				DataSize *= static_cast<size_t>(Vec);
			}

			std::vector<Byte> Data(DataSize);
			if (_MyFile.Read(Data.data(), Data.size(), 1) != 1)
				throw std::runtime_error("Failed to read data");
			return { std::move(ShapeVec), std::move(Data) };
		}

		CudaModules::Module::DictType LoadNumpyFileToDict(
			const std::wstring& _Path
		)
		{
			CudaModules::Module::DictType Dict;
			std::filesystem::path Directory(_Path);
			for (auto& Part : std::filesystem::directory_iterator(Directory))
			{
				if (!Part.is_regular_file() || Part.path().extension() != L".npy")
					continue;
				auto FileName = Part.path().filename().replace_extension(L"").string();
				auto [Shape, Data] = LoadNumpyFile(Part.path().wstring());
				CudaModules::Tensor<float> Tensor;
				switch (Shape.size())
				{
				case 4:
					Tensor.Resize((unsigned)Shape[0], (unsigned)Shape[1], (unsigned)Shape[2], (unsigned)Shape[3]);
					break;
				case 3:
					Tensor.Resize((unsigned)Shape[0], (unsigned)Shape[1], (unsigned)Shape[2]);
					break;
				case 2:
					Tensor.Resize((unsigned)Shape[0], (unsigned)Shape[1]);
					break;
				case 1:
					Tensor.Resize((unsigned)Shape[0]);
					break;
				default:
					throw std::runtime_error("Unsupported shape size: " + std::to_string(Shape.size()));
				}
				if (Data.size() != Tensor.BufferSize * sizeof(float))
					throw std::runtime_error("Data size mismatch for " + Part.path().string());
				stream_t Stream = CudaProvider::createCudaStream();
				
				if (auto Ret = CudaProvider::cpy2Device(
					Tensor.Data,
					(const float*)Data.data(),
					Tensor.BufferSize,
					Stream))
				{
					CudaProvider::destoryCudaStream(Stream);
					throw std::runtime_error(std::string("Failed to copy data to device: ") + CudaProvider::getCudaError(Ret));
				}
				CudaProvider::destoryCudaStream(Stream);
				Dict[FileName] = std::move(Tensor);
			}
			return Dict;
		}
	}
}
