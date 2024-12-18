#include "../NumpyFileFormat.h"

#include <numeric>
#include <regex>

_D_Dragonian_Lib_Space_Begin

namespace NumpyFileFormat
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
		_D_Dragonian_Lib_Throw_Exception("Unsupported type");
	}

	std::pair<Vector<int64_t>, Vector<Byte>> LoadNumpyFile(const std::wstring& _Path)
	{
		FileGuard _MyFile(_Path, L"rb");
		if (!_MyFile.Enabled())
			_D_Dragonian_Lib_Throw_Exception("Failed to open file");
		NumpyHeader Header;
		if (!_MyFile.Read(&Header, sizeof(NumpyHeader), sizeof(NumpyHeader)))
			_D_Dragonian_Lib_Throw_Exception("Failed to read header");
		if (memcmp(Header.magic, "\x93NUMPY", 6) != 0)
			_D_Dragonian_Lib_Throw_Exception("Invalid magic number");
		if (Header.majorVersion != 1 || Header.minorVersion != 0)
			_D_Dragonian_Lib_Throw_Exception("Unsupported version");
		if (Header.headerLength < 10)
			_D_Dragonian_Lib_Throw_Exception("Invalid header length");
		if (Header.headerLength == 10)
			return {};

		Vector<char> HeaderData(Header.headerLength);
		if (!_MyFile.Read(HeaderData.Data(), HeaderData.Size(), HeaderData.Size()))
			_D_Dragonian_Lib_Throw_Exception("Failed to read header data");
		auto HeaderStr = std::string(HeaderData.Data(), HeaderData.Size());
		std::smatch Match;
		if (!std::regex_search(HeaderStr, Match, NpHeaderRegex))
			_D_Dragonian_Lib_Throw_Exception("Invalid header format");
		const auto Type = Match[1].str();
		if (Match[2].str() == "True")
			_D_Dragonian_Lib_Throw_Exception("Fortran order is not supported");
		const auto Shape = Match[3].str();
		Vector<int64_t> ShapeVec;
		for (std::sregex_iterator It(Shape.begin(), Shape.end(), NumberRegex), End; It != End; ++It)
			ShapeVec.EmplaceBack(std::stoll(It->str()));

		auto DataSize = GetNumpyTypeAligsize(Type);
		for (int64_t Vec : ShapeVec)
		{
			if (Vec < 0)
				_D_Dragonian_Lib_Throw_Exception("Invalid shape");
			DataSize *= static_cast<size_t>(Vec);
		}

		Vector<Byte> Data(DataSize);
		if (!_MyFile.Read(Data.Data(), Data.Size(), Data.Size()))
			_D_Dragonian_Lib_Throw_Exception("Failed to read data");
		return { std::move(ShapeVec), std::move(Data) };
	}
}

_D_Dragonian_Lib_Space_End