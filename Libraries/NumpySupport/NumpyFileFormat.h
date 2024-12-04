#pragma once

#include <string>
#include "Libraries/Base.h"
#include "Libraries/MyTemplateLibrary/Vector.h"

_D_Dragonian_Lib_Space_Begin

namespace NumpyFileFormat
{
	using DragonianLibSTL::Vector;

	struct NumpyHeader
	{
		Byte magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
		Byte majorVersion = 1;
		Byte minorVersion = 0;
		uint16_t headerLength = 118;
	};

	template <typename T>
	std::string GetNumpyTypeString()
	{
		if constexpr (std::is_same_v<T, float>)
			return "f4";
		else if constexpr (std::is_same_v<T, double>)
			return "f8";
		else if constexpr (std::is_same_v<T, int>)
			return "i4";
		else if constexpr (std::is_same_v<T, unsigned int>)
			return "u4";
		else if constexpr (std::is_same_v<T, short>)
			return "i2";
		else if constexpr (std::is_same_v<T, unsigned short>)
			return "u2";
		else if constexpr (std::is_same_v<T, char>)
			return "i1";
		else if constexpr (std::is_same_v<T, unsigned char>)
			return "u1";
		else
			_D_Dragonian_Lib_Throw_Exception("Unsupported type");
	}

	size_t GetNumpyTypeAligsize(const std::string& _Type);

	std::pair<Vector<int64_t>, Vector<Byte>> LoadNumpyFile(const std::wstring& _Path);

	template <typename ValueType, size_t Rank>
	void SaveNumpyFile(const std::wstring& _Path, const IDLArray<int64_t, Rank>& _Shape, const Vector<ValueType>& _Data)
	{
		if (_Shape.Multiply() != _Data.Size())
			_D_Dragonian_Lib_Throw_Exception("Invalid shape");
		FileGuard _MyFile(_Path, L"wb");
		if (!_MyFile.Enabled())
			_D_Dragonian_Lib_Throw_Exception("Failed to open file");
		NumpyHeader Header;
		std::string HeaderStr = "{";
		HeaderStr += "'descr': '<" + GetNumpyTypeString<ValueType>() + "', 'fortran_order': False, 'shape': (";
		for (size_t i = 0; i < _Shape.Size(); ++i)
		{
			if (i != 0)
				HeaderStr += ", ";
			HeaderStr += std::to_string(_Shape[i]);
		}
		HeaderStr += "), }\n";
		Header.headerLength = static_cast<uint16_t>(HeaderStr.size()) + 1;
		if (!_MyFile.Write(&Header, sizeof(NumpyHeader)))
			_D_Dragonian_Lib_Throw_Exception("Failed to write header");
		if (!_MyFile.Write(HeaderStr.c_str(), HeaderStr.size()))
			_D_Dragonian_Lib_Throw_Exception("Failed to write header data");
		if (_MyFile.Write(_Data.Data(), sizeof(ValueType), _Data.Size()) != _Data.Size())
			_D_Dragonian_Lib_Throw_Exception("Failed to write data");
	}

	template <typename ValueType, size_t Rank>
	void SaveNumpyFile(const std::wstring& _Path, const IDLArray<int64_t, Rank>& _Shape, const ValueType* _Buffer, size_t _ElementCount)
	{
		if (_Shape.Multiply() != _ElementCount)
			_D_Dragonian_Lib_Throw_Exception("Invalid shape");
		FileGuard _MyFile(_Path, L"wb");
		if (!_MyFile.Enabled())
			_D_Dragonian_Lib_Throw_Exception("Failed to open file");
		NumpyHeader Header;
		std::string HeaderStr = "{";
		HeaderStr += "'descr': '<" + GetNumpyTypeString<ValueType>() + "', 'fortran_order': False, 'shape': (";
		for (size_t i = 0; i < _Shape.Size(); ++i)
		{
			if (i != 0)
				HeaderStr += ", ";
			HeaderStr += std::to_string(_Shape[i]);
		}
		HeaderStr += "), }\n";
		Header.headerLength = static_cast<uint16_t>(HeaderStr.size());
		if (!_MyFile.Write(&Header, sizeof(NumpyHeader)))
			_D_Dragonian_Lib_Throw_Exception("Failed to write header");
		if (!_MyFile.Write(HeaderStr.c_str(), HeaderStr.size()))
			_D_Dragonian_Lib_Throw_Exception("Failed to write header data");
		if (_MyFile.Write(_Buffer, sizeof(ValueType), _ElementCount) != _ElementCount)
			_D_Dragonian_Lib_Throw_Exception("Failed to write data");
	}

}

_D_Dragonian_Lib_Space_End