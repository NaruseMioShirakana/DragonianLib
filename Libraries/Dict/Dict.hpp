#pragma once
#include "Base.h"
#include "MyTemplateLibrary/Vector.h"
#include <regex>

#define _D_Dragonian_Lib_Dict_Header _D_Dragonian_Lib_Space_Begin namespace Dict {
#define _D_Dragonian_Lib_Dict_End _D_Dragonian_Lib_Space_End }

_D_Dragonian_Lib_Dict_Header

using namespace DragonianLibSTL;

class Tokenizer
{
public:
	enum TokenizerMethod { Maximum, ReversedMaximum, Minimum, ReversedMinimum };
	enum TokenizerFix { Prefix, Suffix };
	using TokenizerType = int64_t;

	Tokenizer() = delete;
	~Tokenizer() = default;
	Tokenizer(const std::wstring& _TokenizerModulePath);

	Tokenizer(const Tokenizer&) = default;
	Tokenizer(Tokenizer&&) noexcept = default;
	Tokenizer& operator=(const Tokenizer&) = default;
	Tokenizer& operator=(Tokenizer&&) noexcept = default;

	void Tokenize(
		const std::wstring& _InputText,
		Vector<std::wstring>& _OutputTokens,
		TokenizerMethod _Method = Maximum,
		bool _SkipNonLatin = true,
		size_t _MaximumMatching = 32
	) const;

	void Tokenize(
		const Vector<std::wstring>& _InputSeq,
		Vector<std::wstring>& _OutputTokens,
		TokenizerMethod _Method = Maximum,
		bool _SkipNonLatin = true,
		size_t _MaximumMatching = 32
	) const;

	Vector<TokenizerType> operator()(const Vector<std::wstring>& _Tokens) const;

private:
	std::unordered_map<std::wstring, TokenizerType> _MyVocab;
	std::wstring _MySymbol = L"##";
	TokenizerFix _MyFix = Prefix;

	void Tokenize(
		std::wstring_view _InputText,
		Vector<std::wstring>& _OutputTokens,
		TokenizerMethod _Method = Maximum,
		size_t _MaximumMatching = 32
	) const;
public:
	static Vector<std::wstring> SplitWithSymbol(
		const std::wstring& _InputText,
		const std::wregex& _RegularExpression,
		const std::initializer_list<int>& _SubMatch = { -1 }
	);
	static Vector<std::wstring> SplitWithSymbol(
		const Vector<std::wstring>& _InputSeq,
		const std::wregex& _RegularExpression,
		const std::initializer_list<int>& _SubMatch = { -1 }
	);
	static Vector<std::wstring_view> SplitWithSymbolToViews(
		const std::wstring& _InputSeq,
		const std::wregex& _RegularExpression,
		const std::initializer_list<int>& _SubMatch = { -1 }
	);
};

class Dict
{
public:
	using DictType = std::wstring;
	Dict() = delete;
	Dict(const std::wstring& _DictModulePath);

	Vector<DictType> operator()(const Vector<std::wstring>& _Tokens) const;
private:
	std::unordered_map<std::wstring, std::vector<DictType>> _MyDict;
};

_D_Dragonian_Lib_Dict_End