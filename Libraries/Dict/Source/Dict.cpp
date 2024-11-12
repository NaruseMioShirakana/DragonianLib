#include "../Dict.hpp"
#include <regex>
#include "Util/StringPreprocess.h"
#include "MJson/MJson.h"

_D_Dragonian_Lib_Dict_Header

Vector<std::wstring> Tokenizer::SplitWithSymbol(
	const std::wstring& _InputText,
	const std::wstring& _RegularExpression,
	const std::initializer_list<int>& _SubMatch
)
{
	auto Result = DragonianLibSTL::Vector<std::wstring>();
	std::wregex Regex(_RegularExpression);
	std::wsregex_token_iterator Begin(_InputText.begin(), _InputText.end(), Regex, _SubMatch);
	std::wsregex_token_iterator End;

	for (auto It = Begin; It != End; ++It)
		if (It->length())
			Result.EmplaceBack(*It);

	return Result;
}

Vector<std::wstring> Tokenizer::SplitWithSymbol(
	const Vector<std::wstring>& _InputSeq,
	const std::wstring& _RegularExpression,
	const std::initializer_list<int>& _SubMatch
)
{
	auto Result = DragonianLibSTL::Vector<std::wstring>();
	std::wregex Regex(_RegularExpression);
	for (const auto& Seq : _InputSeq)
	{
		std::wsregex_token_iterator Begin(Seq.begin(), Seq.end(), Regex, _SubMatch);
		std::wsregex_token_iterator End;

		for (auto It = Begin; It != End; ++It)
			if (It->length())
				Result.EmplaceBack(*It);
	}
	return Result;
}

Vector<std::wstring_view> Tokenizer::SplitWithSymbolToViews(
	const std::wstring& _InputSeq,
	const std::wstring& _RegularExpression,
	const std::initializer_list<int>& _SubMatch
)
{
	auto Result = DragonianLibSTL::Vector<std::wstring_view>();
	std::wregex Regex(_RegularExpression);
	std::wsregex_token_iterator Begin(_InputSeq.begin(), _InputSeq.end(), Regex, _SubMatch);
	std::wsregex_token_iterator End;

	for (auto It = Begin; It != End; ++It)
		if (It->length())
			Result.EmplaceBack(It->first, It->second);

	return Result;
}

void Tokenizer::Tokenize(
	std::wstring_view _InputText,
	Vector<std::wstring>& _OutputTokens,
	TokenizerMethod _Method,
	size_t _MaximumMatching
) const
{
	const auto SymbolLength = _MySymbol.length();
	const auto SrcLength = _InputText.length();
	if (_Method == Maximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min(_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = MaxMatch; i > 0; --i)
			{
				auto SubSubView = _InputText.substr(0, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() } ;

				if (_MyFix == Prefix && i != CurLength && i > SymbolLength)
				{
					std::wstring PrefixToken = { SubSubView.begin(), SubSubView.end() - (ptrdiff_t)SymbolLength };
					PrefixToken += _MySymbol;
					if (_MyVocab.contains(PrefixToken))
					{
						_OutputTokens.EmplaceBack(std::move(PrefixToken));
						_InputText.remove_prefix(i);
						Found = true;
						break;
					}
				}
				else if (_MyFix == Suffix && SrcLength != CurLength && i > SymbolLength)
				{
					std::wstring SuffixToken = { SubSubView.begin() + (ptrdiff_t)SymbolLength, SubSubView.end() };
					SuffixToken.insert(SuffixToken.begin(), _MySymbol.begin(), _MySymbol.end());
					if (_MyVocab.contains(SuffixToken))
					{
						_OutputTokens.EmplaceBack(std::move(SuffixToken));
						_InputText.remove_prefix(i);
						Found = true;
						break;
					}
				}

				if (_MyVocab.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_prefix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(L"[UNK]");
				_InputText.remove_prefix(1);
			}
		}
	}
	else if(_Method == Minimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min(_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = 1; i <= MaxMatch; ++i)
			{
				auto SubSubView = _InputText.substr(0, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };

				if (_MyFix == Prefix && i != CurLength)
				{
					std::wstring PrefixToken = { SubSubView.begin(), SubSubView.end() };
					PrefixToken += _MySymbol;
					if (_MyVocab.contains(PrefixToken))
					{
						_OutputTokens.EmplaceBack(std::move(PrefixToken));
						_InputText.remove_prefix(i);
						Found = true;
						break;
					}
				}
				else if (_MyFix == Suffix && SrcLength != CurLength)
				{
					std::wstring SuffixToken = { SubSubView.begin(), SubSubView.end() };
					SuffixToken.insert(SuffixToken.begin(), _MySymbol.begin(), _MySymbol.end());
					if (_MyVocab.contains(SuffixToken))
					{
						_OutputTokens.EmplaceBack(std::move(SuffixToken));
						_InputText.remove_prefix(i);
						Found = true;
						break;
					}
				}

				if (_MyVocab.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_prefix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(L"[UNK]");
				_InputText.remove_prefix(1);
			}
		}
	}
	else if (_Method == ReversedMinimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min(_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = 1; i <= MaxMatch; ++i)
			{
				auto SubSubView = _InputText.substr(CurLength - i, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyFix == Prefix && SrcLength != CurLength)
				{
					std::wstring PrefixToken = { SubSubView.begin(), SubSubView.end() - (ptrdiff_t)SymbolLength };
					PrefixToken += _MySymbol;
					if (_MyVocab.contains(PrefixToken))
					{
						_OutputTokens.EmplaceBack(std::move(PrefixToken));
						_InputText.remove_suffix(i);
						Found = true;
						break;
					}
				}
				else if (_MyFix == Suffix && i != CurLength)
				{
					std::wstring SuffixToken = { SubSubView.begin() + (ptrdiff_t)SymbolLength, SubSubView.end() };
					SuffixToken.insert(SuffixToken.begin(), _MySymbol.begin(), _MySymbol.end());
					if (_MyVocab.contains(SuffixToken))
					{
						_OutputTokens.EmplaceBack(std::move(SuffixToken));
						_InputText.remove_suffix(i);
						Found = true;
						break;
					}
				}

				if (_MyVocab.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_suffix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(L"[UNK]");
				_InputText.remove_suffix(1);
			}
		}
	}
	else if (_Method == ReversedMaximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min(_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = MaxMatch; i > 0; --i)
			{
				auto SubSubView = _InputText.substr(CurLength - i, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyFix == Prefix && SrcLength != CurLength)
				{
					std::wstring PrefixToken = { SubSubView.begin(), SubSubView.end() - (ptrdiff_t)SymbolLength };
					PrefixToken += _MySymbol;
					if (_MyVocab.contains(PrefixToken))
					{
						_OutputTokens.EmplaceBack(std::move(PrefixToken));
						_InputText.remove_suffix(i);
						Found = true;
						break;
					}
				}
				else if (_MyFix == Suffix && i != CurLength)
				{
					std::wstring SuffixToken = { SubSubView.begin() + (ptrdiff_t)SymbolLength, SubSubView.end() };
					SuffixToken.insert(SuffixToken.begin(), _MySymbol.begin(), _MySymbol.end());
					if (_MyVocab.contains(SuffixToken))
					{
						_OutputTokens.EmplaceBack(std::move(SuffixToken));
						_InputText.remove_suffix(i);
						Found = true;
						break;
					}
				}

				if (_MyVocab.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_suffix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(L"[UNK]");
				_InputText.remove_suffix(1);
			}
		}
	}
	else
		_D_Dragonian_Lib_Fatal_Error;
}

void Tokenizer::Tokenize(
	const std::wstring& _InputText,
	Vector<std::wstring>& _OutputTokens,
	TokenizerMethod _Method,
	bool _SkipNonLatin,
	size_t _MaximumMatching
) const
{
	if (_InputText.empty())
		return;
	auto SeqVector = SplitWithSymbol(
		_InputText,
		PreDefinedRegex::_Valdef_Regex_All_Symbol_Group,
		{ -1, 0 }
	);
	if(_SkipNonLatin)
		SeqVector = SplitWithSymbol(
			SeqVector,
			PreDefinedRegex::_Valdef_Regex_Chinese_And_Japanese,
			{ -1, 0 }
		);

	for (const auto& Seq : SeqVector)
	{
		std::wstring SubSeq = std::regex_replace(Seq, std::wregex(L"[ ]+"), L" ");
		std::wstring_view SubSeqView = SubSeq;
		Tokenize(SubSeqView, _OutputTokens, _Method, _MaximumMatching);
	}
}

void Tokenizer::Tokenize(
	const Vector<std::wstring>& _InputSeq,
	Vector<std::wstring>& _OutputTokens,
	TokenizerMethod _Method,
	bool _SkipNonLatin,
	size_t _MaximumMatching
) const
{
	if (_InputSeq.Empty())
		return;

	if (_SkipNonLatin)
	{
		auto SeqVector = SplitWithSymbol(
			_InputSeq,
			PreDefinedRegex::_Valdef_Regex_Chinese_And_Japanese,
			{ -1, 0 }
		);
		for (const auto& Seq : SeqVector)
		{
			std::wstring SubSeq = std::regex_replace(Seq, std::wregex(L"[ ]+"), L" ");
			std::wstring_view SubSeqView = SubSeq;
			Tokenize(SubSeqView, _OutputTokens, _Method, _MaximumMatching);
		}
		return;
	}

	for (const auto& Seq : _InputSeq)
	{
		std::wstring SubSeq = std::regex_replace(Seq, std::wregex(L"[ ]+"), L" ");
		std::wstring_view SubSeqView = SubSeq;
		Tokenize(SubSeqView, _OutputTokens, _Method, _MaximumMatching);
	}
}

Vector<Tokenizer::TokenizerType> Tokenizer::operator()(const Vector<std::wstring>& _Tokens) const
{
	auto Result = DragonianLibSTL::Vector<TokenizerType>();

	Result.EmplaceBack(_MyVocab.at(L"[CLS]"));
	for (const auto& Token : _Tokens)
	{
		auto Match = _MyVocab.find(Token);
		if (Match == _MyVocab.end())
			Result.EmplaceBack(_MyVocab.at(L"[UNK]"));
		else
			Result.EmplaceBack(Match->second);
	}
	Result.EmplaceBack(_MyVocab.at(L"[SEP]"));

	return Result;
}

Tokenizer::Tokenizer(const std::wstring& _TokenizerModulePath)
{
	const MJson::MJsonDocument _VocabJson(_TokenizerModulePath.c_str());
	if (!_VocabJson.HasMember("ContinuingSubwordPrefix") ||
		!_VocabJson.HasMember("Type") ||
		!_VocabJson.HasMember("Vocab") ||
		_VocabJson["ContinuingSubwordPrefix"].Empty() ||
		_VocabJson["Type"].Empty() ||
		!_VocabJson["ContinuingSubwordPrefix"].IsString() ||
		!_VocabJson["Type"].IsString())
		_D_Dragonian_Lib_Throw_Exception("Tokenizer Model Prase Error");
		const std::string Type = _VocabJson["Type"].GetString();
	if (Type == "Suffix") _MyFix = Suffix;

	_MySymbol = UTF8ToWideString(_VocabJson["ContinuingSubwordPrefix"].GetString());

	if (_VocabJson["Vocab"].IsArray())
	{
		const auto _VocabArray = _VocabJson["Vocab"].GetArray();
		int64_t Index = 0;
		for (const auto& Object : _VocabArray)
		{
			if (!(Object.IsString() || Object.IsArray()))
			{
				auto Beg = Object.GetMemberArray();
				if (Beg.Empty())
					continue;
				_MyVocab[UTF8ToWideString(Beg.Front().first)] = Beg.Front().second.GetInt64();
			}
			else
				_MyVocab[UTF8ToWideString(Object.IsString() ? Object.GetString() : Object.GetArray()[0].GetString())] = Index++;
		}
	}
	else
	{
		const auto _VocabDict = _VocabJson["Vocab"].GetMemberArray();
		for (const auto& Pair : _VocabDict)
		{
			if (Pair.second.IsInt())
				_MyVocab[UTF8ToWideString(Pair.first)] = TokenizerType(Pair.second.GetInt());
			else if (Pair.second.IsFloat())
				_MyVocab[UTF8ToWideString(Pair.first)] = TokenizerType(Pair.second.GetFloat());
		}
	}
}

Dict::Dict(const std::wstring& _DictModulePath)
{
	if (_DictModulePath.empty())
		return;

	MJson::MJsonDocument PhoneJson(_DictModulePath.c_str());

	for (const auto& itr : PhoneJson.GetMemberArray())
	{
		std::wstring Key = UTF8ToWideString(itr.first);
		const auto Value = itr.second.GetArray();
		_MyDict[Key] = std::vector<std::wstring>();
		for (const auto& it : Value)
			_MyDict[Key].push_back(UTF8ToWideString(it.GetString()));
	}
}

Vector<Dict::DictType> Dict::operator()(const Vector<std::wstring>& _Tokens) const
{
	auto Result = DragonianLibSTL::Vector<DictType>();
	for (const auto& Token : _Tokens)
	{
		auto Match = _MyDict.find(Token);
		if (Match == _MyDict.end())
			Result.EmplaceBack(L"[UNKNOWN]");
		else
			for (const auto& SubToken : Match->second)
				Result.EmplaceBack(SubToken);
	}
	return Result;
}

_D_Dragonian_Lib_Dict_End