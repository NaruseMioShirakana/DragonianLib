#include "../Dict.hpp"
#include "Libraries/Util/StringPreprocess.h"
#include "Libraries/MJson/MJson.h"

_D_Dragonian_Lib_Dict_Header

DLogger& GetDefaultLogger() noexcept
{
	static std::shared_ptr<Logger> DefaultLogger = std::make_shared<Logger>(
		*_D_Dragonian_Lib_Namespace GetDefaultLogger(),
		L"Tokenizer"
	);
	return DefaultLogger;
}

Vector<std::wstring> Tokenizer::SplitWithSymbol(
	const std::wstring& _InputText,
	const std::wregex& _RegularExpression,
	const std::initializer_list<int>& _SubMatch
)
{
	auto Result = DragonianLibSTL::Vector<std::wstring>();
	std::wsregex_token_iterator Begin(_InputText.begin(), _InputText.end(), _RegularExpression, _SubMatch);
	std::wsregex_token_iterator End;

	for (auto It = Begin; It != End; ++It)
		if (It->length())
			Result.EmplaceBack(*It);

	return Result;
}

Vector<std::wstring> Tokenizer::SplitWithSymbol(
	const Vector<std::wstring>& _InputSeq,
	const std::wregex& _RegularExpression,
	const std::initializer_list<int>& _SubMatch
)
{
	auto Result = DragonianLibSTL::Vector<std::wstring>();
	for (const auto& Seq : _InputSeq)
	{
		std::wsregex_token_iterator Begin(Seq.begin(), Seq.end(), _RegularExpression, _SubMatch);
		std::wsregex_token_iterator End;

		for (auto It = Begin; It != End; ++It)
			if (It->length())
				Result.EmplaceBack(*It);
	}
	return Result;
}

Vector<std::wstring_view> Tokenizer::SplitWithSymbolToViews(
	const std::wstring& _InputSeq,
	const std::wregex& _RegularExpression,
	const std::initializer_list<int>& _SubMatch
)
{
	auto Result = DragonianLibSTL::Vector<std::wstring_view>();
	std::wsregex_token_iterator Begin(_InputSeq.begin(), _InputSeq.end(), _RegularExpression, _SubMatch);
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
	Int64 _MaximumMatching
) const
{
	if (_MaximumMatching == -1)
		_MaximumMatching = MaximumLength;
	const auto SymbolLength = _MySymbol.length();
	const auto SrcLength = _InputText.length();
	if (_Method == Maximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
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
				_OutputTokens.EmplaceBack(_MyUNKText);
				_InputText.remove_prefix(1);
			}
		}
	}
	else if(_Method == Minimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
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
				_OutputTokens.EmplaceBack(_MyUNKText);
				_InputText.remove_prefix(1);
			}
		}
	}
	else if (_Method == ReversedMinimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
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
				_OutputTokens.EmplaceBack(_MyUNKText);
				_InputText.remove_suffix(1);
			}
		}
	}
	else if (_Method == ReversedMaximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
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
				_OutputTokens.EmplaceBack(_MyUNKText);
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
	Int64 _MaximumMatching
) const
{
	if (_InputText.empty())
		return;
	auto SeqVector = SplitWithSymbol(
		_InputText,
		PreDefinedRegex::AllSymbolGroupRegex,
		{ -1, 0 }
	);
	if(_SkipNonLatin)
		SeqVector = SplitWithSymbol(
			SeqVector,
			PreDefinedRegex::ChineseAndJapaneseRegex,
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
	Int64 _MaximumMatching
) const
{
	if (_InputSeq.Empty())
		return;

	if (_SkipNonLatin)
	{
		auto SeqVector = SplitWithSymbol(
			_InputSeq,
			PreDefinedRegex::ChineseAndJapaneseRegex,
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

Tensor<Tokenizer::TokenizerType, 2, Device::CPU> Tokenizer::operator()(
	const Vector<Vector<std::wstring>>& _Tokens,
	bool _AddBegin,
	bool _AddEnd
	) const
{
	if (_Tokens.Empty())
		_D_Dragonian_Lib_Throw_Exception("Tokens is empty");

	const auto TokenCount = static_cast<SizeType>(_Tokens.Front().Size()) + (_AddBegin ? 1 : 0) + (_AddEnd ? 1 : 0);
	const auto Shape = Dimensions<2>{ static_cast<SizeType>(_Tokens.Size()), TokenCount };
	

	auto Result = Tensor<Tokenizer::TokenizerType, 2, Device::CPU>::New(Shape);
	auto GDataBuffer = Result.Data();
	TokenizerType UnkId;
	_D_Dragonian_Lib_Rethrow_Block(UnkId = _MyVocab.at(_MyUNKText););

	for (const auto [Index, _MyTokens] : Enumrate(_Tokens))
	{
		Result.AppendTask(
			[_AddBegin, _AddEnd, this, &_MyTokens, GDataBuffer, Index, TokenCount, UnkId]
			{
				auto DataBuffer = GDataBuffer + Index * TokenCount;

				if (_AddBegin)
					_D_Dragonian_Lib_Rethrow_Block(*DataBuffer++ = _MyVocab.at(_MyBeginText););

				for (const auto& Token : _MyTokens)
				{
					auto Match = _MyVocab.find(Token);
					if (Match == _MyVocab.end())
						*DataBuffer++ = UnkId;
					else
						*DataBuffer++ = Match->second;
				}

				if (_AddEnd)
					_D_Dragonian_Lib_Rethrow_Block(*DataBuffer = _MyVocab.at(_MyEndText););
			}
		);
	}

	return std::move(Result.Evaluate());
}

Tokenizer::TokenizerType Tokenizer::GetToken(
	const std::wstring& _Token
) const
{
	auto Match = _MyVocab.find(_Token);
	if (Match == _MyVocab.end())
		return _MyVocab.at(_MyUNKText);
	return Match->second;
}

void Tokenizer::LoadUserVocab(
	const std::unordered_map<std::wstring, TokenizerType>& _Vocab
)
{
	for (const auto& Token : _Vocab)
	{
		if (static_cast<Int64>(Token.first.size()) > MaximumLength)
			MaximumLength = static_cast<Int64>(Token.first.size());
		_MyVocab[Token.first] = Token.second;
	}
}

Tokenizer::Tokenizer(
	const std::wstring& _TokenizerModulePath,
	std::wstring _BeginText,
	std::wstring _EndText,
	std::wstring _EOSText,
	std::wstring _UNKText
): _MyBeginText(std::move(_BeginText)), _MyEndText(std::move(_EndText)), _MyEOSText(std::move(_EOSText)), _MyUNKText(std::move(_UNKText))
{
	const MJson::MJsonDocument _VocabJson(_TokenizerModulePath.c_str());

	if (_VocabJson.HasMember("Type") && _VocabJson["Type"].IsString() && _VocabJson["Type"].GetStringLength())
	{
		const std::string Type = _VocabJson["Type"].GetString();
		if (Type == "Suffix") _MyFix = Suffix;
	}
	if (_VocabJson.HasMember("ContinuingSubwordPrefix") && _VocabJson["ContinuingSubwordPrefix"].IsString() && _VocabJson["ContinuingSubwordPrefix"].GetStringLength())
		_MySymbol = UTF8ToWideString(_VocabJson["ContinuingSubwordPrefix"].GetString());

	if (_VocabJson.HasMember("Vocab"))
	{
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
					auto Key = UTF8ToWideString(Beg.Front().first);
					if (static_cast<Int64>(Key.size()) > MaximumLength)
						MaximumLength = static_cast<Int64>(Key.size());
					_MyVocab[Key] = Beg.Front().second.GetInt64();
				}
				else
				{
					auto Key = UTF8ToWideString(Object.IsString() ? Object.GetString() : Object.GetArray()[0].GetString());
					if (static_cast<Int64>(Key.size()) > MaximumLength)
						MaximumLength = static_cast<Int64>(Key.size());
					_MyVocab[Key] = Index++;
				}
			}
		}
		else
		{
			const auto _VocabDict = _VocabJson["Vocab"].GetMemberArray();
			for (const auto& Pair : _VocabDict)
			{
				if (Pair.second.IsInt64())
				{
					const auto& Key = UTF8ToWideString(Pair.first);
					if (static_cast<Int64>(Key.size()) > MaximumLength)
						MaximumLength = static_cast<Int64>(Key.size());
					_MyVocab[UTF8ToWideString(Pair.first)] = TokenizerType(Pair.second.GetInt64());
				}
			}
		}
	}
	else
	{
		const auto _VocabDict = _VocabJson.GetMemberArray();
		for (const auto& Pair : _VocabDict)
			if (Pair.second.IsInt64())
			{
				const auto& Key = UTF8ToWideString(Pair.first);
				if (static_cast<Int64>(Key.size()) > MaximumLength)
					MaximumLength = static_cast<Int64>(Key.size());
				_MyVocab[Key] = TokenizerType(Pair.second.GetInt64());
			}
	}
}

Dict::Dict(
	const std::wstring& _DictModulePath
)
{
	if (_DictModulePath.empty())
		return;

	MJson::MJsonDocument PhoneJson(_DictModulePath.c_str());

	for (const auto& itr : PhoneJson.GetMemberArray())
	{
		std::wstring Key = UTF8ToWideString(itr.first);
		_MyDict[Key] = Vector<std::wstring>();
		if (static_cast<Int64>(Key.size()) > MaximumLength)
			MaximumLength = static_cast<Int64>(Key.size());
		if (itr.second.IsString())
		{
			_MyDict[Key].EmplaceBack(UTF8ToWideString(itr.second.GetString()));
			continue;
		}
		if (!itr.second.IsArray())
			_D_Dragonian_Lib_Throw_Exception("Dict Module Prase Error");
		const auto Value = itr.second.GetArray();
		
		for (const auto& it : Value)
		{
			std::wstring CValue;
			if (it.IsArray())
				CValue = UTF8ToWideString(it.GetArray()[0].GetString());
			else if (!it.IsString())
				_D_Dragonian_Lib_Throw_Exception("Dict Module Prase Error");
			else
				CValue = UTF8ToWideString(it.GetString());
			_MyDict[Key].EmplaceBack(std::move(CValue));
		}
	}
}

void Dict::AppendTokens(
	const std::unordered_map<std::wstring, Vector<DictType>>& _Tokens
)
{
	for (const auto& Token : _Tokens)
	{
		if (static_cast<Int64>(Token.first.size()) > MaximumLength)
			MaximumLength = static_cast<Int64>(Token.first.size());
		_MyDict[Token.first] = Token.second;
	}
}

Vector<Dict::DictType> Dict::operator()(
	const Vector<std::wstring>& _Tokens
	) const
{
	auto Result = DragonianLibSTL::Vector<DictType>();
	for (const auto& Token : _Tokens)
	{
		auto Match = _MyDict.find(Token);
		if (Match == _MyDict.end())
			Result.EmplaceBack(L"UNK");
		else
			for (const auto& SubToken : Match->second)
				Result.EmplaceBack(SubToken);
	}
	return Result;
}

const Vector<Dict::DictType>& Dict::Search(
	const std::wstring& _Token,
	Vector<DictType>* _Result
) const
{
	auto Match = _MyDict.find(_Token);
	if (Match == _MyDict.end())
	{
		if (_Result)
			_Result->EmplaceBack(L"UNK");
		return _MyUnk;
	}
	if (_Result)
		for (const auto& SubToken : Match->second)
			_Result->EmplaceBack(SubToken);
	return Match->second;
}

void Dict::Tokenize(
	std::wstring_view _InputText,
	Vector<std::wstring>& _OutputTokens,
	Tokenizer::TokenizerMethod _Method,
	Int64 _MaximumMatching,
	const std::optional<std::wstring>& _UNKID
) const
{
	if (_MaximumMatching == -1)
		_MaximumMatching = MaximumLength;

	if (_Method == Tokenizer::Maximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = MaxMatch; i > 0; --i)
			{
				auto SubSubView = _InputText.substr(0, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_prefix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk[0]);
				_InputText.remove_prefix(1);
			}
		}
	}
	else if (_Method == Tokenizer::Minimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = 1; i <= MaxMatch; ++i)
			{
				auto SubSubView = _InputText.substr(0, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_prefix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk[0]);
				_InputText.remove_prefix(1);
			}
		}
	}
	else if (_Method == Tokenizer::ReversedMinimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = 1; i <= MaxMatch; ++i)
			{
				auto SubSubView = _InputText.substr(CurLength - i, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_suffix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk[0]);
				_InputText.remove_suffix(1);
			}
		}
	}
	else if (_Method == Tokenizer::ReversedMaximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = MaxMatch; i > 0; --i)
			{
				auto SubSubView = _InputText.substr(CurLength - i, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_suffix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk[0]);
				_InputText.remove_suffix(1);
			}
		}
	}
	else
		_D_Dragonian_Lib_Fatal_Error;
}

IdsDict::IdsDict(
	const std::wstring& _DictModulePath
)
{
	if (_DictModulePath.empty())
		return;
	MJson::MJsonDocument PhoneJson(_DictModulePath.c_str());
	for (const auto& itr : PhoneJson.GetMemberArray())
	{
		std::wstring Key = UTF8ToWideString(itr.first);
		if (itr.second.IsInt64())
		{
			if (static_cast<Int64>(Key.size()) > MaximumLength)
				MaximumLength = static_cast<Int64>(Key.size());
			_MyDict[Key] = itr.second.GetInt64();
			_MyReverseDict[itr.second.GetInt64()] = Key;
			continue;
		}

		if (!itr.second.IsString() || !itr.second.GetStringLength())
			_D_Dragonian_Lib_Throw_Exception("Value is not a string or empty");
		auto Value = UTF8ToWideString(itr.second.GetString());
		if (std::regex_match(Key, PreDefinedRegex::IntegerRegex))
		{
			if (static_cast<Int64>(Value.size()) > MaximumLength)
				MaximumLength = static_cast<Int64>(Value.size());
			_MyReverseDict[wcstoll(Key.c_str(), nullptr, 10)] = UTF8ToWideString(Value);
			_MyDict[Value] = wcstoll(Key.c_str(), nullptr, 10);
		}
		else if (std::regex_match(Value, PreDefinedRegex::IntegerRegex))
		{
			if (static_cast<Int64>(Key.size()) > MaximumLength)
				MaximumLength = static_cast<Int64>(Key.size());
			_MyDict[Key] = wcstoll(Value.c_str(), nullptr, 10);
			_MyReverseDict[wcstoll(Value.c_str(), nullptr, 10)] = Key;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Dict Module Prase Error");
	}
}

void IdsDict::AppendTokens(
	const std::unordered_map<std::wstring, DictType>& _Tokens
)
{
	for (const auto& Token : _Tokens)
	{
		if (static_cast<Int64>(Token.first.size()) > MaximumLength)
			MaximumLength = static_cast<Int64>(Token.first.size());
		_MyDict[Token.first] = Token.second;
		_MyReverseDict[Token.second] = Token.first;
	}
}

Vector<IdsDict::DictType> IdsDict::operator()(
	const Vector<std::wstring>& _Tokens
	) const
{
	auto Result = DragonianLibSTL::Vector<DictType>();
	for (const auto& Token : _Tokens)
	{
		auto Match = _MyDict.find(Token);
		if (Match == _MyDict.end())
			Result.EmplaceBack(_MyUnkId);
		else
			Result.EmplaceBack(Match->second);
	}
	return Result;
}

Vector<std::wstring> IdsDict::operator[](
	const Vector<DictType>& _TokenIds
	) const
{
	auto Result = DragonianLibSTL::Vector<std::wstring>();
	for (const auto& TokenId : _TokenIds)
	{
		auto Match = _MyReverseDict.find(TokenId);
		if (Match == _MyReverseDict.end())
			Result.EmplaceBack(_MyUnk);
		else
			Result.EmplaceBack(Match->second);
	}
	return Result;
}

const IdsDict::DictType& IdsDict::operator[](
	const std::wstring& _Token
) const
{
	auto Match = _MyDict.find(_Token);
	if (Match == _MyDict.end())
		return _MyUnkId;
	return Match->second;
}

const std::wstring& IdsDict::operator[](
	const DictType& _TokenId
	) const
{
	auto Match = _MyReverseDict.find(_TokenId);
	if (Match == _MyReverseDict.end())
		return _MyUnk;
	return Match->second;
}

void IdsDict::Tokenize(
	std::wstring_view _InputText,
	Vector<std::wstring>& _OutputTokens,
	Tokenizer::TokenizerMethod _Method,
	Int64 _MaximumMatching,
	const std::optional<std::wstring>& _UNKID
) const
{
	if (_MaximumMatching == -1)
		_MaximumMatching = MaximumLength;

	if (_Method == Tokenizer::Maximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = MaxMatch; i > 0; --i)
			{
				auto SubSubView = _InputText.substr(0, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_prefix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk);
				_InputText.remove_prefix(1);
			}
		}
	}
	else if (_Method == Tokenizer::Minimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = 1; i <= MaxMatch; ++i)
			{
				auto SubSubView = _InputText.substr(0, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_prefix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk);
				_InputText.remove_prefix(1);
			}
		}
	}
	else if (_Method == Tokenizer::ReversedMinimum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = 1; i <= MaxMatch; ++i)
			{
				auto SubSubView = _InputText.substr(CurLength - i, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_suffix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk);
				_InputText.remove_suffix(1);
			}
		}
	}
	else if (_Method == Tokenizer::ReversedMaximum)
	{
		while (!_InputText.empty())
		{
			const auto CurLength = _InputText.length();
			const auto MaxMatch = std::min((size_t)_MaximumMatching, CurLength);
			bool Found = false;
			for (size_t i = MaxMatch; i > 0; --i)
			{
				auto SubSubView = _InputText.substr(CurLength - i, i);
				std::wstring TokenStr = { SubSubView.begin(), SubSubView.end() };
				if (_MyDict.contains(TokenStr))
				{
					_OutputTokens.EmplaceBack(std::move(TokenStr));
					_InputText.remove_suffix(i);
					Found = true;
					break;
				}
			}
			if (!Found)
			{
				_OutputTokens.EmplaceBack(_UNKID ? *_UNKID : _MyUnk);
				_InputText.remove_suffix(1);
			}
		}
	}
	else
		_D_Dragonian_Lib_Fatal_Error;
}

_D_Dragonian_Lib_Dict_End