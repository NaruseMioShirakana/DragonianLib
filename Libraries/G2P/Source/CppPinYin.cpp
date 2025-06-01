#include "Libraries/G2P/CppPinYin.hpp"

_D_Dragonian_Lib_G2P_Header

//[ā, á, ǎ, à, ō, ó, ǒ, ò, ê, ê̄, ế, ê̌, ề, ē, é, ě, è, ī, í, ǐ, ì, ū, ú, ǔ, ù, ǖ, ǘ, ǚ, ǜ]
const inline std::wstring ToneMap[28]
{
	L"ā", L"á", L"ǎ", L"à",
	L"ō", L"ó", L"ǒ", L"ò",
	L"ê̄", L"ế", L"ê̌", L"ề",
	L"ē", L"é", L"ě", L"è",
	L"ī", L"í", L"ǐ", L"ì",
	L"ū", L"ú", L"ǔ", L"ù",
	L"ǖ", L"ǘ", L"ǚ", L"ǜ"
};
const inline std::wstring ToneMap2[28]
{
	L"a", L"a", L"a", L"a",
	L"o", L"o", L"o", L"o",
	L"e", L"e", L"e", L"e",
	L"e", L"e", L"e", L"e",
	L"i", L"i", L"i", L"i",
	L"u", L"u", L"u", L"u",
	L"ü", L"ü", L"ü", L"ü"
};
//[m̄, ḿ, m̀, ń, ň, ǹ, ẑ, ĉ, ŝ, ŋ]
const inline std::pair<std::wstring, Int64> ToneMapSP[]
{
	{L"m̄", 1}, {L"ḿ", 2}, {L"m̀", 4},
	{L"ń", 2}, {L"ň", 3}, {L"ǹ", 4},
};
const inline std::wstring ToneMapSP2[]
{
	L"m", L"m", L"m",
	L"n", L"n", L"n",
};

CppPinYin::CppPinYin(
	const void* Parameter
)
{
	static auto _StaticLogger = _D_Dragonian_Lib_Namespace Dict::GetDefaultLogger();
	if (!Parameter)
		_D_Dragonian_Lib_Throw_Exception("Parameter is nullptr");
	const auto Param = (const CppPinYinConfigs*)Parameter;
	if (Param->DictPath)
		_MyPhrasesDict = Dict::Dict(Param->DictPath);
	else
		_StaticLogger->LogWarn(L"DictPath is null, use empty dict");
	if (Param->PinYinDictPath)
		_MyPinYinDict = Dict::IdsDict(Param->PinYinDictPath);
	else
		_StaticLogger->LogWarn(L"PinYinDictPath is null, use empty dict");
	if (Param->Bopomofo2PinyinPath)
		_MyBopomofo2PinyinDict = Dict::Dict(Param->Bopomofo2PinyinPath);
	else
		_StaticLogger->LogMessage(L"Bopomofo2PinyinPath is null, use empty dict");
	if (Param->RareDict)
		_MyRareDict = Dict::Dict(Param->RareDict);
	else
		_StaticLogger->LogMessage(L"RareDict is null, use empty dict");
}

void CppPinYin::Initialize(const void* Parameter)
{

}

void CppPinYin::Release()
{

}

void CppPinYin::ConvertChinese(
	const Vector<std::wstring>& Tokens,
	Vector<std::wstring>& PinYinResult,
	Vector<Int64>& ToneResult,
	const std::wstring& Seg,
	CppPinYinParameters Parameters
) const
{
	static std::wregex HeteronymRe = std::wregex(L"[,]");
	std::wstring_view SegText = Seg;
	for (const auto& Token : Tokens)
	{
		if (Token == L"UNK")
		{
			std::match_results<std::wstring_view::const_iterator> Match;
			std::regex_search(
				SegText.begin(),
				SegText.end(),
				Match,
				PreDefinedRegex::ChineseRegex
			);
			auto Word = Match.str();
			const auto WordSize = Word.size();
			auto CurToken = _MyPinYinDict[static_cast<Int64>(U16Word2Unicode(Word))];
			if (CurToken != L"UNK")
			{
				std::regex_token_iterator TokenIter(
					CurToken.begin(),
					CurToken.end(),
					HeteronymRe,
					-1
				);

				Vector<std::wstring> PinYinRes;
				Vector<Int64> ToneRes;
				for (decltype(TokenIter) TokenEnd; TokenIter != TokenEnd; ++TokenIter)
				{
					auto Cur = TokenIter->str();
					if (Cur.empty())
						continue;
					InsertPhonemeAndTone(Parameters.Style, Cur, PinYinRes, ToneRes, Parameters.NeutralToneWithFive);
					if (!Parameters.Heteronym)
						break;
				}
				std::wstring PinYin;
				for (size_t i = 0; i < PinYinRes.Size(); ++i)
				{
					if (i) PinYin += L",";
					PinYin += PinYinRes[i];
				}
				PinYinResult.EmplaceBack(PinYin);
				if (Parameters.Heteronym && ToneRes.Size() > 1)
					ToneResult.EmplaceBack(Parameters.HeteronymTone);
				else
					ToneResult.EmplaceBack(ToneRes[0]);
			}
			else if (Parameters.ChineseError == CppPinYinParameters::NONE)
			{
				PinYinResult.EmplaceBack(std::move(Word));
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.ChineseError == CppPinYinParameters::UNK)
			{
				PinYinResult.EmplaceBack(L"UNK");
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.ChineseError == CppPinYinParameters::THROWORSPLIT)
				_D_Dragonian_Lib_Throw_Exception("Token not found");
			SegText.remove_prefix(WordSize);
			continue;
		}
		SegText.remove_prefix(Token.size());
		const auto& Searched = _MyPhrasesDict.Search(Token);
		for (const auto& PinYin : Searched)
			InsertPhonemeAndTone(Parameters.Style, PinYin, PinYinResult, ToneResult, Parameters.NeutralToneWithFive);
	}
}

void CppPinYin::Chinese2PinYin(
	Vector<std::wstring>& PinYinResult,
	Vector<Int64>& ToneResult,
	const std::wstring& Seg,
	const CppPinYinParameters& Parameters
) const
{
	auto [Tokens, Tones] = ConvertSegment(Seg, Parameters);
	
	if (!Tones)
		ConvertChinese(Tokens, PinYinResult, ToneResult, Seg, Parameters);
	else
	{
		for (size_t i = 0; i < Tokens.Size(); ++i)
		{
			PinYinResult.EmplaceBack(std::move(Tokens[i]));
			ToneResult.EmplaceBack(Tones->operator[](i));
		}
	}
}

void CppPinYin::InsertPhonemeAndTone(
	CppPinYinParameters::Type Style,
	const std::wstring& PinYin,
	Vector<std::wstring>& PinYinResult,
	Vector<Int64>& ToneResult,
	bool NeutralToneWithFive
)
{
	auto [NewPinYin, Tone] = StyleCast(Style, PinYin, NeutralToneWithFive);
	PinYinResult.EmplaceBack(std::move(NewPinYin));
	ToneResult.EmplaceBack(Tone);
}

std::pair<std::wstring, Int64> CppPinYin::StyleCast(
	CppPinYinParameters::Type Style,
	const std::wstring& PinYin,
	bool NeutralToneWithFive
)
{
	if (Style == CppPinYinParameters::NORMAL)
	{
		for (size_t i = 0; i < 28; ++i)
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				return {
					PinYin.substr(0, Idx) +
					ToneMap2[i] +
					PinYin.substr(Idx + ToneMap[i].size()),
					(i % 4) + 1
				};
			}
		for (size_t i = 0; i < 6; ++i)
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				return {
					PinYin.substr(0, Idx) +
					ToneMapSP2[i] +
					PinYin.substr(Idx + ToneMapSP[i].first.size()),
					ToneMapSP[i].second
				};
			}
		return {
			PinYin,
			NeutralToneWithFive ? 5 : 0
		};
	}
	if (Style == CppPinYinParameters::TONE)
	{
		for (size_t i = 0; i < 28; ++i)
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				return {
					PinYin,
					(i % 4) + 1
				};
			}
		for (size_t i = 0; i < 6; ++i)
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				return {
					PinYin,
					ToneMapSP[i].second
				};
			}
		return {
			PinYin,
			NeutralToneWithFive ? 5 : 0
		};
	}
	if (Style == CppPinYinParameters::TONE2)
	{
		for (size_t i = 0; i < 28; ++i)
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				return {
					PinYin.substr(0, Idx) +
					ToneMap2[i] +
					std::to_wstring((i % 4) + 1) +
					PinYin.substr(Idx + ToneMap[i].size()),
					(i % 4) + 1
				};
			}
		for (size_t i = 0; i < 6; ++i)
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				return {
					PinYin.substr(0, Idx) +
					ToneMapSP2[i] +
					std::to_wstring(ToneMapSP[i].second) +
					PinYin.substr(Idx + ToneMapSP[i].first.size()),
					ToneMapSP[i].second
				};
			}
		for (size_t i = 0; i < 28; ++i)
		{
			if (auto Idx = PinYin.find(ToneMap2[i]); Idx != std::wstring::npos)
			{
				return {
					PinYin.substr(0, Idx) +
					ToneMap2[i] +
					std::to_wstring(NeutralToneWithFive ? 5 : 0) +
					PinYin.substr(Idx + ToneMap2[i].size()),
					NeutralToneWithFive ? 5 : 0
				};
			}
		}
		for (size_t i = 0; i < 6; ++i)
		{
			if (auto Idx = PinYin.find(ToneMapSP2[i]); Idx != std::wstring::npos)
			{
				return {
					PinYin.substr(0, Idx) +
					ToneMapSP2[i] +
					std::to_wstring(NeutralToneWithFive ? 5 : 0) +
					PinYin.substr(Idx + ToneMapSP2[i].size()),
					NeutralToneWithFive ? 5 : 0
				};
			}
		}
		return { PinYin, NeutralToneWithFive ? 5 : 0 };
	}

	for (size_t i = 0; i < 28; ++i)
	{
		if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
		{
			return {
				PinYin.substr(0, Idx) +
				ToneMap2[i] +
				PinYin.substr(Idx + ToneMap[i].size()) +
				std::to_wstring((i % 4) + 1),
				(i % 4) + 1
			};
		}
	}
	for (size_t i = 0; i < 6; ++i)
	{
		if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
		{
			return {
				PinYin.substr(0, Idx) +
				ToneMapSP2[i] +
				PinYin.substr(Idx + ToneMapSP[i].first.size()) +
				std::to_wstring(ToneMapSP[i].second),
				ToneMapSP[i].second
			};
		}
	}
	return {
		PinYin + std::to_wstring(NeutralToneWithFive ? 5 : 0),
		NeutralToneWithFive ? 5 : 0
	};
}

void CppPinYin::LoadUserDict(
	const std::unordered_map<std::wstring, Vector<Dict::Dict::DictType>>& _PhrasesTokens,
	const std::unordered_map<std::wstring, Dict::IdsDict::DictType>& _PinYinTokens
)
{
	_MyPhrasesDict.AppendTokens(_PhrasesTokens);
	_MyPinYinDict.AppendTokens(_PinYinTokens);
}

constexpr const wchar_t* ShengmuDict[]{
	L"b", L"p", L"m", L"f",
	L"d", L"t", L"n", L"l",
	L"g", L"k", L"h",
	L"j", L"q", L"x",
	L"zh", L"ch", L"sh", L"r",
	L"z", L"c", L"s",
	L"y", L"w"
};

std::pair<Vector<std::wstring>, Vector<Int64>> CppPinYin::SplitYunmu(
	const Vector<std::wstring>& PinYin
)
{
	Vector<std::wstring> Phonemes;
	Vector<Int64> Phoneme2Word;
	Int64 WordIdx = 0;
	for (const auto& Cur : PinYin)
	{
		bool Found = false;
		for (const auto& Shengmu : ShengmuDict)
		{
			if (Cur.starts_with(Shengmu))
			{
				Phonemes.EmplaceBack(Shengmu);
				Phonemes.EmplaceBack(Cur.substr(wcslen(Shengmu)));
				Phoneme2Word.EmplaceBack(WordIdx);
				Phoneme2Word.EmplaceBack(WordIdx);
				Found = true;
				break;
			}
		}
		if (!Found)
		{
			Phonemes.EmplaceBack(Cur);
			Phoneme2Word.EmplaceBack(WordIdx);
		}
		++WordIdx;
	}
	return { std::move(Phonemes), std::move(Phoneme2Word) };
}

void* CppPinYin::GetExtraInfo() const
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

std::pair<Vector<std::wstring>, Vector<Int64>> CppPinYin::Convert(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
) const
{
	return PinYin(InputText, LanguageID, UserParameter);
}

std::pair<Vector<std::wstring>, Vector<Int64>> CppPinYin::PinYin(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
) const
{
	CppPinYinParameters Parameters;
	if (UserParameter)
		Parameters = *(const CppPinYinParameters*)UserParameter;
	if (Parameters.HeteronymTone == Parameters.UNKTone)
		_D_Dragonian_Lib_Throw_Exception("HeteronymTone cannot be UNKTone");
	auto SegData = Seg(InputText);
	Vector<std::wstring> PinYinResult;
	Vector<Int64> ToneResult;

	auto CallbackFn = [&](const wchar_t* Text, CppPinYinParameters::Language Lang)
	{
		if (Parameters.Callback && Parameters.DeleterCallback)
		{
			wchar_t* Phs = nullptr;
			Int64* Ts = nullptr;
			UniqueScopeExit Guard([=, &Parameters]
				{
					if (Phs)
						Parameters.DeleterCallback(Phs);
					if (Ts)
						Parameters.DeleterCallback(Ts);
				}
			);

			Parameters.Callback(Text, Lang, &Phs, &Ts);

			if (!Phs || !Ts)
				return;

			Int64 Size = 0;
			auto Iter = Phs;
			while (*Iter)
			{
				std::wstring Ph = Iter;
				Iter += Ph.size() + 1;
				PinYinResult.EmplaceBack(std::move(Ph));
				ToneResult.EmplaceBack(Ts[Size++]);
			}
		}
	};

	for (auto& Seg : SegData)
	{
		if (Seg.SegType == Segment::NUMBER)
		{
			if (Parameters.NumberStyle == CppPinYinParameters::DEFAULT)
			{
				PinYinResult.EmplaceBack(Seg.Text);
				ToneResult.EmplaceBack(Parameters.UNKTone);
				continue;
			}
			if (Parameters.NumberStyle == CppPinYinParameters::CHINESE)
			{
				Seg.Text = Number2Chinese(Seg.Text);
				Seg.SegType = Segment::CHINESE;
			}
			else if (Parameters.NumberStyle == CppPinYinParameters::SPLITCHINESE)
			{
				std::wstring NewText;
				for (const auto& Ch : Seg.Text)
				{
					if (Ch != L'.')
						NewText += PreDefinedRegex::ChineseNumber[Ch - L'0'];
					else
						NewText += L"点";
				}
				Seg.Text = NewText;
				Seg.SegType = Segment::CHINESE;
			}
			else if (Parameters.NumberStyle == CppPinYinParameters::SPLIT)
			{
				for (auto& Ch : Seg.Text)
				{
					PinYinResult.EmplaceBack(std::wstring(1, Ch));
					ToneResult.EmplaceBack(Parameters.UNKTone);
				}
				continue;
			}
			else if (Parameters.NumberStyle == CppPinYinParameters::SPLITDOT)
			{
				const auto Idx = Seg.Text.find(L'.');
				PinYinResult.EmplaceBack(Seg.Text.substr(0, Idx));
				ToneResult.EmplaceBack(Parameters.UNKTone);
				PinYinResult.EmplaceBack(L".");
				ToneResult.EmplaceBack(Parameters.UNKTone);
				PinYinResult.EmplaceBack(Seg.Text.substr(Idx + 1));
				ToneResult.EmplaceBack(Parameters.UNKTone);
				continue;
			}
			else if (Parameters.NumberStyle == CppPinYinParameters::DEL)
				continue;
			else if (Parameters.NumberStyle == UInt8(CppPinYinParameters::CALLBACK))
			{
				CallbackFn(Seg.Text.c_str(), CppPinYinParameters::__NUM);
				continue;
			}
		}
		if (Seg.SegType == Segment::CHINESE)
			Chinese2PinYin(PinYinResult, ToneResult, Seg.Text, Parameters);
		else if (Seg.SegType == Segment::ENGLISH)
		{
			if (Parameters.English == CppPinYinParameters::NONE)
			{
				PinYinResult.EmplaceBack(Seg.Text);
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.English == CppPinYinParameters::UNK)
			{
				PinYinResult.EmplaceBack(L"UNK");
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.English == CppPinYinParameters::THROWORSPLIT)
			{
				for (const auto& Ch : Seg.Text)
				{
					PinYinResult.EmplaceBack(std::wstring(1, Ch));
					ToneResult.EmplaceBack(Parameters.UNKTone);
				}
			}
			else if (Parameters.English == CppPinYinParameters::CALLBACK)
				CallbackFn(Seg.Text.c_str(), CppPinYinParameters::__EN);
		}
		else if (Seg.SegType == Segment::PUNCTUATION)
		{
			if (Parameters.Symbol == CppPinYinParameters::NONE)
			{
				PinYinResult.EmplaceBack(Seg.Text);
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.Symbol == CppPinYinParameters::UNK)
			{
				PinYinResult.EmplaceBack(L"UNK");
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.Symbol == CppPinYinParameters::THROWORSPLIT)
			{
				for (const auto& Ch : Seg.Text)
				{
					PinYinResult.EmplaceBack(std::wstring(1, Ch));
					ToneResult.EmplaceBack(Parameters.UNKTone);
				}
			}
			else if (Parameters.Symbol == CppPinYinParameters::CALLBACK)
				CallbackFn(Seg.Text.c_str(), CppPinYinParameters::__SYMB);
		}
		else if (Seg.SegType == Segment::SPACE)
		{
			PinYinResult.EmplaceBack(L" ");
			ToneResult.EmplaceBack(Parameters.UNKTone);
		}
		else if (Seg.SegType == Segment::NEWLINE)
		{
			PinYinResult.EmplaceBack(L"\n");
			ToneResult.EmplaceBack(Parameters.UNKTone);
		}
		else if (Seg.SegType == Segment::UNKNOWN)
		{
			if (Parameters.Unknown == CppPinYinParameters::NONE)
			{
				PinYinResult.EmplaceBack(Seg.Text);
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.Unknown == CppPinYinParameters::UNK)
			{
				PinYinResult.EmplaceBack(L"UNK");
				ToneResult.EmplaceBack(Parameters.UNKTone);
			}
			else if (Parameters.Unknown == CppPinYinParameters::THROWORSPLIT)
			{
				for (const auto& Ch : Seg.Text)
				{
					PinYinResult.EmplaceBack(std::wstring(1, Ch));
					ToneResult.EmplaceBack(Parameters.UNKTone);
				}
			}
			else if (Parameters.Unknown == CppPinYinParameters::CALLBACK)
				CallbackFn(Seg.Text.c_str(), CppPinYinParameters::__UNK);
		}
	}
	if (Parameters.ReplaceASV)
		for (auto& PinYin : PinYinResult)
			for (size_t i = 0; i < PinYin.size(); ++i)
				if (PinYin[i] == L'ü')
					PinYin[i] = L'v';
	return { std::move(PinYinResult), std::move(ToneResult) };
}

Vector<CppPinYin::Segment> CppPinYin::Seg(
	const std::wstring& InputText
) const
{
	auto SegData = PreSeg(InputText);
	SegData = MidSeg(std::move(SegData));
	return PostSeg(std::move(SegData));
}

Vector<std::wstring> CppPinYin::Tokenize(
	const std::wstring& Seg,
	const CppPinYinParameters& Parameters
) const
{
	Vector<std::wstring> Tokens;
	_MyPhrasesDict.Tokenize(
		Seg,
		Tokens,
		Dict::Tokenizer::Maximum,
		Parameters.MaximumMatch
	);
	return Tokens;
}

Vector<CppPinYin::Segment> CppPinYin::PreSeg(
	const std::wstring& InputText
) const
{
	Vector<CppPinYin::Segment> SegData;
	std::regex_token_iterator TokenIter(
		InputText.begin(),
		InputText.end(),
		PreDefinedRegex::NewLineRegex,
		{ -1, 0 }
	);
	for (decltype(TokenIter) TokenEnd; TokenIter != TokenEnd; ++TokenIter)
	{
		auto Token = TokenIter->str();
		if (Token.empty())
			continue;
		if (std::regex_match(Token, PreDefinedRegex::NewLineRegex))
		{
			SegData.EmplaceBack(Token, Segment::NEWLINE);
			continue;
		}
		std::regex_token_iterator TokenIter2(
			Token.begin(),
			Token.end(),
			PreDefinedRegex::SpaceRegex,
			{ -1, 0 }
		);
		for (decltype(TokenIter2) TokenIter3; TokenIter2 != TokenIter3; ++TokenIter2)
		{
			auto Token2 = TokenIter2->str();
			if (Token2.empty())
				continue;
			if (std::regex_match(Token2, PreDefinedRegex::SpaceRegex))
			{
				SegData.EmplaceBack(Token2, Segment::SPACE);
				continue;
			}
			std::regex_token_iterator TokenIter4(
				Token2.begin(),
				Token2.end(),
				PreDefinedRegex::RealRegex,
				{ -1, 0 }
			);
			for (decltype(TokenIter4) TokenIter5; TokenIter4 != TokenIter5; ++TokenIter4)
			{
				auto Token3 = TokenIter4->str();
				if (Token3.empty())
					continue;
				if (std::regex_match(Token3, PreDefinedRegex::RealRegex))
				{
					SegData.EmplaceBack(Token3, Segment::NUMBER);
					continue;
				}
				SegData.EmplaceBack(Token3, Segment::UNKNOWN);
			}
		}
	}
	return SegData;
}

Vector<CppPinYin::Segment> CppPinYin::MidSeg(
	Vector<CppPinYin::Segment>&& InputText
) const
{
	auto SegData = std::move(InputText);
	Vector<CppPinYin::Segment> NewSegData;
	for (auto& Seg : SegData)
	{
		if (Seg.SegType == Segment::UNKNOWN)
		{
			std::regex_token_iterator TokenIter(
				Seg.Text.begin(),
				Seg.Text.end(),
				PreDefinedRegex::PunctuationGroupRegex,
				{ -1, 0 }
			);
			for (decltype(TokenIter) TokenEnd; TokenIter != TokenEnd; ++TokenIter)
			{
				auto Token = TokenIter->str();
				if (Token.empty())
					continue;
				if (std::regex_match(Token, PreDefinedRegex::PunctuationGroupRegex))
				{
					NewSegData.EmplaceBack(Token, Segment::PUNCTUATION);
					continue;
				}
				std::regex_token_iterator TokenIter2(
					Token.begin(),
					Token.end(),
					PreDefinedRegex::ChineseStringRegex,
					{ -1, 0 }
				);
				for (decltype(TokenIter2) TokenEnd2; TokenIter2 != TokenEnd2; ++TokenIter2)
				{
					auto Token2 = TokenIter2->str();
					if (Token2.empty())
						continue;
					if (std::regex_match(Token2, PreDefinedRegex::ChineseStringRegex))
					{
						NewSegData.EmplaceBack(Token2, Segment::CHINESE);
						continue;
					}
					std::regex_token_iterator TokenIter3(
						Token2.begin(),
						Token2.end(),
						PreDefinedRegex::AlphabetStringRegex,
						{ -1, 0 }
					);
					for (decltype(TokenIter3) TokenEnd3; TokenIter3 != TokenEnd3; ++TokenIter3)
					{
						auto Token3 = TokenIter3->str();
						if (Token3.empty())
							continue;
						if (std::regex_match(Token3, PreDefinedRegex::AlphabetStringRegex))
						{
							NewSegData.EmplaceBack(Token3, Segment::ENGLISH);
							continue;
						}
						NewSegData.EmplaceBack(Token3, Segment::UNKNOWN);
					}
				}
			}
		}
		else
			NewSegData.EmplaceBack(std::move(Seg));
	}
	return NewSegData;
}

Vector<CppPinYin::Segment> CppPinYin::PostSeg(
	Vector<CppPinYin::Segment>&& InputText
) const
{
	return std::move(InputText);
}

std::pair<Vector<std::wstring>, std::optional<Vector<Int64>>> CppPinYin::ConvertSegment(
	const std::wstring& Seg,
	const CppPinYinParameters& Parameters
) const
{
	return { Tokenize(Seg, Parameters), std::nullopt };
}

std::wstring CppPinYin::SearchRare(
	const std::wstring& Word
) const
{
	return Bopomofo2Pinyin(_MyRareDict.Search(Word));
}

std::wstring CppPinYin::Bopomofo2Pinyin(
	const Vector<std::wstring>& Bopomofo
) const
{
	std::wstring Result;
	for (size_t i = 0; i < Bopomofo.Size(); ++i)
	{
		if (i) Result += L",";
		Result += Bopomofo2Pinyin(Bopomofo[i]);
	}
	return Result;
}

std::wstring CppPinYin::Bopomofo2Pinyin(
	const std::wstring& Bopomofo
) const
{
	auto Ch = Bopomofo.back();
	if (Ch >= L'0' && Ch <= L'9')
		return _MyBopomofo2PinyinDict.Search(Bopomofo.substr(0, Bopomofo.size() - 1))[0] + Ch;
	return _MyBopomofo2PinyinDict.Search(Bopomofo)[0];
}

const Vector<std::wstring>& CppPinYin::SearchCommon(
	const std::wstring& Word
) const
{
	return _MyPhrasesDict.Search(Word);
}

std::wstring CppPinYin::SearchChar(
	const std::wstring& Char
) const
{
	return _MyPinYinDict[static_cast<Int64>(U16Word2Unicode(Char))];
}

_D_Dragonian_Lib_G2P_End