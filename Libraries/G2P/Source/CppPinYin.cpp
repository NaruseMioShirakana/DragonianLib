#include "../CppPinYin.hpp"

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
) : _MyPhrasesDict(((const CppPinYinConfigs*)Parameter)->DictPath),
_MyPinYinDict(((const CppPinYinConfigs*)Parameter)->PinYinDictPath)
{

}

void CppPinYin::Initialize(const void* Parameter)
{

}

void CppPinYin::Release()
{

}

void CppPinYin::InsertPhonemeAndTone(
	CppPinYinParameters::Type Style,
	const std::wstring& PinYin,
	Vector<std::wstring>& PinYinResult,
	Vector<Int64>& ToneResult,
	bool NeutralToneWithFive
)
{
	if (Style == CppPinYinParameters::NORMAL)
	{
		for (size_t i = 0; i < 28; ++i)
		{
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack((i % 4) + 1);
				PinYinResult.EmplaceBack(PinYin.substr(0, Idx) + ToneMap2[i] + PinYin.substr(Idx + ToneMap[i].size()));
				return;
			}
		}
		for (size_t i = 0; i < 6; ++i)
		{
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack(ToneMapSP[i].second);
				PinYinResult.EmplaceBack(PinYin.substr(0, Idx) + ToneMapSP2[i] + PinYin.substr(Idx + ToneMapSP[i].first.size()));
				return;
			}
		}
		PinYinResult.EmplaceBack(PinYin);
		ToneResult.EmplaceBack(NeutralToneWithFive ? 5 : 0);
	}
	else if (Style == CppPinYinParameters::TONE)
	{
		PinYinResult.EmplaceBack(PinYin);
		for (size_t i = 0; i < 28; ++i)
		{
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack((i % 4) + 1);
				return;
			}
		}
		for (size_t i = 0; i < 6; ++i)
		{
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack(ToneMapSP[i].second);
				return;
			}
		}
		ToneResult.EmplaceBack(NeutralToneWithFive ? 5 : 0);
	}
	else if (Style == CppPinYinParameters::TONE2)
	{
		for (size_t i = 0; i < 28; ++i)
		{
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack((i % 4) + 1);
				PinYinResult.EmplaceBack(
					PinYin.substr(0, Idx) +
					ToneMap2[i] +
					std::to_wstring((i % 4) + 1) +
					PinYin.substr(Idx + ToneMap[i].size())
				);
				return;
			}
		}
		for (size_t i = 0; i < 6; ++i)
		{
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack(ToneMapSP[i].second);
				PinYinResult.EmplaceBack(
					PinYin.substr(0, Idx) +
					ToneMapSP2[i] +
					std::to_wstring(ToneMapSP[i].second) +
					PinYin.substr(Idx + ToneMapSP[i].first.size())
				);
				return;
			}
		}
		for (size_t i = 0; i < 28; ++i)
		{
			if (auto Idx = PinYin.find(ToneMap2[i]); Idx != std::wstring::npos)
			{
				PinYinResult.EmplaceBack(
					PinYin.substr(0, Idx) +
					ToneMap2[i] +
					std::to_wstring(NeutralToneWithFive ? 5 : 0) +
					PinYin.substr(Idx + ToneMap2[i].size())
				);
				ToneResult.EmplaceBack(NeutralToneWithFive ? 5 : 0);
				return;
			}
		}
		for (size_t i = 0; i < 6; ++i)
		{
			if (auto Idx = PinYin.find(ToneMapSP2[i]); Idx != std::wstring::npos)
			{
				PinYinResult.EmplaceBack(
					PinYin.substr(0, Idx) +
					ToneMapSP2[i] +
					std::to_wstring(NeutralToneWithFive ? 5 : 0) +
					PinYin.substr(Idx + ToneMapSP2[i].size())
				);
				ToneResult.EmplaceBack(NeutralToneWithFive ? 5 : 0);
				return;
			}
		}
		PinYinResult.EmplaceBack(PinYin);
		ToneResult.EmplaceBack(NeutralToneWithFive ? 5 : 0);
	}
	else
	{
		for (size_t i = 0; i < 28; ++i)
		{
			if (auto Idx = PinYin.find(ToneMap[i]); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack((i % 4) + 1);
				PinYinResult.EmplaceBack(
					PinYin.substr(0, Idx) +
					ToneMap2[i] +
					PinYin.substr(Idx + ToneMap[i].size()) +
					std::to_wstring((i % 4) + 1)
				);
				return;
			}
		}
		for (size_t i = 0; i < 6; ++i)
		{
			if (auto Idx = PinYin.find(ToneMapSP[i].first); Idx != std::wstring::npos)
			{
				ToneResult.EmplaceBack(ToneMapSP[i].second);
				PinYinResult.EmplaceBack(
					PinYin.substr(0, Idx) +
					ToneMapSP2[i] +
					PinYin.substr(Idx + ToneMapSP[i].first.size()) +
					std::to_wstring(ToneMapSP[i].second)
				);
				return;
			}
		}
		PinYinResult.EmplaceBack(
			PinYin +
			std::to_wstring(NeutralToneWithFive ? 5 : 0)
		);
		ToneResult.EmplaceBack(NeutralToneWithFive ? 5 : 0);
	}
}

void CppPinYin::LoadUserDict(
	const std::unordered_map<std::wstring, Vector<Dict::Dict::DictType>>& _PhrasesTokens,
	const std::unordered_map<std::wstring, Dict::IdsDict::DictType>& _PinYinTokens
)
{
	_MyPhrasesDict.AppendTokens(_PhrasesTokens);
	_MyPinYinDict.AppendTokens(_PinYinTokens);
}

std::pair<std::unique_lock<std::mutex>, void*> CppPinYin::GetExtraInfo()
{
	_D_Dragonian_Lib_Not_Implemented_Error;
}

std::pair<Vector<std::wstring>, Vector<Int64>> CppPinYin::Convert(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
)
{
	return PinYin(InputText, LanguageID, UserParameter);
}

inline std::wregex HeteronymRe = std::wregex(L"[,]");

std::pair<Vector<std::wstring>, Vector<Int64>> CppPinYin::PinYin(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
)
{
	CppPinYinParameters Parameters;
	if (UserParameter)
		Parameters = *(const CppPinYinParameters*)UserParameter;
	auto SegData = Seg(InputText);
	Vector<std::wstring> PinYinResult;
	Vector<Int64> ToneResult;
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
		}
		if (Seg.SegType == Segment::CHINESE)
		{
			std::wstring_view SegText = Seg.Text;
			Vector<std::wstring> Tokens = ConvertSegment(Seg.Text, Parameters);
			
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
					auto CurToken = _MyPinYinDict[static_cast<Int64>(U16Word2Unicode(Word))];
					if (CurToken != L"UNK")
					{
						std::regex_token_iterator TokenIter(
							CurToken.begin(),
							CurToken.end(),
							HeteronymRe,
							-1
						);
						for (decltype(TokenIter) TokenEnd; TokenIter != TokenEnd; ++TokenIter)
						{
							auto Cur = TokenIter->str();
							if (Cur.empty())
								continue;
							InsertPhonemeAndTone(Parameters.Style, Cur, PinYinResult, ToneResult, Parameters.NeutralToneWithFive);
							if (Parameters.Heteronym)
								break;
						}
					}
					else if (Parameters.ChineseError == CppPinYinParameters::NONE)
					{
						PinYinResult.EmplaceBack(Token);
						ToneResult.EmplaceBack(Parameters.UNKTone);
					}
					else if (Parameters.ChineseError == CppPinYinParameters::UNK)
					{
						PinYinResult.EmplaceBack(L"UNK");
						ToneResult.EmplaceBack(Parameters.UNKTone);
					}
					else if (Parameters.ChineseError == CppPinYinParameters::THROWORSPLIT)
						_D_Dragonian_Lib_Throw_Exception("Token not found");
					SegText.remove_prefix(1);
					continue;
				}
				SegText.remove_prefix(Token.size());
				const auto& Searched = _MyPhrasesDict.Search(Token);
				for (const auto& PinYin : Searched)
					InsertPhonemeAndTone(Parameters.Style, PinYin, PinYinResult, ToneResult, Parameters.NeutralToneWithFive);
			}
		}
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
		}
	}
	return { std::move(PinYinResult), std::move(ToneResult) };
}

Vector<CppPinYin::Segment> CppPinYin::Seg(const std::wstring& InputText)
{
	auto SegData = PreSeg(InputText);
	SegData = MidSeg(std::move(SegData));
	return PostSeg(std::move(SegData));
}

Vector<CppPinYin::Segment> CppPinYin::PreSeg(
	const std::wstring& InputText
)
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
)
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
)
{
	return std::move(InputText);
}

Vector<std::wstring> CppPinYin::ConvertSegment(
	const std::wstring& Seg,
	const CppPinYinParameters& Parameters
)
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

_D_Dragonian_Lib_G2P_End