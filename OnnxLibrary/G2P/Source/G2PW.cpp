#include "../G2PW.hpp"
#include "OnnxLibrary/Base/Source/OrtDlib.hpp"

_D_Dragonian_Lib_G2P_Header

G2PWModel::G2PWModel(
	const void* Parameter
) : CppPinYin(((const G2PWModelHParams*)Parameter)->Configs), OnnxModelBase(
	*(const OnnxRuntime::OnnxRuntimeEnvironment*)((const G2PWModelHParams*)Parameter)->Enviroment,
	((const G2PWModelHParams*)Parameter)->ModelPath,
	*(const DLogger*)(((const G2PWModelHParams*)Parameter)->Logger))
{
	if (!Parameter)
		_D_Dragonian_Lib_Throw_Exception("Parameter is nullptr");

	FileGuard PolyphonicFile(((const G2PWModelHParams*)Parameter)->PolyphonicPath, L"r");
	char Buffer[128];
	std::vector<std::pair<std::wstring, std::wstring>> Char2Polyphonic;
	while (fgets(Buffer, 128, PolyphonicFile))
	{
		auto Line = UTF8ToWideString(Buffer);
		Line.pop_back();
		const auto Pos = Line.find(L'\t');
		Char2Polyphonic.emplace_back(
			Line.substr(0, Pos),
			Line.substr(Pos + 1)
		);
	}
	if (_UseCharLabels)
		for (const auto& [Char, Polyphonic] : Char2Polyphonic)
		{
			auto Cp = Char + L" " + Polyphonic;
			if (!std::ranges::contains(_MyLabels, Cp))
				_MyLabels.EmplaceBack(std::move(Cp));
		}
	else
		for (const auto& Polyphonic : Char2Polyphonic | std::views::values)
			if (!std::ranges::contains(_MyLabels, Polyphonic))
				_MyLabels.EmplaceBack(Polyphonic);
	std::ranges::sort(_MyLabels);
	if (_UseCharLabels)
		for (const auto& [Char, Polyphonic] : Char2Polyphonic)
		{
			const auto Cp = Char + L" " + Polyphonic;
			const auto Offset = std::ranges::distance(_MyLabels.begin(), std::ranges::find(_MyLabels, Cp));
			if (auto it = _MyPolyphonic2Ids.find(Char); it != _MyPolyphonic2Ids.end())
				it->second.EmplaceBack(Offset);
			else
			{
				_MyPolyphonic2Ids[Char] = { Offset };
				_MyPolyphonicLabels.EmplaceBack(Char);
			}
		}
	else
		for (const auto& [Char, Polyphonic] : Char2Polyphonic)
		{
			const auto Offset = std::ranges::distance(_MyLabels.begin(), std::ranges::find(_MyLabels, Polyphonic));
			if (auto it = _MyPolyphonic2Ids.find(Char); it != _MyPolyphonic2Ids.end())
				it->second.EmplaceBack(Offset);
			else
			{
				_MyPolyphonic2Ids[Char] = { Offset };
				_MyPolyphonicLabels.EmplaceBack(Char);
			}
		}

	std::ranges::sort(_MyPolyphonicLabels);

	for (Int64 i = 0; i < static_cast<Int64>(_MyPolyphonicLabels.Size()); ++i)
		_MyPolyphonicLabels2Ids[_MyPolyphonicLabels[i]] = i;

	FileGuard VocabFile(((const G2PWModelHParams*)Parameter)->VocabPath, L"r");
	for (Int64 i = 0; fgets(Buffer, 128, VocabFile); ++i)
	{
		auto Line = UTF8ToWideString(Buffer);
		Line.pop_back();
		_MyVocab.emplace(std::move(Line), i);
	}
	_MyMaxLength = ((const G2PWModelHParams*)Parameter)->MaxLength;
}

std::pair<Vector<std::wstring>, Vector<Int64>> G2PWModel::Convert(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
) const
{
	return Forward(InputText, LanguageID, UserParameter);
}

std::pair<Vector<std::wstring>, Vector<Int64>> G2PWModel::Forward(
	const std::wstring& InputText,
	const std::string& LanguageID,
	const void* UserParameter
) const
{
	return PinYin(InputText, LanguageID, UserParameter);
}

std::pair<Vector<std::wstring>, std::optional<Vector<Int64>>> G2PWModel::ConvertSegment(
	const std::wstring& Seg,
	const CppPinYinParameters& Parameters
) const
{
	static std::vector ToneDict
	{
		L'a', L'o', L'e', L'u', L'i', L'ü'
	};
	static std::vector ToneDict2
	{
		L'ā', L'ō', L'ē', L'ū', L'ī', L'ǖ',
		L'á', L'ó', L'é', L'ú', L'í', L'ǘ',
		L'ǎ', L'ǒ', L'ě', L'ǔ', L'ǐ', L'ǚ',
		L'à', L'ò', L'è', L'ù', L'ì', L'ǜ'
	};
	auto MPar = Parameters;
	MPar.Heteronym = true;

	Vector<std::wstring> PinYinResult;
	Vector<Int64> QueryIds, ToneResult;

	PinYinResult.Reserve(Seg.size());
	ToneResult.Reserve(Seg.size());
	//Texts.Reserve(Seg.size());
	QueryIds.Reserve(Seg.size());

	ConvertChinese(Tokenize(Seg, Parameters), PinYinResult, ToneResult, Seg, MPar);

	for (size_t i = 0; i < PinYinResult.Size(); ++i)
		if (ToneResult[i] == Parameters.HeteronymTone)
			QueryIds.EmplaceBack(i);
	if (QueryIds.Empty())
		return { std::move(PinYinResult), std::move(ToneResult) };

	Vector<Int64> RawTokenIds;
	RawTokenIds.EmplaceBack(_MyVocab.at(L"[CLS]"));
	for (auto Ch : Seg)
	{
		if (static_cast<Int64>(RawTokenIds.Size()) >= _MyMaxLength - 1)
			break;
		auto It = _MyVocab.find(std::wstring(1, Ch));
		if (It == _MyVocab.end())
			RawTokenIds.EmplaceBack(_MyVocab.at(L"[UNK]"));
		else
			RawTokenIds.EmplaceBack(It->second);
	}
	RawTokenIds.EmplaceBack(_MyVocab.at(L"[SEP]"));

	Vector<Int64> TokenIds;
	Vector<Int64> TokenTypeIds(QueryIds.Size() * RawTokenIds.Size(), 0);
	Vector<Int64> AttentionMasks(QueryIds.Size() * RawTokenIds.Size(), 1);
	Vector<Float32> PhonemeMasks;
	Vector<Int64> CharIds;
	Vector<Int64> PositionIds;
	TokenIds.Reserve(QueryIds.Size() * RawTokenIds.Size());
	PhonemeMasks.Reserve(_MyLabels.Size() * QueryIds.Size());
	CharIds.Reserve(QueryIds.Size());
	PositionIds.Reserve(QueryIds.Size());

	Int64 BatchSize = 0;
	for (auto& QueryId : QueryIds)
	{
		auto Word = std::wstring(1, Seg[QueryId]);
		const auto Iter = _MyPolyphonic2Ids.find(Word);
		if (Iter == _MyPolyphonic2Ids.end())
		{
			auto Py = SearchChar(Word);
			auto [NewPinYin, NewTone] = StyleCast(
				Parameters.Style, Py.substr(0, Py.find(L',')), Parameters.NeutralToneWithFive
			);
			PinYinResult[QueryId] = NewPinYin;
			ToneResult[QueryId] = NewTone;
			QueryId = -1;
			continue;
		}

		Vector PhonemeMask(_MyLabels.Size(), _UseMask ? 0.f : 1.f);
		if (_UseMask)
			for (size_t i = 0; i < _MyLabels.Size(); ++i)
				if (std::ranges::contains(Iter->second, static_cast<Int64>(i)))
					PhonemeMask[i] = 1.0f;
		auto CharId = _MyPolyphonicLabels2Ids.at(Word);
		auto PositionId = QueryId + 1;
		++BatchSize;
		TokenIds.Insert(TokenIds.End(), RawTokenIds.Begin(), RawTokenIds.End());
		PhonemeMasks.Insert(PhonemeMasks.End(), PhonemeMask.Begin(), PhonemeMask.End());
		CharIds.EmplaceBack(CharId);
		PositionIds.EmplaceBack(PositionId);
	}
	if (BatchSize == 0)
		return { std::move(PinYinResult), std::move(ToneResult) };

	Int64 TokenShape[]{ BatchSize, static_cast<Int64>(RawTokenIds.Size()) };
	Int64 PhonemeMaskShape[]{ BatchSize, static_cast<Int64>(_MyLabels.Size()) };
	Int64 CharShape[]{ BatchSize };

	OnnxRuntime::OrtTuple InputTensors;
	InputTensors.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			TokenIds.Data(),
			TokenIds.Size(),
			TokenShape,
			2
		)
	);
	InputTensors.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			TokenTypeIds.Data(),
			TokenIds.Size(),
			TokenShape,
			2
		)
	);
	InputTensors.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			AttentionMasks.Data(),
			TokenIds.Size(),
			TokenShape,
			2
		)
	);
	InputTensors.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			PhonemeMasks.Data(),
			PhonemeMasks.Size(),
			PhonemeMaskShape,
			2
		)
	);
	InputTensors.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			CharIds.Data(),
			CharIds.Size(),
			CharShape,
			1
		)
	);
	InputTensors.emplace_back(
		Ort::Value::CreateTensor(
			*_MyMemoryInfo,
			PositionIds.Data(),
			PositionIds.Size(),
			CharShape,
			1
		)
	);

	OnnxRuntime::OrtTuple OutputTensors;

	_D_Dragonian_Lib_Rethrow_Block(
		OutputTensors = RunModel(
			InputTensors
		);
	);

	Tensor<Float32, 2, Device::CPU> Preds;
	auto OutShape = OutputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
	Dimensions<2> Shape;
	if (OutShape.size() == 2)
		Shape = { OutShape[0], OutShape[1] };
	else
		_D_Dragonian_Lib_Throw_Exception("Output shape not match!");

	_D_Dragonian_Lib_Rethrow_Block(
		Preds = OnnxRuntime::CreateTensorViewFromOrtValue<Float32>(
			std::move(OutputTensors[0]),
			Shape
		);
	);

	const auto Ids = Preds.ArgMax(1).Evaluate();
	auto QueryBegin = QueryIds.Begin();
	for (const auto Id : Ids)
	{
		while (*QueryBegin == -1)
			++QueryBegin;
		auto Bopomofo = _MyLabels[Id];
		auto Tone = Bopomofo.back() - L'0';
		Bopomofo.pop_back();
		auto PinYin = Bopomofo2Pinyin(Bopomofo);
		if (Tone > 0 && Tone < 5)
			for (auto& Ch : PinYin)
				if (std::ranges::contains(ToneDict, Ch))
				{
					Ch = ToneDict2[(Tone - 1) * 6 + std::ranges::distance(ToneDict.begin(), std::ranges::find(ToneDict, Ch))];
					break;
				}
		auto [NewPinYin, NewTone] = StyleCast(
			Parameters.Style, PinYin, Parameters.NeutralToneWithFive
		);
		PinYinResult[*QueryBegin] = NewPinYin;
		ToneResult[*QueryBegin] = NewTone;
		++QueryBegin;
	}
	return { std::move(PinYinResult), std::move(ToneResult) };
}

_D_Dragonian_Lib_G2P_End