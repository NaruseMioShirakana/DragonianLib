/**
 * @file CppPinYin.hpp
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief C++ version of pypinyin, for orginal version, see https://github.com/mozillazg/python-pinyin
 * @changes
 *  > 2025/3/29 NaruseMioShirakana Refactored <
 */

#pragma once
#include "Libraries/G2P/G2PBase.hpp"
#include "Libraries/Dict/Dict.hpp"

_D_Dragonian_Lib_G2P_Header

struct CppPinYinConfigs
{
	const wchar_t* DictPath;
	const wchar_t* PinYinDictPath;
};

struct CppPinYinParameters
{
	enum Type {
		// 普通风格，不带声调。如： 中国 -> ``zhong guo``
		NORMAL = 0,
		// 标准声调风格，拼音声调在韵母第一个字母上（默认风格）。如： 中国 -> ``zhōng guó``
		TONE = 1,
		// 声调风格2，即拼音声调在各个韵母之后，用数字 [1-4] 进行表示。如： 中国 -> ``zho1ng guo2``
		TONE2 = 2,
		// 声调风格3，即拼音声调在各个拼音之后，用数字 [1-4] 进行表示。如： 中国 -> ``zhong1 guo2``
		TONE3 = 3,
	};
	enum Number
	{
		// 输出阿拉伯数字
		DEFAULT = 0,
		// 输出中文数字
		CHINESE = 1,
		// 切分单个数字后转为中文输出
		SPLITCHINESE = 2,
		// 将阿拉伯数字单独切分后输出
		SPLIT = 3,
		// 将阿拉伯数字在小数点处切分后输出
		SPLITDOT = 4,
		// 将阿拉伯数字直接删除
		DEL = 5,
	};
	enum ErrorType {
		// 不处理错误音节，保持原样
		NONE = 0,
		// 替换为 UNK
		UNK = 1,
		// 忽略错误音节，直接删除
		IGNORE = 2,
		// 抛出异常或逐字母切分（中文外语言）
		THROWORSPLIT = 3
	};

	/**
	 * @brief 是否使用5标注轻声
	 */
	bool NeutralToneWithFive = false;

	/**
	 * @brief 是否 将 ü 替换为 v
	 */
	bool ReplaceASV = false;

	/**
	 * @brief 拼音风格
	 */
	Type Style = NORMAL;

	/**
	 * @brief 数字风格
	 */
	Number NumberStyle = DEFAULT;

	/**
	 * @brief 是否启用多音字模式
	 */
	bool Heteronym = false;

	/**
	 * @brief 中文错误处理方式
	 */
	ErrorType ChineseError = UNK;

	/**
	 * @brief 英文处理方式
	 */
	ErrorType English = NONE;

	/**
	 * @brief 符号处理方式
	 */
	ErrorType Symbol = NONE;

	/**
	 * @brief 未知音节处理方式
	 */
	ErrorType Unknown = UNK;

	/**
	 * @brief 未知音节 UNK 的音调
	 */
	Int64 UNKTone = 0;

	/**
	 * @brief 只获取声母或只获取韵母相关拼音风格的返回结果，是否严格遵照《汉语拼音方案》来处理声母和韵母
	 */
	bool Strict = false;

	/**
	 * @brief 分词器最大匹配长度（若为-1则设置为词典中最大长度）
	 */
	Int64 MaximumMatch = -1;
};

/**
 * @class CppPinYin
 * @brief C++ version of pypinyin
 */
class CppPinYin : public G2PBase
{
public:
	struct Segment
	{
		enum Type {
			// 未知
			UNKNOWN = 0,
			// 汉字
			CHINESE = 1,
			// 英文
			ENGLISH = 2,
			// 数字
			NUMBER = 3,
			// 标点符号
			PUNCTUATION = 4,
			// 空格
			SPACE = 5,
			// 换行符
			NEWLINE = 6
		};
		std::wstring Text;
		Type SegType = UNKNOWN;
	};

	/**
	 * @brief 创建一个新的 CppPinYin 对象
	 * @param Parameter 指向 CppPinYinConfigs 的指针
	 */
	CppPinYin(
		const void* Parameter = nullptr
	);
	~CppPinYin() override = default;

	CppPinYin(const CppPinYin&) = default;
	CppPinYin(CppPinYin&&) = default;
	CppPinYin& operator=(const CppPinYin&) = default;
	CppPinYin& operator=(CppPinYin&&) = default;

	std::pair<std::unique_lock<std::mutex>, void*> GetExtraInfo() override;

	/**
	 * @brief 加载用户词典
	 * @param _PhrasesTokens 词典
	 * @param _PinYinTokens 拼音词典
	 */
	void LoadUserDict(
		const std::unordered_map<std::wstring, Vector<Dict::Dict::DictType>>& _PhrasesTokens,
		const std::unordered_map<std::wstring, Dict::IdsDict::DictType>& _PinYinTokens
	);
private:
	Dict::Dict _MyPhrasesDict;
	Dict::IdsDict _MyPinYinDict;

protected:
	void Initialize(const void* Parameter) override;
	void Release() override;
	static void InsertPhonemeAndTone(
		CppPinYinParameters::Type Style,
		const std::wstring& PinYin,
		Vector<std::wstring>& PinYinResult,
		Vector<Int64>& ToneResult,
		bool NeutralToneWithFive
	);

public:
	/**
	 * @brief 转换文本到拼音和音调
	 * @param InputText 输入文本
	 * @param LanguageID 语言ID（该参数无效，置空即可）
	 * @param UserParameter 用户参数（指向 CppPinYinParameters）
	 * @return 转换结果 {拼音, 音调}
	 */
	std::pair<Vector<std::wstring>, Vector<Int64>> Convert(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	) override;

	/**
	 * @brief 转换文本到拼音和音调
	 * @param InputText 输入文本
	 * @param LanguageID 语言ID（该参数无效，置空即可）
	 * @param UserParameter 用户参数（指向 CppPinYinParameters）
	 * @return 拼音结果 {拼音, 音调}
	 */
	std::pair<Vector<std::wstring>, Vector<Int64>> PinYin(
		const std::wstring& InputText,
		const std::string& LanguageID,
		const void* UserParameter = nullptr
	);

	/**
	 * @brief 分词
	 * @param InputText 输入文本
	 * @return 分词结果
	 */
	Vector<Segment> Seg(
		const std::wstring& InputText
	);

	/**
	 * @brief 预分词 (默认为将输入文本按照换行符空格和数字进行分割）
	 * @param InputText 输入文本
	 * @return 分词结果
	 */
	virtual Vector<Segment> PreSeg(
		const std::wstring& InputText
	);

	/**
	 * @brief 分词 (默认为将输入文本按照语言和标点符号进行分割）
	 * @param InputText 输入文本
	 * @return 分词结果
	 */
	virtual Vector<Segment> MidSeg(
		Vector<Segment>&& InputText
	);

	/**
	 * @brief 后分词 (默认不进行任何处理）
	 * @param InputText 输入文本
	 * @return 分词结果
	 */
	virtual Vector<Segment> PostSeg(
		Vector<Segment>&& InputText
	);

	virtual Vector<std::wstring> ConvertSegment(
		const std::wstring& Seg,
		const CppPinYinParameters& Parameters
	);
};


_D_Dragonian_Lib_G2P_End