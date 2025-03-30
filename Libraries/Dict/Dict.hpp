/**
 * @file Dict.hpp
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
 * @brief Dictionary and tokenizer for neural networks
 * @changes
 *  > 2025/3/21 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Tensor.h"

#define _D_Dragonian_Lib_Dict_Header _D_Dragonian_Lib_Space_Begin namespace Dict {
#define _D_Dragonian_Lib_Dict_End _D_Dragonian_Lib_Space_End }

_D_Dragonian_Lib_Dict_Header

using namespace DragonianLibSTL;

DLogger& GetDefaultLogger() noexcept;

/**
 * @class Tokenizer
 * @brief Tokenizer
 */
class Tokenizer
{
public:
	enum TokenizerMethod { Maximum, ReversedMaximum, Minimum, ReversedMinimum };
	enum TokenizerFix { Prefix, Suffix };
	using TokenizerType = int64_t;

	Tokenizer() = delete;
	~Tokenizer() = default;

	/**
	 * @brief Construct a new Tokenizer object
	 * @param _TokenizerModulePath Path to the tokenizer module, a tokenizer module is a text file which contains the vocabulary (json format) which is key-value pairs of token and token id or vector of token text
	 * @param _BeginText Begin token text
	 * @param _EndText End token text
	 * @param _EOSText End of sentence token text
	 * @param _UNKText Unknown token text
	 */
	Tokenizer(
		const std::wstring& _TokenizerModulePath,
		std::wstring _BeginText = L"[CLS]",
		std::wstring _EndText = L"[SEP]",
		std::wstring _EOSText = L"[EOS]",
		std::wstring _UNKText = L"[UNK]"
	);

	Tokenizer(const Tokenizer&) = default;
	Tokenizer(Tokenizer&&) noexcept = default;
	Tokenizer& operator=(const Tokenizer&) = default;
	Tokenizer& operator=(Tokenizer&&) noexcept = default;

	/**
	 * @brief Tokenize the input text
	 * @param _InputText Text to be tokenized
	 * @param _OutputTokens Result buffer for tokenized tokens
	 * @param _Method Tokenizer method
	 * @param _SkipNonLatin Skip non-latin characters
	 * @param _MaximumMatching Maximum matching length
	 */
	void Tokenize(
		const std::wstring& _InputText,
		Vector<std::wstring>& _OutputTokens,
		TokenizerMethod _Method = Maximum,
		bool _SkipNonLatin = true,
		Int64 _MaximumMatching = 32
	) const;

	/**
	 * @brief Tokenize the input text
	 * @param _InputSeq Text sequence to be tokenized
	 * @param _OutputTokens Result buffer for tokenized tokens
	 * @param _Method Tokenizer method
	 * @param _SkipNonLatin Skip non-latin characters
	 * @param _MaximumMatching Maximum matching length
	 */
	void Tokenize(
		const Vector<std::wstring>& _InputSeq,
		Vector<std::wstring>& _OutputTokens,
		TokenizerMethod _Method = Maximum,
		bool _SkipNonLatin = true,
		Int64 _MaximumMatching = 32
	) const;

	/**
	 * @brief Convert tokens to tensor which includes token ids
	 * @param _Tokens Tokens to be converted, each vector represents a sentence, Shape [BatchSize, TokenCount]
	 * @param _AddBegin Whether to add begin token
	 * @param _AddEnd Whether to add end token
	 * @return Tensor which includes token ids, Shape [BatchSize, TokenCount (+ 1 if _AddBegin) (+ 1 if _AddEnd)], you don't need to call evaluate function
	 */
	Tensor<TokenizerType, 2, Device::CPU> operator()(
		const Vector<Vector<std::wstring>>& _Tokens,
		bool _AddBegin = true,
		bool _AddEnd = true
		) const;

	/**
	 * @brief Get token id of a token
	 * @param _Token Token to be converted
	 * @return Token id
	 */
	TokenizerType GetToken(
		const std::wstring& _Token
	) const;

	/**
	 * @brief Load user vocabulary
	 * @param _Vocab User vocabulary
	 */
	void LoadUserVocab(
		const std::unordered_map<std::wstring, TokenizerType>& _Vocab
	);

private:
	std::unordered_map<std::wstring, TokenizerType> _MyVocab;
	Int64 MaximumLength = 0;
	std::wstring _MySymbol = L"##";
	TokenizerFix _MyFix = Prefix;
	std::wstring _MyBeginText = L"[CLS]";
	std::wstring _MyEndText = L"[SEP]";
	std::wstring _MyEOSText = L"[EOS]";
	std::wstring _MyUNKText = L"[UNK]";

	void Tokenize(
		std::wstring_view _InputText,
		Vector<std::wstring>& _OutputTokens,
		TokenizerMethod _Method = Maximum,
		Int64 _MaximumMatching = 32
	) const;
public:
	/**
	 * @brief Split the input text with regular expression
	 * @param _InputText Text to be split
	 * @param _RegularExpression Regular expression for splitting
	 * @param _SubMatch Submatch index
	 * @return Splitted tokens
	 */
	static Vector<std::wstring> SplitWithSymbol(
		const std::wstring& _InputText,
		const std::wregex& _RegularExpression,
		const std::initializer_list<int>& _SubMatch = { -1 }
	);

	/**
	 * @brief Split the input text with regular expression
	 * @param _InputSeq Text sequence to be split
	 * @param _RegularExpression Regular expression for splitting
	 * @param _SubMatch Submatch index
	 * @return Splitted tokens
	 */
	static Vector<std::wstring> SplitWithSymbol(
		const Vector<std::wstring>& _InputSeq,
		const std::wregex& _RegularExpression,
		const std::initializer_list<int>& _SubMatch = { -1 }
	);

	/**
	 * @brief Split the input text with regular expression
	 * @param _InputSeq Text sequence to be split
	 * @param _RegularExpression Regular expression for splitting
	 * @param _SubMatch Submatch index
	 * @return Splitted tokens
	 */
	static Vector<std::wstring_view> SplitWithSymbolToViews(
		const std::wstring& _InputSeq,
		const std::wregex& _RegularExpression,
		const std::initializer_list<int>& _SubMatch = { -1 }
	);
};

/**
 * @class Dict
 * @brief Dictionary
 */
class Dict
{
public:
	using DictType = std::wstring;
	Dict() = default;
	/**
	 * @brief Construct a new Dict object
	 * @param _DictModulePath Path to the dictionary module, a dictionary module is a text file which contains the dictionary (json format) which is key-value pairs of token and vector of token text
	 */
	Dict(const std::wstring& _DictModulePath);
	~Dict() = default;

	/**
	 * @brief Append tokens to the dictionary
	 * @param _Tokens Tokens to be appended
	 */
	void AppendTokens(
		const std::unordered_map<std::wstring, Vector<DictType>>& _Tokens
	);

	/**
	 * @brief Search tokens in the dictionary and replace them with the corresponding token text
	 * @param _Tokens Tokens to be searched
	 * @return Tokens with replaced token text
	 */
	Vector<DictType> operator()(
		const Vector<std::wstring>& _Tokens
		) const;

	/**
	 * @brief Search token in the dictionary and replace them with the corresponding token text
	 * @param _Token Token to be searched
	 * @param _Result Result buffer for token text
	 * @return Tokens with replaced token text
	 */
	const Vector<DictType>& Search(
		const std::wstring& _Token,
		Vector<DictType>* _Result = nullptr
	) const;

	/**
	 * @brief Tokenize the input text
	 * @param _InputText Text to be tokenized
	 * @param _OutputTokens Result buffer for tokenized tokens
	 * @param _Method Tokenizer method
	 * @param _MaximumMatching Maximum matching length
	 */
	void Tokenize(
		std::wstring_view _InputText,
		Vector<std::wstring>& _OutputTokens,
		Tokenizer::TokenizerMethod _Method = Tokenizer::Maximum,
		Int64 _MaximumMatching = 32,
		const std::optional<std::wstring>& _UNKID = std::nullopt
	) const;
private:
	std::unordered_map<std::wstring, Vector<DictType>> _MyDict;
	Int64 MaximumLength = 0;
	static inline Vector<DictType> _MyUnk{ L"UNK" };

public:
	Dict(const Dict&) = default;
	Dict(Dict&&) noexcept = default;
	Dict& operator=(const Dict&) = default;
	Dict& operator=(Dict&&) noexcept = default;
};

/**
 * @class IdsDict
 * @brief Dictionary for token ids
 */
class IdsDict
{
public:
	using DictType = Int64;
	IdsDict() = default;
	/**
	 * @brief Construct a new Dict object
	 * @param _DictModulePath Path to the dictionary module, a dictionary module is a text file which contains the dictionary (json format) which is key-value pairs of token and vector of token text
	 */
	IdsDict(const std::wstring& _DictModulePath);
	~IdsDict() = default;

	/**
	 * @brief Append tokens to the dictionary
	 * @param _Tokens Tokens to be appended
	 */
	void AppendTokens(
		const std::unordered_map<std::wstring, DictType>& _Tokens
	);

	/**
	 * @brief Search tokens in the dictionary and replace them with the corresponding token ids
	 * @param _Tokens Tokens to be searched
	 * @return Tokens with replaced token ids, -1 means UNK
	 */
	Vector<DictType> operator()(
		const Vector<std::wstring>& _Tokens
		) const;

	/**
	 * @brief Search token id in the dictionary and replace them with the corresponding token text
	 * @param _TokenIds Token ids to be searched
	 * @return Tokens with replaced token text
	 */
	Vector<std::wstring> operator[](
		const Vector<DictType>& _TokenIds
		) const;

	/**
	 * @brief Get token id of a token
	 * @param _Token Token to be searched
	 * @return Token id, -1 means UNK
	 */
	const DictType& operator[](
		const std::wstring& _Token
		) const;

	/**
	 * @brief Get token text of a token id
	 * @param _TokenId Token id to be searched
	 * @return Token text, UNK means unknown
	 */
	const std::wstring& operator[](
		const DictType& _TokenId
		) const;

	/**
	 * @brief Tokenize the input text
	 * @param _InputText Text to be tokenized
	 * @param _OutputTokens Result buffer for tokenized tokens
	 * @param _Method Tokenizer method
	 * @param _MaximumMatching Maximum matching length
	 */
	void Tokenize(
		std::wstring_view _InputText,
		Vector<std::wstring>& _OutputTokens,
		Tokenizer::TokenizerMethod _Method = Tokenizer::Maximum,
		Int64 _MaximumMatching = 32,
		const std::optional<std::wstring>& _UNKID = std::nullopt
	) const;
private:
	std::unordered_map<std::wstring, DictType> _MyDict;
	std::unordered_map<DictType, std::wstring> _MyReverseDict;
	Int64 MaximumLength = 0;
	static inline std::wstring _MyUnk = L"UNK";
	static inline DictType _MyUnkId = -1;

public:
	IdsDict(const IdsDict&) = default;
	IdsDict(IdsDict&&) noexcept = default;
	IdsDict& operator=(const IdsDict&) = default;
	IdsDict& operator=(IdsDict&&) noexcept = default;
};

_D_Dragonian_Lib_Dict_End