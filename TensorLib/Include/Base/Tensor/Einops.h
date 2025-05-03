/**
 * @file Einops.h
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
 * @brief Einops
 * @changes
 *  > 2025/5/2 NaruseMioShirakana Refactored <
 */

#pragma once

#include "TensorLib/Include/Base/Tensor/Tensor.h"

_D_Dragonian_Lib_Space_Begin

namespace Einops
{
	inline const std::regex _GlobalTokenRegex{
		R"([ \t\n\r]{0,}([0-9a-zA-Z]{1,})[ \t\n\r]{0,})"
	};
	inline const std::regex _GlobalNumberTokenRegex{
		R"([ \t\n\r]{0,}([0-9]{1,})[ \t\n\r]{0,})"
	};

	static inline struct AutoDimTag {} AutoDim;
	static inline struct AutoCatTag {} AutoCat;

	template <size_t _Rank>
	struct Token
	{
		static_assert(_Rank > 0);
		static constexpr size_t _MyRank = _Rank;
		using SingleToken = Token<1>;

		~Token() = default;
		Token(const Token&) = default;
		Token(Token&&) noexcept = default;
		Token& operator=(const Token&) = default;
		Token& operator=(Token&&) noexcept = default;

		std::string_view& Name(size_t _Index)
		{
			return _Names[_Index];
		}
		Rational& Dimension(size_t _Index)
		{
			return _Dims[_Index];
		}

	protected:
		Token() = default;
		TemplateLibrary::Array<std::string_view, _Rank> _Names;
		TemplateLibrary::Array<Rational, _Rank> _Dims;

	public:
		template <size_t _NumRank>
		friend Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, const Token<1>& _Right);
		template <size_t _NumRank>
		friend Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, Int64 _Right);
		template <size_t _NumRank>
		friend Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, const std::string_view& _Token);
		template <size_t _NumRank>
		friend Token<_NumRank> operator/(const Token<_NumRank>& _Left, Int64 _Right);
	};

	template <>
	struct Token<1>
	{
		static constexpr size_t _MyRank = 1;

		Token() = default;
		~Token() = default;
		Token(const Token&) = default;
		Token(Token&&) noexcept = default;
		Token& operator=(const Token&) = default;
		Token& operator=(Token&&) noexcept = default;
		Token(const std::string_view& _Token)
		{
			std::match_results<decltype(_Token.cbegin())> _Results;
			if (std::regex_match(_Token.cbegin(), _Token.cend(), _Results, _GlobalTokenRegex))
			{
				_Names[0] = { _Results[1].first, _Results[1].second };
				if (std::regex_match(_Token.cbegin(), _Token.cend(), _Results, _GlobalNumberTokenRegex))
				{
					_Names[0] = std::string_view();
					_Dims[0] = std::stoll(_Results[1].str(), nullptr, 10);
				}
				else
					_Dims[0] = 0;
			}
			else
				_D_Dragonian_Lib_Throw_Exception("Illegal expressions");
		}
		Token(SizeType _Value)
		{
			_Dims[0] = _Value;
		}

		std::string_view& Name(size_t _Index)
		{
			return _Names[_Index];
		}
		Rational& Dimension(size_t _Index)
		{
			return _Dims[_Index];
		}

	protected:
		TemplateLibrary::Array<std::string_view, 1> _Names;
		TemplateLibrary::Array<Rational, 1> _Dims;

	public:
		template <size_t _NumRank>
		friend Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, const Token<1>& _Right);
		template <size_t _NumRank>
		friend Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, Int64 _Right);
		template <size_t _NumRank>
		friend Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, const std::string_view& _Token);
		template <size_t _NumRank>
		friend Token<_NumRank> operator/(const Token<_NumRank>& _Left, Int64 _Right);
	};

	template <typename>
	struct IsToken
	{
		static constexpr bool Cond = false;
	};
	template <size_t _Rank>
	struct IsToken<Token<_Rank>>
	{
		static constexpr bool Cond = true;
	};
	template <typename>
	struct IsAutoDim
	{
		static constexpr bool Cond = false;
	};
	template <>
	struct IsAutoDim<AutoDimTag>
	{
		static constexpr bool Cond = true;
	};
	template <typename _Type>
	constexpr bool IsAutoDimValue = IsAutoDim<TypeTraits::RemoveARPCVType<_Type>>::Cond;
	template <typename>
	struct IsAutoCat
	{
		static constexpr bool Cond = false;
	};
	template <>
	struct IsAutoCat<AutoCatTag>
	{
		static constexpr bool Cond = true;
	};
	template <typename _Type>
	constexpr bool IsAutoCatValue = IsAutoCat<TypeTraits::RemoveARPCVType<_Type>>::Cond;

	template <typename _Type>
	constexpr bool IsTokenValue = IsToken<TypeTraits::RemoveARPCVType<_Type>>::Cond || IsAutoDimValue<_Type> || IsAutoCatValue<_Type>;
	template <typename _Type>
	constexpr bool IsTokenLikeValue = IsTokenValue<_Type> || TypeTraits::CouldBeConvertedFromValue<Token<1>, _Type>;
	template <typename _TupleType, size_t... _Index>
	constexpr bool TupleIsAllToken(std::index_sequence<_Index...>)
	{
		return BoolConditionAndValue<IsTokenLikeValue<std::tuple_element_t<_Index, _TupleType>>...>;
	}
	template <typename _TupleType>
	constexpr bool TupleIsAllToken()
	{
		if constexpr (std::tuple_size_v<_TupleType> == 0)
			return false;
		return TupleIsAllToken<_TupleType>(std::make_index_sequence<std::tuple_size_v<_TupleType>>{});
	}

	template <size_t _NumRank>
	Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, const Token<1>& _Right)
	{
		Token<_NumRank + 1> _Ret;
		_Ret._Names = TemplateLibrary::ArrayCat(_Left._Names, _Right._Names);
		_Ret._Dims = TemplateLibrary::ArrayCat(_Left._Dims, _Right._Dims);
		return _Ret;
	}
	template <size_t _NumRank>
	Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, Int64 _Right)
	{
		Token<_NumRank + 1> _Ret;
		_Ret._Names = TemplateLibrary::ArrayCat(_Left._Names, TemplateLibrary::Array<std::string_view, 1>());
		_Ret._Dims = TemplateLibrary::ArrayCat(_Left._Dims, TemplateLibrary::Array<Rational, 1>{_Right});
		return _Ret;
	}
	template <size_t _NumRank>
	Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, const std::string_view& _Token)
	{
		return _Left * Token<1>(_Token);
	}
	template <size_t _NumRank>
	Token<_NumRank> operator/(const Token<_NumRank>& _Left, Int64 _Right)
	{
		auto _Ret = _Left;
		_Ret._Dims.Back().Denominator *= _Right;
		return _Ret;
	}

	template <typename _ArgType>
	decltype(auto) MakeRearrangeArg(
		const _ArgType& _Arg
	)
	{
		if constexpr (IsTokenValue<_ArgType>)
			return _Arg;
		else
			return Token<1>(_Arg);
	}

	template <typename _TupleArg, size_t... _Index>
	auto ImplMakeRearrangeArgs(
		const _TupleArg& _RawArgs,
		std::index_sequence<_Index...>
	)
	{
		return std::make_tuple(MakeRearrangeArg(std::get<_Index>(_RawArgs))...);
	}

	template <typename _TupleArg>
	auto MakeRearrangeArgsWithTuple(
		const _TupleArg& _RawArgs
	)
		requires (TupleIsAllToken<_TupleArg>())
	{
		return ImplMakeRearrangeArgs(
			_RawArgs,
			std::make_index_sequence<std::tuple_size_v<_TupleArg>>{}
		);
	}

	template <typename... _Args>
	auto MakeRearrangeArgs(
		_Args&&... _RawArgs
	)
		requires (TupleIsAllToken<std::tuple<_Args...>>())
	{
		return ImplMakeRearrangeArgs(
			std::make_tuple(std::forward<_Args>(_RawArgs)...),
			std::make_index_sequence<sizeof...(_Args)>{}
		);
	}

	template <size_t _Rank>
	bool FixSourceRearrangeArg(
		Token<_Rank>& _Arg,
		SizeType _CurDims,
		const std::unordered_map<std::string_view, Int64>& _DimensionCounts
	)
	{
		if constexpr (_Rank == 1)
		{
			if (_Arg.Dimension(0).Numerator == 0)
				_Arg.Dimension(0).Numerator = _CurDims;
		}
		else
		{
			for (size_t i = 0; i < _Rank; ++i)
			{
				if (_Arg.Dimension(i).Numerator == 0)
					_Arg.Dimension(i).Numerator = _DimensionCounts.at(_Arg.Name(i));
				_Arg.Dimension(i) = _Arg.Dimension(i).Integer();
			}
		}
		return true;
	}

	using RearrangeArg = Token<1>;

	template <typename _MyValueType, size_t _Rank, Device _MyDevice, typename _TupleArg1, typename _TupleArg2>
	auto Rearrange(
		const Tensor<_MyValueType, _Rank, _MyDevice>& _Tensor,
		const _TupleArg1& _RawSourceArgs,
		const _TupleArg2& _RawDestArgs,
		const std::unordered_map<std::string_view, Int64>& _DimensionCounts = {}
	)
		requires (TupleIsAllToken<_TupleArg1>() && TupleIsAllToken<_TupleArg2>() && (std::tuple_size_v<_TupleArg1> <= _Rank) && (TypeTraits::CountTypeValue<AutoDimTag, _TupleArg1> < 2) && (TypeTraits::CountTypeValue<AutoCatTag, _TupleArg1> == 0) && (TypeTraits::CountTypeValue<AutoDimTag, _TupleArg2> + TypeTraits::CountTypeValue<AutoCatTag, _TupleArg2> < 2))
	{
		constexpr auto SourceArgSize = std::tuple_size_v<_TupleArg1>;
		constexpr auto DestArgSize = std::tuple_size_v<_TupleArg1>;
		constexpr auto SourceAutoSize = _Rank - SourceArgSize;
		auto NewSourceArgs = MakeRearrangeArgsWithTuple(_RawSourceArgs);
		auto NewDestArgs = MakeRearrangeArgsWithTuple(_RawDestArgs);
		
		if constexpr (std::is_same_v<std::tuple_element_t<0, _TupleArg1>, AutoDimTag> && 
			!std::is_same_v<std::tuple_element_t<std::tuple_size_v<_TupleArg1> -1, _TupleArg1>, AutoDimTag>)
		{
			auto AutoDimArgs = MakeRearrangeArgsWithTuple(
				TemplateLibrary::MakeTuple<SourceAutoSize + 1>(
					_Tensor.Size()
				)
			);
			auto SourceArgs = std::tuple_cat(
				TemplateLibrary::SubTupleView(AutoDimArgs),
				TemplateLibrary::DropElementView<0>(NewSourceArgs)
			);

			[&] <size_t... _Index>(std::index_sequence<_Index...>) {
				ExpandExpression(
					FixSourceRearrangeArg(
						std::get<_Index>(SourceArgs),
						_Tensor.Size(_Index),
						_DimensionCounts
					)...
				);
			}(std::make_index_sequence<std::tuple_size_v<decltype(SourceArgs)>>{});

			return 2;
		}
		else if constexpr (!std::is_same_v<std::tuple_element_t<0, _TupleArg1>, AutoDimTag> &&
			std::is_same_v<std::tuple_element_t<std::tuple_size_v<_TupleArg1> - 1, _TupleArg1>, AutoDimTag>)
		{
			return 1;
		}
		else if constexpr (!std::is_same_v<std::tuple_element_t<0, _TupleArg1>, AutoDimTag> &&
			!std::is_same_v<std::tuple_element_t<std::tuple_size_v<_TupleArg1> -1, _TupleArg1>, AutoDimTag>)
		{
			return 0;
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Illegal expressions");

	}
}

_D_Dragonian_Lib_Space_End
