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
 *  > 2025/5/2 NaruseMioShirakana Created <
 *  > 2025/5/4 NaruseMioShirakana Add Rearrange <
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
		const std::string_view& Name(size_t _Index) const
		{
			return _Names[_Index];
		}
		const Rational& Dimension(size_t _Index) const
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
		const std::string_view& Name(size_t _Index) const
		{
			return _Names[_Index];
		}
		const Rational& Dimension(size_t _Index) const
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
	template <>
	struct IsToken<Token<1>>
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
		_Ret._Names = TemplateLibrary::CombineArray(_Left._Names, _Right._Names);
		_Ret._Dims = TemplateLibrary::CombineArray(_Left._Dims, _Right._Dims);
		return _Ret;
	}
	template <size_t _NumRank>
	Token<_NumRank + 1> operator*(const Token<_NumRank>& _Left, Int64 _Right)
	{
		Token<_NumRank + 1> _Ret;
		_Ret._Names = TemplateLibrary::CombineArray(_Left._Names, TemplateLibrary::Array<std::string_view, 1>());
		_Ret._Dims = TemplateLibrary::CombineArray(_Left._Dims, TemplateLibrary::Array<Rational, 1>{_Right});
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
		std::unordered_map<std::string_view, Int64>& _DimensionCounts
	)
	{
		if constexpr (_Rank == 1)
		{
			if (_Arg.Dimension(0).Numerator == 0)
			{
				_Arg.Dimension(0).Numerator = _CurDims;
				_DimensionCounts[_Arg.Name(0)] = _CurDims;
			}
		}
		else
		{
			size_t EmptyCount = 0;
			for (size_t i = 0; i < _Rank; ++i)
				if (_Arg.Dimension(i).Numerator == 0)
					++EmptyCount;
			for (size_t i = 0; i < _Rank; ++i)
			{
				if (_Arg.Dimension(i).Numerator == 0)
				{
					if (EmptyCount > 1)
						_Arg.Dimension(i).Numerator = _DimensionCounts.at(_Arg.Name(i));
					else
					{
						_Arg.Dimension(i).Numerator = _CurDims;
						_DimensionCounts[_Arg.Name(i)] = _CurDims;
					}
				}
				_Arg.Dimension(i) = _Arg.Dimension(i).Integer();
			}
		}
		return true;
	}

	template <size_t _Rank>
	bool FixDestRearrangeArg(
		Token<_Rank>& _Arg,
		const std::unordered_map<std::string_view, Int64>& _DimensionCounts
	)
	{
		if constexpr (_Rank == 1)
		{
			if (_Arg.Dimension(0).Numerator == 0)
				_Arg.Dimension(0).Numerator = _DimensionCounts.at(_Arg.Name(0));
			_Arg.Dimension(0) = _Arg.Dimension(0).Integer();
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

	template <typename _ArgTy>
	auto MakeResizeArg(
		const _ArgTy& _Arg
	)
	{
		constexpr auto _Size = _ArgTy::_MyRank;
		Dimensions<_Size> _Ret;
		for (size_t i = 0; i< _Size; ++i)
			_Ret[i] = _Arg.Dimension(i).Integer();
		return _Ret;
	}

	template <typename _TupleArg, size_t... _Index>
	auto MakeResizeArgs(
		const _TupleArg& _Args,
		std::index_sequence<_Index...>
	)
	{
		return TemplateLibrary::CombineArray(
			MakeResizeArg(std::get<_Index>(_Args))...
		);
	}

	template <typename _ArgTy>
	auto MakeDestResizeArg(
		const _ArgTy& _Arg
	)
	{
		constexpr auto _Size = _ArgTy::_MyRank;
		SizeType Ret = 1;
		for (size_t i = 0; i < _Size; ++i)
			Ret *= _Arg.Dimension(i).Integer();
		return Ret;
	}

	template <typename _TupleArg, size_t... _Index>
	auto MakeDestResizeArgs(
		const _TupleArg& _Args,
		std::index_sequence<_Index...>
	)
	{
		return TemplateLibrary::Array{
			MakeDestResizeArg(std::get<_Index>(_Args))...
		};
	}

	template <Int64 _TotalSize, typename _ArgTy>
	auto MakePermuteMapArg(
		const _ArgTy& _Arg,
		std::unordered_map<std::string, Int64>& _PermuteMap
	)
	{
		constexpr auto _Size = _ArgTy::_MyRank;
		const auto _CurSize = _TotalSize - _PermuteMap.size() - _Size;
		for (size_t i = 0; i < _Size; ++i)
		{
			std::string _View = { _Arg.Name(i).begin(), _Arg.Name(i).end()};
			if (_View.empty())
				_View = std::to_string(_Arg.Dimension(i).Integer());
			if (auto Iter = _PermuteMap.find(_View); Iter == _PermuteMap.end())
				_PermuteMap.emplace(_View, _CurSize + i);
			else
				_D_Dragonian_Lib_Throw_Exception("Duplicate parameters were found");
		}
		return _Size;
	}

	template <Int64 _TotalSize, typename _TupleArg, size_t... _Index>
	auto MakePermuteMap(
		const _TupleArg& _Args,
		std::unordered_map<std::string, Int64>& _PermuteMap,
		IndexSequence<_Index...>
	)
	{
		ExpandExpression(
			MakePermuteMapArg<_TotalSize>(std::get<_Index>(_Args), _PermuteMap)...
		);
	}

	template <typename _ArrTy, typename _ArgTy>
	auto MakePermuteArg(
		const _ArgTy& _Arg,
		_ArrTy& _PermuteArgs,
		const std::unordered_map<std::string, Int64>& _PermuteMap
	)
	{
		constexpr auto _Size = _ArgTy::_MyRank;
		auto _MyEnd = TemplateLibrary::End(_PermuteArgs) - 1;
		while (*_MyEnd != -1)
			--_MyEnd;
		for (size_t i = 0; i < _Size; ++i)
		{
			std::string _View = { _Arg.Name(i).begin(), _Arg.Name(i).end() };
			if (_View.empty())
				_View = std::to_string(_Arg.Dimension(i).Integer());
			if (auto Iter = _PermuteMap.find(_View); Iter != _PermuteMap.end())
				(_MyEnd - _Size + 1)[i] = Iter->second;
			else
				_D_Dragonian_Lib_Throw_Exception("Missing arg [" + _View + "]");
		}
		return _Size;
	}

	template <typename _TupleArg, typename _ArrTy, size_t... _Index>
	auto MakePermuteArgs(
		const _TupleArg& _Args,
		_ArrTy& _PermuteArgs,
		const std::unordered_map<std::string, Int64>& _PermuteMap,
		IndexSequence<_Index...>
	)
	{
		ExpandExpression(
			MakePermuteArg(std::get<_Index>(_Args), _PermuteArgs, _PermuteMap)...
		);
	}

	template <typename _ArgTy>
	constexpr auto ExtractArgSize() { return TypeTraits::RemoveARPCVType<_ArgTy>::_MyRank; }
	template <typename _TupleArg, size_t... _Index>
	constexpr auto ExtractArgsSize(IndexSequence<_Index...>)
	{
		return TemplateLibrary::ExpandSum(
			ExtractArgSize<std::tuple_element_t<_Index, _TupleArg>>()...
		);
	}

	template <typename _TupleArg, size_t... _Index>
	auto CombineArgs(
		const _TupleArg& _Args,
		std::index_sequence<_Index...>
	)
	{
		return TemplateLibrary::ExpandMul(std::get<_Index>(_Args)...);
	}

	using RearrangeToken = Token<1>;

	template <typename _MyValueType, size_t _Rank, Device _MyDevice, typename _TupleArg1, typename _TupleArg2>
	auto Rearrange(
		const Tensor<_MyValueType, _Rank, _MyDevice>& _Tensor,
		const _TupleArg1& _RawSourceArgs,
		const _TupleArg2& _RawDestArgs,
		const std::unordered_map<std::string_view, Int64>& _RawDimensionCounts = {}
	)
		requires (TupleIsAllToken<_TupleArg1>() && TupleIsAllToken<_TupleArg2>() && (std::tuple_size_v<_TupleArg1> <= _Rank) && (TypeTraits::CountTypeValue<AutoDimTag, _TupleArg1> < 2) && (TypeTraits::CountTypeValue<AutoCatTag, _TupleArg1> == 0) && (TypeTraits::CountTypeValue<AutoDimTag, _TupleArg2> + TypeTraits::CountTypeValue<AutoCatTag, _TupleArg2> < 2))
	{
		constexpr auto RawSourceArgSize = std::tuple_size_v<_TupleArg1>;
		constexpr auto RawDestArgSize = std::tuple_size_v<_TupleArg2>;
		constexpr auto SourceAutoSize = _Rank - RawSourceArgSize;
		auto NewSourceArgs = MakeRearrangeArgsWithTuple(_RawSourceArgs);
		auto NewDestArgs = MakeRearrangeArgsWithTuple(_RawDestArgs);
		auto _DimensionCounts = _RawDimensionCounts;
		
		if constexpr (std::is_same_v<std::tuple_element_t<0, _TupleArg1>, AutoDimTag> && 
			!std::is_same_v<std::tuple_element_t<std::tuple_size_v<_TupleArg1> -1, _TupleArg1>, AutoDimTag>)
		{
			if constexpr (!TypeTraits::IsAnyOfValue<std::tuple_element_t<0, _TupleArg2>, AutoDimTag, AutoCatTag> ||
				TypeTraits::IsAnyOfValue<std::tuple_element_t<std::tuple_size_v<_TupleArg2> -1, _TupleArg2>, AutoDimTag, AutoCatTag>)
				_D_Dragonian_Lib_Throw_Exception("Illegal expressions");

			auto AutoDimArgs = MakeRearrangeArgsWithTuple(
				TemplateLibrary::MakeTuple<SourceAutoSize + 1>(
					_Tensor.Size()
				)
			);
			auto SourceArgs = std::tuple_cat(
				TemplateLibrary::SubTupleView(AutoDimArgs),
				TemplateLibrary::DropElementView<0>(NewSourceArgs)
			);
			auto DestArgs = std::tuple_cat(
				TemplateLibrary::SubTupleView(AutoDimArgs),
				TemplateLibrary::DropElementView<0>(NewDestArgs)
			);
			constexpr auto SourceArgSize = std::tuple_size_v<decltype(SourceArgs)>;
			constexpr auto DestArgSize = std::tuple_size_v<decltype(DestArgs)>;
			constexpr auto AutoDimSize = std::tuple_size_v<decltype(AutoDimArgs)>;

			[&] <size_t... _Index>(std::index_sequence<_Index...>) {
				ExpandExpression(
					FixSourceRearrangeArg(
						std::get<_Index>(SourceArgs),
						_Tensor.Size(_Index),
						_DimensionCounts
					)...
				);
			}(std::make_index_sequence<std::tuple_size_v<decltype(SourceArgs)>>{});

			auto ResizeOptions = MakeResizeArgs(SourceArgs, std::make_index_sequence<std::tuple_size_v<decltype(SourceArgs)>>{});

			auto Tensor1 = _Tensor.ReShape(*reinterpret_cast<Dimensions<decltype(ResizeOptions)::_MyRank>*>(&ResizeOptions));

			std::unordered_map<std::string, Int64> _PermuteMap;
			constexpr auto UserDefSize = ExtractArgsSize<decltype(DestArgs)>(MakeIndexRange<AutoDimSize, DestArgSize>{});
			Dimensions<AutoDimSize + UserDefSize> _PremuteArgs;

			MakePermuteMap<AutoDimSize + UserDefSize>(DestArgs, _PermuteMap, MakeIndexRange<AutoDimSize, DestArgSize>{});

			static_assert(decltype(_PremuteArgs)::Rank() == decltype(Tensor1)::Rank(), "Size of rearrange args mismatch");

			_PremuteArgs.AssignConstant(-1);
			for (size_t i = 0; i < AutoDimSize; ++i)
				_PremuteArgs[i] = i;
			MakePermuteArgs(SourceArgs, _PremuteArgs, _PermuteMap, MakeIndexRange<AutoDimSize, SourceArgSize>{});

			auto Tensor2 = Tensor1.Permute(_PremuteArgs);

			if constexpr (std::is_same_v<std::tuple_element_t<0, _TupleArg2>, AutoDimTag>)
			{
				[&] <size_t... _Index>(std::index_sequence<_Index...>) {
					ExpandExpression(
						FixDestRearrangeArg(
							std::get<_Index>(DestArgs),
							_DimensionCounts
						)...
					);
				}(std::make_index_sequence<std::tuple_size_v<decltype(DestArgs)>>{});

				auto DestResizeOptions = MakeDestResizeArgs(DestArgs, std::make_index_sequence<std::tuple_size_v<decltype(DestArgs)>>{});
				return Tensor2.ReShape(*reinterpret_cast<Dimensions<decltype(DestResizeOptions)::_MyRank>*>(&DestResizeOptions));
			}
			else if constexpr (std::is_same_v<std::tuple_element_t<0, _TupleArg2>, AutoCatTag>)
			{
				auto CatDestArgs = std::tuple_cat(
					std::make_tuple(CombineArgs(AutoDimArgs, std::make_index_sequence<std::tuple_size_v<decltype(AutoDimArgs)>>{})),
					TemplateLibrary::DropElementView<0>(NewDestArgs)
				);

				[&] <size_t... _Index>(std::index_sequence<_Index...>) {
					ExpandExpression(
						FixDestRearrangeArg(
							std::get<_Index>(CatDestArgs),
							_DimensionCounts
						)...
					);
				}(std::make_index_sequence<std::tuple_size_v<decltype(CatDestArgs)>>{});

				auto DestResizeOptions = MakeDestResizeArgs(CatDestArgs, std::make_index_sequence<std::tuple_size_v<decltype(CatDestArgs)>>{});
				return Tensor2.ReShape(*reinterpret_cast<Dimensions<decltype(DestResizeOptions)::_MyRank>*>(&DestResizeOptions));
			}
			else
				_D_Dragonian_Lib_Throw_Exception("Illegal expressions");
		}
		else if constexpr (!std::is_same_v<std::tuple_element_t<0, _TupleArg1>, AutoDimTag> &&
			std::is_same_v<std::tuple_element_t<std::tuple_size_v<_TupleArg1> - 1, _TupleArg1>, AutoDimTag>)
		{
			if constexpr (TypeTraits::IsAnyOfValue<std::tuple_element_t<0, _TupleArg2>, AutoDimTag, AutoCatTag> ||
				!TypeTraits::IsAnyOfValue<std::tuple_element_t<std::tuple_size_v<_TupleArg2> -1, _TupleArg2>, AutoDimTag, AutoCatTag>)
				_D_Dragonian_Lib_Throw_Exception("Illegal expressions");

			auto AutoDimArgs = MakeRearrangeArgsWithTuple(
				TemplateLibrary::MakeTuple<SourceAutoSize + 1>(
					_Tensor.Size().Data() + RawSourceArgSize - 1
				)
			);
			auto SourceArgs = std::tuple_cat(
				TemplateLibrary::DropElementView<RawSourceArgSize - 1>(NewSourceArgs),
				TemplateLibrary::SubTupleView(AutoDimArgs)
			);
			auto DestArgs = std::tuple_cat(
				TemplateLibrary::DropElementView<RawDestArgSize - 1>(NewDestArgs),
				TemplateLibrary::SubTupleView(AutoDimArgs)
			);
			constexpr auto SourceArgSize = std::tuple_size_v<decltype(SourceArgs)>;
			constexpr auto DestArgSize = std::tuple_size_v<decltype(DestArgs)>;
			constexpr auto AutoDimSize = std::tuple_size_v<decltype(AutoDimArgs)>;

			[&] <size_t... _Index>(std::index_sequence<_Index...>) {
				ExpandExpression(
					FixSourceRearrangeArg(
						std::get<_Index>(SourceArgs),
						_Tensor.Size(_Index),
						_DimensionCounts
					)...
				);
			}(std::make_index_sequence<std::tuple_size_v<decltype(SourceArgs)>>{});

			auto ResizeOptions = MakeResizeArgs(SourceArgs, std::make_index_sequence<std::tuple_size_v<decltype(SourceArgs)>>{});

			auto Tensor1 = _Tensor.ReShape(*reinterpret_cast<Dimensions<decltype(ResizeOptions)::_MyRank>*>(&ResizeOptions));

			std::unordered_map<std::string, Int64> _PermuteMap;
			constexpr auto UserDefSize = ExtractArgsSize<decltype(DestArgs)>(MakeIndexRange<0, DestArgSize - AutoDimSize>{});
			Dimensions<AutoDimSize + UserDefSize> _PremuteArgs;

			MakePermuteMap<UserDefSize>(DestArgs, _PermuteMap, MakeIndexRange<0, DestArgSize - AutoDimSize>{});

			static_assert(decltype(_PremuteArgs)::Rank() == decltype(Tensor1)::Rank(), "Size of rearrange args mismatch");

			_PremuteArgs.AssignConstant(-1);
			for (size_t i = UserDefSize; i < AutoDimSize + UserDefSize; ++i)
				_PremuteArgs[i] = i;

			auto _PermuteRanges = TemplateLibrary::Ranges(
				TemplateLibrary::Begin(_PremuteArgs),
				TemplateLibrary::Begin(_PremuteArgs) + UserDefSize
			);
			MakePermuteArgs(
				SourceArgs,
				_PermuteRanges,
				_PermuteMap,
				MakeIndexRange<0, SourceArgSize - AutoDimSize>{}
			);

			auto Tensor2 = Tensor1.Permute(_PremuteArgs);

			if constexpr (std::is_same_v<std::tuple_element_t<RawDestArgSize - 1, _TupleArg2>, AutoDimTag>)
			{
				[&] <size_t... _Index>(std::index_sequence<_Index...>) {
					ExpandExpression(
						FixDestRearrangeArg(
							std::get<_Index>(DestArgs),
							_DimensionCounts
						)...
					);
				}(std::make_index_sequence<std::tuple_size_v<decltype(DestArgs)>>{});

				auto DestResizeOptions = MakeDestResizeArgs(DestArgs, std::make_index_sequence<std::tuple_size_v<decltype(DestArgs)>>{});
				return Tensor2.ReShape(*reinterpret_cast<Dimensions<decltype(DestResizeOptions)::_MyRank>*>(&DestResizeOptions));
			}
			else if constexpr (std::is_same_v<std::tuple_element_t<RawDestArgSize - 1, _TupleArg2>, AutoCatTag>)
			{
				auto CatDestArgs = std::tuple_cat(
					TemplateLibrary::DropElementView<RawDestArgSize - 1>(NewDestArgs),
					std::make_tuple(CombineArgs(AutoDimArgs, std::make_index_sequence<std::tuple_size_v<decltype(AutoDimArgs)>>{}))
				);

				[&] <size_t... _Index>(std::index_sequence<_Index...>) {
					ExpandExpression(
						FixDestRearrangeArg(
							std::get<_Index>(CatDestArgs),
							_DimensionCounts
						)...
					);
				}(std::make_index_sequence<std::tuple_size_v<decltype(CatDestArgs)>>{});

				auto DestResizeOptions = MakeDestResizeArgs(CatDestArgs, std::make_index_sequence<std::tuple_size_v<decltype(CatDestArgs)>>{});
				return Tensor2.ReShape(*reinterpret_cast<Dimensions<decltype(DestResizeOptions)::_MyRank>*>(&DestResizeOptions));
			}
			else
				_D_Dragonian_Lib_Throw_Exception("Illegal expressions");
		}
		else if constexpr (!std::is_same_v<std::tuple_element_t<0, _TupleArg1>, AutoDimTag> &&
			!std::is_same_v<std::tuple_element_t<std::tuple_size_v<_TupleArg1> -1, _TupleArg1>, AutoDimTag>)
		{
			constexpr auto SourceArgSize = std::tuple_size_v<decltype(NewSourceArgs)>;
			constexpr auto DestArgSize = std::tuple_size_v<decltype(NewDestArgs)>;

			[&] <size_t... _Index>(std::index_sequence<_Index...>) {
				ExpandExpression(
					FixSourceRearrangeArg(
						std::get<_Index>(NewSourceArgs),
						_Tensor.Size(_Index),
						_DimensionCounts
					)...
				);
			}(std::make_index_sequence<std::tuple_size_v<decltype(NewSourceArgs)>>{});

			auto ResizeOptions = MakeResizeArgs(NewSourceArgs, std::make_index_sequence<std::tuple_size_v<decltype(NewSourceArgs)>>{});

			auto Tensor1 = _Tensor.ReShape(*reinterpret_cast<Dimensions<decltype(ResizeOptions)::_MyRank>*>(&ResizeOptions));

			std::unordered_map<std::string, Int64> _PermuteMap;
			constexpr auto UserDefSize = ExtractArgsSize<decltype(NewDestArgs)>(MakeIndexRange<0, DestArgSize>{});
			Dimensions<UserDefSize> _PremuteArgs;

			MakePermuteMap<UserDefSize>(NewDestArgs, _PermuteMap, MakeIndexRange<0, DestArgSize>{});

			static_assert(decltype(_PremuteArgs)::Rank() == decltype(Tensor1)::Rank(), "Size of rearrange args mismatch");

			_PremuteArgs.AssignConstant(-1);
			MakePermuteArgs(NewSourceArgs, _PremuteArgs, _PermuteMap, MakeIndexRange<0, SourceArgSize>{});

			auto Tensor2 = Tensor1.Permute(_PremuteArgs);

			[&] <size_t... _Index>(std::index_sequence<_Index...>) {
				ExpandExpression(
					FixDestRearrangeArg(
						std::get<_Index>(NewDestArgs),
						_DimensionCounts
					)...
				);
			}(std::make_index_sequence<std::tuple_size_v<decltype(NewDestArgs)>>{});

			auto DestResizeOptions = MakeDestResizeArgs(NewDestArgs, std::make_index_sequence<std::tuple_size_v<decltype(NewDestArgs)>>{});
			return Tensor2.ReShape(*reinterpret_cast<Dimensions<decltype(DestResizeOptions)::_MyRank>*>(&DestResizeOptions));
		}
		else
			_D_Dragonian_Lib_Throw_Exception("Illegal expressions");

	}
}

_D_Dragonian_Lib_Space_End
