/**
 * @file Binary.h
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
 * @brief Binary ops for DragonianLib
 * @changes
 *  > 2025/6/3 NaruseMioShirakana Created <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Impl/Tensor.h"
#include "TensorLib/Include/Base/Tensor/Operators/Binary.h"
#include "TensorLib/Include/Base/Tensor/Operators/CPU/Binary.h"
#include "TensorLib/Include/Base/Tensor/Operators/CPU/Assign.h"
#include "TensorLib/Include/Base/Tensor/Operators/CPU/Comparison.h"
#ifdef DRAGONIANLIB_ENABLECUDA
#include "TensorLib/Include/Base/Tensor/Operators/CUDA/Binary.h"
#endif

_D_Dragonian_Lib_Space_Begin

namespace AutoGrad
{
	template <typename _MyValueType, size_t _MyRank, Device _MyDevice>
	class Add : public Function
	{
	public:
		using ValueType = _MyValueType;
		static constexpr auto Rank = _MyRank;
		static constexpr auto Device = _MyDevice;

		Add(
			const Tensor<ValueType, Rank, Device>& lhs,
			const ValueType& rhs,
			const Tensor<ValueType, Rank, Device>& res
		)
		{
			lhs.ThrowOnNotEnabled();
			rhs.ThrowOnNotEnabled();
			res.ThrowOnNotEnabled();
			_lhs = lhs;
			_rhs = rhs;
			_res = res;
		}
		Add(
			const ValueType& lhs,
			const Tensor<ValueType, Rank, Device>& rhs,
			const Tensor<ValueType, Rank, Device>& res
		)
		{
			lhs.ThrowOnNotEnabled();
			rhs.ThrowOnNotEnabled();
			res.ThrowOnNotEnabled();
			_lhs = lhs;
			_rhs = rhs;
			_res = res;
		}
		Add(
			const Tensor<ValueType, Rank, Device>& lhs,
			const Tensor<ValueType, Rank, Device>& rhs,
			const Tensor<ValueType, Rank, Device>& res
		)
		{
			lhs.ThrowOnNotEnabled();
			rhs.ThrowOnNotEnabled();
			res.ThrowOnNotEnabled();
			_lhs = lhs;
			_rhs = rhs;
			_res = res;
		}

	protected:
		void Forward() override
		{
			Operators::OperatorsBase<ValueType, Device>::ImplAddTensor(
				_res.Data(),
				_res.GetDefaultOperatorParameter(),
				_lhs.Data(),
				_lhs.GetDefaultOperatorParameter(),
				_rhs.Data(),
				_rhs.GetDefaultOperatorParameter(),
				!_lhs.IsBroadCasted() && _lhs.IsContiguous() &&
				!_rhs.IsBroadCasted() && _rhs.IsContiguous() &&
				!_res.IsBroadCasted() && _res.IsContiguous()
			);
		}

		void Backward() override
		{
			if (_lhs.RequiresGrad())
			{
				auto& Grad = _lhs.Grad();
				if (Grad.Null()) Grad = _grad.View();
			}
			if (_rhs.RequiresGrad())
			{
				auto& Grad = _rhs.Grad();
				if (Grad.Null()) Grad = _grad.View();
			}
		}

		DlibValue* GetGrad() const override
		{
			return &_grad;
		}

		void ZeroGrad() override
		{
			_res.ThrowOnNotEnabled();
		}

	private:
		Tensor<ValueType, Rank, Device> _lhs, _rhs, _res;
		Tensor<ValueType, Rank, Device> _grad;
	};

}

_D_Dragonian_Lib_Space_End