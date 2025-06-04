#pragma once
#include "TensorLib/Include/Base/Tensor/Impl/Tensor.h"
#include "TensorLib/Include/Base/Tensor/Operators/Binary.h"
#include "TensorLib/Include/Base/Tensor/Operators/CPU/Binary.h"
#include "TensorLib/Include/Base/Tensor/Operators/CPU/Comparison.h"

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