/**
 * @file AutoGrad.h
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
 * @brief AutoGrad for DragonianLib
 * @changes
 *  > 2025/6/3 NaruseMioShirakana Created <
 */

#pragma once
#include "TensorLib/Include/Base/Value.h"

#define _D_Dragonian_Lib_Auto_Grad(_FN, ...) do \
{ \
	 \
} while(0)

_D_Dragonian_Lib_Space_Begin

namespace AutoGrad
{
	using Value = std::shared_ptr<_D_Dragonian_Lib_Namespace DlibValue>;

	class Function
	{
	public:
		friend class Graph;
		using Node = std::shared_ptr<Function>;

	protected:
		Function() = default;

	public:
		virtual ~Function() = default;
		Function(const Function&) = default;
		Function(Function&&) = default;
		Function& operator=(const Function&) = default;
		Function& operator=(Function&&) = default;

	protected:
		virtual DlibValue* GetGrad() const = 0;

	public:
		virtual void Forward() = 0;
		virtual void Backward() = 0;
		virtual void ZeroGrad() = 0;

		template <typename T>
		decltype(auto) GetGrad() const
		{
			return *static_cast<T*>(GetGrad());
		}
	};

	class Graph final
	{
	public:
		friend Function;
		using Node = std::shared_ptr<Function>;
		Graph() = default;

		
	protected:
		TemplateLibrary::Vector<Node> _MyOps;
		TemplateLibrary::Vector<Value> _MyTensors;
		TemplateLibrary::Vector<UInt64> _MyInputIds;
		TemplateLibrary::Vector<UInt64> _MyOutputIds;

	public:

	};
}

_D_Dragonian_Lib_Space_End