/**
 * FileName: MoeSuperResolution.hpp
 *
 * Copyright (C) 2022-2024 NaruseMioShirakana (shirakanamio@foxmail.com)
 *
 * This file is part of DragonianLib.
 * DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 * GNU Affero General Public License as published by the Free Software Foundation, either version 3
 * of the License, or any later version.
 *
 * DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License along with Foobar.
 * If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 *
*/

#pragma once
#include "OnnxLibrary/SuperResolution/SuperResolution.hpp"

_D_Dragonian_Lib_Lib_Super_Resolution_Header

class SuperResolutionBase : public SuperResolution , public OnnxModelBase<SuperResolutionBase>
{
public:
    SuperResolutionBase(
        const OnnxRuntimeEnvironment& _Environment,
        const HyperParameters& _Parameters,
        const DLogger& _Logger = _D_Dragonian_Lib_Onnx_Super_Resolution_Space GetDefaultLogger()
    );
    ~SuperResolutionBase() override = default;

    std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
        int64_t _BatchSize = 1
    ) const override = 0;

    SuperResolutionBase(const SuperResolutionBase&) = default;
    SuperResolutionBase(SuperResolutionBase&&) noexcept = default;
    SuperResolutionBase& operator=(const SuperResolutionBase&) = default;
    SuperResolutionBase& operator=(SuperResolutionBase&&) noexcept = default;
};

class SuperResolutionBCRGBHW : public SuperResolutionBase
{
public:
    SuperResolutionBCRGBHW(
        const OnnxRuntimeEnvironment& _Environment,
        const HyperParameters& _Parameters,
        const DLogger& _Logger = _D_Dragonian_Lib_Onnx_Super_Resolution_Space GetDefaultLogger()
    ) : SuperResolutionBase(_Environment, _Parameters, _Logger)
    {

    }

    std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
        int64_t _BatchSize = 1
    ) const override;
};

class SuperResolutionBHWCRGB : public SuperResolutionBase
{
public:
    SuperResolutionBHWCRGB(
        const OnnxRuntimeEnvironment& _Environment,
        const HyperParameters& _Parameters,
        const DLogger& _Logger = _D_Dragonian_Lib_Onnx_Super_Resolution_Space GetDefaultLogger()
    ) : SuperResolutionBase(_Environment, _Parameters, _Logger)
    {

    }

    std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
        int64_t _BatchSize = 1
    ) const override;
};

class SuperResolutionBCRGBAHW : public SuperResolutionBase
{
public:
    SuperResolutionBCRGBAHW(
        const OnnxRuntimeEnvironment& _Environment,
        const HyperParameters& _Parameters,
        const DLogger& _Logger = _D_Dragonian_Lib_Onnx_Super_Resolution_Space GetDefaultLogger()
    ) : SuperResolutionBase(_Environment, _Parameters, _Logger)
    {

    }

    std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
        int64_t _BatchSize = 1
    ) const override;
};

class SuperResolutionBHWCRGBA : public SuperResolutionBase
{
public:
    SuperResolutionBHWCRGBA(
        const OnnxRuntimeEnvironment& _Environment,
        const HyperParameters& _Parameters,
        const DLogger& _Logger = _D_Dragonian_Lib_Onnx_Super_Resolution_Space GetDefaultLogger()
    ) : SuperResolutionBase(_Environment, _Parameters, _Logger)
    {

    }

    std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64> Infer(
        const std::tuple<ImageVideo::NormalizedImage5D, Int64, Int64>& _Image,
        int64_t _BatchSize = 1
    ) const override;
};

_D_Dragonian_Lib_Lib_Super_Resolution_End