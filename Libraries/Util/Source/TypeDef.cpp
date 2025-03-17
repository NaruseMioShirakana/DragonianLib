#include "../TypeDef.h"

_D_Dragonian_Lib_Space_Begin

float16_t::float16_t(float _Val) noexcept
{
	Val = float32_to_float16(*(uint32_t*)(&_Val)).Val;
}

float16_t& float16_t::operator=(float _Val) noexcept
{
	Val = float32_to_float16(*(uint32_t*)(&_Val)).Val;
	return *this;
}

float16_t::operator float() const noexcept
{
	auto f32 = float16_to_float32(*this);
	return *(float*)(&f32);
}

float16_t float16_t::float32_to_float16(uint32_t f32) noexcept
{
	uint32_t f16;
	uint32_t sign = (f32 & 0x80000000) >> 16;
	uint32_t exp = (f32 & 0x7F800000) >> 23;
	uint32_t frac = f32 & 0x007FFFFF;
	if (exp == 0)
	{
		f16 = sign >> 16;
	}
	else if (exp == 0xFF)
	{
		f16 = (sign | 0x7C00) >> 16;
		if (frac)
			f16 |= 1;
	}
	else
	{
		exp = (exp + 0x70) << 10;
		frac = (frac + 0x00000FFF) >> 13;
		f16 = (sign | exp | frac) >> 16;
	}
	return *reinterpret_cast<float16_t*>(&f16);
}

uint32_t float16_t::float16_to_float32(float16_t f16) noexcept
{
	uint32_t f32;
	uint32_t sign = (f16.Val & 0x8000) << 16;
	uint32_t exp = (f16.Val & 0x7C00) >> 10;
	uint32_t frac = f16.Val & 0x03FF;
	if (exp == 0)
	{
		f32 = sign;
	}
	else if (exp == 0x1F)
	{
		f32 = (sign | 0x7F800000);
		if (frac)
			f32 |= 0x007FFFFF;
	}
	else
	{
		exp = (exp - 0x70) << 23;
		frac = (frac << 13);
		f32 = (sign | exp | frac);
	}
	return f32;
}

float8_t::float8_t(float _Val) noexcept
{
	Val = float32_to_float8(*(uint32_t*)(&_Val)).Val;
}

float8_t& float8_t::operator=(float _Val) noexcept
{
	Val = float32_to_float8(*(uint32_t*)(&_Val)).Val;
	return *this;
}

float8_t::operator float() const noexcept
{
	auto f32 = float8_to_float32(*this);
	return *(float*)(&f32);
}

float8_t float8_t::float32_to_float8(uint32_t f32) noexcept
{
	uint32_t f8;
	uint32_t sign = (f32 & 0x80000000) >> 24;
	uint32_t exp = (f32 & 0x7F800000) >> 23;
	uint32_t frac = f32 & 0x007FFFFF;
	if (exp == 0)
	{
		f8 = sign >> 24;
	}
	else if (exp == 0xFF)
	{
		f8 = (sign | 0x7F) >> 24;
		if (frac)
			f8 |= 1;
	}
	else
	{
		exp = (exp + 0x70) << 1;
		frac = (frac + 0x00000FFF) >> 12;
		f8 = (sign | exp | frac) >> 24;
	}
	return *reinterpret_cast<float8_t*>(&f8);
}

uint32_t float8_t::float8_to_float32(float8_t f8) noexcept
{
	uint32_t f32;
	uint32_t sign = (f8.Val & 0x80) << 24;
	uint32_t exp = (f8.Val & 0x7F) >> 1;
	uint32_t frac = f8.Val & 0x0F;
	if (exp == 0)
	{
		f32 = sign;
	}
	else if (exp == 0x7F)
	{
		f32 = (sign | 0x7F800000);
		if (frac)
			f32 |= 0x007FFFFF;
	}
	else
	{
		exp = (exp - 0x70) << 23;
		frac = (frac << 12);
		f32 = (sign | exp | frac);
	}
	return f32;
}

bfloat16_t::bfloat16_t(float _Val) noexcept
{
	Val = float32_to_bfloat16(*(uint32_t*)(&_Val)).Val;
}

bfloat16_t& bfloat16_t::operator=(float _Val) noexcept
{
	Val = float32_to_bfloat16(*(uint32_t*)(&_Val)).Val;
	return *this;
}

bfloat16_t::operator float() const noexcept
{
	auto f32 = bfloat16_to_float32(*this);
	return *(float*)(&f32);
}

bfloat16_t bfloat16_t::float32_to_bfloat16(uint32_t f32) noexcept
{
	uint32_t bf16;
	uint32_t sign = (f32 & 0x80000000) >> 16;
	uint32_t exp = (f32 & 0x7F800000) >> 23;
	uint32_t frac = f32 & 0x007FFFFF;
	if (exp == 0)
	{
		bf16 = sign >> 16;
	}
	else if (exp == 0xFF)
	{
		bf16 = (sign | 0x7F80) >> 16;
		if (frac)
			bf16 |= 1;
	}
	else
	{
		exp = (exp + 0x70) << 7;
		frac = (frac + 0x00000FFF) >> 12;
		bf16 = (sign | exp | frac) >> 16;
	}
	return *reinterpret_cast<bfloat16_t*>(&bf16);
}

uint32_t bfloat16_t::bfloat16_to_float32(bfloat16_t bf16) noexcept
{
	uint32_t f32;
	uint32_t sign = (bf16.Val & 0x8000) << 16;
	uint32_t exp = (bf16.Val & 0x7F80) >> 7;
	uint32_t frac = bf16.Val & 0x007F;
	if (exp == 0)
	{
		f32 = sign;
	}
	else if (exp == 0x1F)
	{
		f32 = (sign | 0x7F800000);
		if (frac)
			f32 |= 0x007FFFFF;
	}
	else
	{
		exp = (exp - 0x70) << 23;
		frac = (frac << 12);
		f32 = (sign | exp | frac);
	}
	return f32;
}

_D_Dragonian_Lib_Space_End
