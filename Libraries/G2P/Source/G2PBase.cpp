#include "Libraries/G2P/G2PBase.hpp"

_D_Dragonian_Lib_G2P_Header

void G2PBase::Construct(const void* Parameter)
{
	Initialize(Parameter);
}

void G2PBase::Destory()
{
	Release();
}

_D_Dragonian_Lib_G2P_End