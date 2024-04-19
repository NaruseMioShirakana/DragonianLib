#include "Tensor/Tensor.h"
#include <iostream>

void PrintTensor(libsvc::Tensor& _Tensor)
{
	std::cout << '[';
	for (libsvc::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (libsvc::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << _Tensor.Item<float>({ i,j }) << ", ";
		std::cout << "]";
	}
	std::cout << "]\n";
}

int main()
{
	const float Temp[10]{ 114,514,1919,810,1453,721,996,7,1919,810 };
	libsvc::Tensor Ten({ 2,3,5 });
	Ten.Assign(Temp, sizeof(Temp));
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto TenA = Ten[0];
	TenA.Slice({ {0,-1, 1 },{0,1,1} }).Assign(1.f);
	TenA.Slice({ {0,-1, 1 },{1,2,1} }).Assign(2.f);
	TenA.Slice({ {0,-1, 1 },{2,3,1} }).Assign(3.f);
	TenA.Slice({ {0,-1, 1 },{3,4,1} }).Assign(4.f);
	TenA.Slice({ {0,-1, 1 },{4,5,1} }).Assign(5.f);
	TenA = Ten[1];
	TenA.Slice({ {0,-1, 1 },{0,1,1} }).Assign(6.f);
	TenA.Slice({ {0,-1, 1 },{1,2,1} }).Assign(7.f);
	TenA.Slice({ {0,-1, 1 },{2,3,1} }).Assign(8.f);
	TenA.Slice({ {0,-1, 1 },{3,4,1} }).Assign(9.f);
	TenA.Slice({ {0,-1, 1 },{4,5,1} }).Assign(10.f);
	TenA = TenA[2];
	TenA = TenA[4];
	TenA = TenA[0];
	TenA = TenA[0];
	auto TenB = std::move(Ten);
	TenA = std::move(TenB);
	auto TenC = TenA.Slice({ {-1,0,-1},{0,-1, 1 },{2,-1,2} });
	TenA.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Assign(Temp, sizeof(Temp));
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenA.Permute({ 2,0,1 }).Invoke(1, PrintTensor);;
	return 0;
}