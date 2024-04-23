#include "Tensor/Tensor.h"
#include <iostream>
#include <windows.h>

void PrintTensor(libsvc::Tensor& _Tensor)
{
	for (libsvc::SizeType i = 0; i < _Tensor.Size(0); ++i)
	{
		std::cout << '[';
		for (libsvc::SizeType j = 0; j < _Tensor.Size(1); ++j)
			std::cout << _Tensor.Item<float>({ i,j }) << ", ";
		std::cout << "]\n";
	}
	std::cout << "\n";
}

int main()
{
	constexpr float Temp[10]{ 114,514,1919,810,1453,721,996,7,1919,810 };
	libsvc::Tensor Ten({ 2,3,5 }, libsvc::TensorType::Float32);
	const libsvc::Tensor Ten114514({ 1,514,1,1919 }, libsvc::TensorType::Float32);
	LARGE_INTEGER Time1, Time2, Freq;
	QueryPerformanceFrequency(&Freq);
	QueryPerformanceCounter(&Time1);
	libsvc::Tensor::Pad(Ten114514.Permute({ 2,1,3,0 }), { 19,19 });
	QueryPerformanceCounter(&Time2);
	std::cout << double(Time2.QuadPart - Time1.QuadPart) * 1000. / (double)Freq.QuadPart << '\n';
	Ten.FixOnes();
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten.Fix(114514.);
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	Ten.Assign(Temp, sizeof(Temp));
	Ten.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto TenA = Ten[0];
	TenA.Slice({ {0,-1 },{0,1} }).Assign(1.f);
	TenA.Slice({ {0,-1 },{1,2} }).Assign(2.f);
	TenA.Slice({ {0,-1 },{2,3} }).Assign(3.f);
	TenA.Slice({ {0,-1 },{3,4} }).Assign(4.f);
	TenA.Slice({ {0,-1 },{4,5} }).Assign(5.f);
	TenA = Ten[1];
	TenA.Slice({ {0,-1 },{0,1} }).Assign(6.f);
	TenA.Slice({ {0,-1 },{1,2} }).Assign(7.f);
	TenA.Slice({ {0,-1 },{2,3} }).Assign(8.f);
	TenA.Slice({ {0,-1 },{3,4} }).Assign(9.f);
	TenA.Slice({ {0,-1 },{4,5} }).Assign(10.f);
	TenA = TenA[2];
	TenA = TenA[4];
	TenA = TenA[0];
	TenA = TenA[0];
	auto TenB = std::move(Ten);
	TenA = std::move(TenB);
	auto TenC = TenA.Slice({ {-1, -1,-3},{0,-1 },{2,2,-1} });
	TenA.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	TenC.Assign(Temp, sizeof(Temp));
	TenC.Invoke(1, PrintTensor);
	std::cout << '\n';
	auto Tennnnn = TenA.Clone();
	TenA.Permute({ 2,0,1 }).Invoke(1, PrintTensor);
	std::cout << '\n';
	Tennnnn.Invoke(1, PrintTensor);
	std::cout << '\n';
	Tennnnn.Clone().Invoke(1, PrintTensor);
	return 0;
}