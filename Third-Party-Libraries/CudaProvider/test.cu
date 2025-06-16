#include "fcpe.h"


int main()
{
	auto Handle = DragonianLib::CudaModules::createHandle();
	auto Stream = DragonianLib::CudaProvider::createCudaStream();

	if (!Handle || !Stream)
		return -1;

	constexpr auto BatchSize = 2ull;
	constexpr auto InChannels = 128ull;
	constexpr auto OutChannels = 360ull;
	constexpr auto Length = 810ull;
	constexpr auto Groups = 8ull;
	constexpr auto KernelSize = 4ull;

	DragonianLib::CudaModules::Tensor<float> Input(BatchSize, InChannels, Length);
	DragonianLib::CudaModules::Tensor<float> Weight(OutChannels, InChannels / Groups, KernelSize);
	DragonianLib::CudaModules::Tensor<float> Bias(OutChannels);
	DragonianLib::CudaModules::Tensor<float> Output;

	std::vector<float> InputData(BatchSize * InChannels * Length);
	std::vector<float> WeightData(OutChannels * InChannels / Groups * KernelSize);
	std::vector<float> BiasData(OutChannels);

	for (size_t i = 0; i < InputData.size(); ++i)
		InputData[i] = float(i) / float(InputData.size());
	for (size_t i = 0; i < WeightData.size(); ++i)
		WeightData[i] = float(i) / float(WeightData.size());
	for (size_t i = 0; i < OutChannels; ++i)
		BiasData[i] = float(i) / float(OutChannels);

	if (auto Ret = setHandleStream(Handle, Stream))
	{
		fprintf(stderr, "%s\n", getErrorString(Ret));
		return -1;
	}

	Input.Handle = Handle;

	DragonianLib::CudaProvider::cpy2Device(Input.Data, InputData.data(), InputData.size(), Stream);
	DragonianLib::CudaProvider::cpy2Device(Weight.Data, WeightData.data(), WeightData.size(), Stream);
	DragonianLib::CudaProvider::cpy2Device(Bias.Data, BiasData.data(), BiasData.size(), Stream);

	DragonianLib::CudaModules::FCPE::Model Test(
		InChannels,
		OutChannels
	);

	DragonianLib::CudaModules::FCPE::Model::CacheTensors Cache;
	Test.Forward(Input, Output, Cache);
	DragonianLib::CudaProvider::asyncCudaStream(Stream);
	return 0;
}
