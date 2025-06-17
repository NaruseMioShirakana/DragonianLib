#include "fcpe.h"
#include "npy.h"


int main()
{
	auto Handle = DragonianLib::CudaModules::createHandle();
	auto Stream = DragonianLib::CudaProvider::createCudaStream();

	if (!Handle || !Stream)
		return -1;

	constexpr auto BatchSize = 2ull;
	constexpr auto InChannels = 128ull;
	constexpr auto OutChannels = 360ull;
	constexpr auto Length = 1024ull;
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

	auto Dict = DragonianLib::Util::LoadNumpyFileToDict(
		L"C:/DataSpace/torchfcpe/assets/fcpe"
	);

	Test.LoadModel(Dict);

	auto Time = clock();
	DragonianLib::CudaModules::FCPE::Model::CacheTensors Cache{ Input.Clone(Stream) };
	Test.Forward(Cache);
	auto Vec = Cache.output.Cpy2Host(Stream);
	DragonianLib::CudaProvider::asyncCudaStream(Stream);
	printf("%ld\n\n\n", clock() - Time);
	Cache.input.Copy(Input, Stream);
	Time = clock();
	Test.Forward(Cache);
	DragonianLib::CudaProvider::asyncCudaStream(Stream);
	printf("%ld\n", clock() - Time);
	return 0;
}
