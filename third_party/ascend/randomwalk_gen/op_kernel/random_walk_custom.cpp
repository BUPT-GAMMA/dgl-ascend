#include "kernel_operator.h"
#include <cstdint>
#include <cstdlib>
#include <stdlib.h>
#include <sys/types.h>
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024; 
constexpr uint32_t UB_TOTAL_SIZE = 192 * 1024; 
//将个数为n的元素向上对齐到32字节。
__aicore__ uint32_t ceiltoBlock(uint32_t n, uint32_t dataSize){
    return (n * dataSize + 31) / 32 * 32 / dataSize;
}


class KernelRandomWalk {
    public:
        __aicore__ inline KernelRandomWalk() {}
        
        // Initialize kernel with CSR matrix and dense matrix pointers
        __aicore__ inline void Init(GM_ADDR seeds_data, GM_ADDR row_ptr, GM_ADDR col_ind, GM_ADDR metapath_data,
            GM_ADDR output, GM_ADDR randomNumber_data, uint32_t numSeeds, uint32_t numRows, uint32_t numCols, uint32_t numMetapath, 
            uint32_t randomNumberDataLength, uint32_t outputLength, uint32_t tileLength, uint32_t dataSize, UnPadTiling UnPadTilingData)
        {
            this->tileLength = tileLength;
            this->numSeeds = numSeeds;
            this->numMetapath = numMetapath;
            this->dataSize = dataSize;
            this->numRows = numRows;
            this->numCols = numCols;
            this->UnPadTilingData = UnPadTilingData;
            rowPtrGm.SetGlobalBuffer((__gm__ int32_t *)row_ptr, numRows);
            colIndGm.SetGlobalBuffer((__gm__ int32_t *)col_ind, numCols);
            metapathGm.SetGlobalBuffer((__gm__ int32_t *)metapath_data, numMetapath);
            //只有seedsdata, randomNumber和output进行划分,划分可以在init函数进行，也可以在process过程进行。
            randomNumberGm.SetGlobalBuffer((__gm__ int32_t *)randomNumber_data, randomNumberDataLength);
            seedsGm.SetGlobalBuffer((__gm__ int32_t *)seeds_data, numSeeds);
            outputGm.SetGlobalBuffer((__gm__ int32_t *)output, outputLength);
    
           
            pipe.InitBuffer(metapathQue, BUFFER_NUM, ceiltoBlock(numMetapath, dataSize) * dataSize);
            pipe.InitBuffer(colIndQue, BUFFER_NUM, ceiltoBlock(numCols, dataSize) * dataSize);
            pipe.InitBuffer(rowPtrQue, BUFFER_NUM, ceiltoBlock(numRows, dataSize) * dataSize);

            pipe.InitBuffer(walkQue, BUFFER_NUM, ceiltoBlock(tileLength * (1+numMetapath), dataSize) * dataSize);
            pipe.InitBuffer(randomNumberQue, BUFFER_NUM, ceiltoBlock(tileLength * numMetapath, dataSize) * dataSize);

            pipe.InitBuffer(buf1, ceiltoBlock(tileLength, dataSize) * dataSize);
            pipe.InitBuffer(buf2, ceiltoBlock(tileLength, dataSize) * dataSize);
            pipe.InitBuffer(buf3, ceiltoBlock(tileLength, dataSize) * dataSize);
            pipe.InitBuffer(buf4, ceiltoBlock(tileLength, dataSize) * dataSize);
            pipe.InitBuffer(buf5, ceiltoBlock(tileLength, dataSize) * dataSize);
            pipe.InitBuffer(maskBuf, ceiltoBlock(tileLength, 1) * dataSize);
            pipe.InitBuffer(indexBuf, ceiltoBlock(tileLength * (1+numMetapath), dataSize) * dataSize);
            pipe.InitBuffer(indexBufNew, ceiltoBlock(tileLength * (1+numMetapath), dataSize) * dataSize);
        }
        
        // Process SpMM computation for assigned rows
        __aicore__ inline void Process(int start, int len)
        {
            //块内按照tile来划分，当前规定一个tile的size为256字节，即为8个block，
            int loopTime = len / tileLength;
            for(int i = 0; i < loopTime; ++i){
                CopyIn(start + i * tileLength, tileLength);
                Compute(tileLength);
                CopyOut(start + i * tileLength, tileLength);
            }
            // 尾块处理，这里需要处理未能组织成一个block的尾数据。理论上只会在最后一个block出现。这种情况下，起始地址应该满足32字节对齐，但是datacopy搬运长度不满足。
            if(len % tileLength != 0){
                CopyIn(start + loopTime * tileLength, len - loopTime * tileLength);
                Compute(len - loopTime * tileLength);
                CopyOut(start + loopTime * tileLength, len - loopTime * tileLength);
            }
        }
    
    private:
        __aicore__ inline void CopyIn(int start, int len) {
            AscendC::LocalTensor<int32_t> rowPtrLocal = rowPtrQue.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> colIndLocal = colIndQue.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> randomNumberLocal = randomNumberQue.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> metapathLocal = metapathQue.AllocTensor<int32_t>();
            AscendC::LocalTensor<int32_t> walksLocal = walkQue.AllocTensor<int32_t>();
            //datacopy函数在搬运长度不满足32字节对齐时，会向下取整。
            AscendC::DataCopy(rowPtrLocal, rowPtrGm[0], ceiltoBlock(numRows, dataSize));
            AscendC::DataCopy(colIndLocal, colIndGm[0], ceiltoBlock(numCols, dataSize));
            AscendC::DataCopy(walksLocal, seedsGm[start], ceiltoBlock(len, dataSize));
            AscendC::DataCopy(randomNumberLocal, randomNumberGm[start * numMetapath], ceiltoBlock(len * numMetapath, dataSize));
            AscendC::DataCopy(metapathLocal, metapathGm[0], ceiltoBlock(numMetapath, dataSize));
            //直接将seeds初始值拷贝到walksLocal中。
            // AscendC::DumpTensor(seedsLocal, 77, len);
            rowPtrQue.EnQue(rowPtrLocal);
            colIndQue.EnQue(colIndLocal);
            randomNumberQue.EnQue(randomNumberLocal);
            metapathQue.EnQue(metapathLocal);
            walkQue.EnQue(walksLocal);
        }
    
        __aicore__ inline void Compute(int len)
        {
            AscendC::LocalTensor<int32_t> rowPtrLocal = rowPtrQue.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> colIndLocal = colIndQue.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> randomNumberLocal = randomNumberQue.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> metapathLocal = metapathQue.DeQue<int32_t>();
            AscendC::LocalTensor<int32_t> walksLocal = walkQue.DeQue<int32_t>();
            LocalTensor<int32_t> indexTensor = indexBuf.Get<int32_t>();
            LocalTensor<int32_t> indexTensorNew = indexBufNew.Get<int32_t>();
            for(uint32_t i = 0; i < numMetapath; i++){
                local1 = buf1.Get<int32_t>();
                local2 = buf2.Get<int32_t>();
                local3 = buf3.Get<int32_t>();
                LocalTensor<int32_t> seedsLocal = walksLocal[i * len];
                
                //DUMP_TENSOR(seedsLocal, 0, len, 39);
                Muls(local3, seedsLocal, dataSize, len);  // 计算下一步的地址
                Gather(local1, rowPtrLocal, local3.ReinterpretCast<uint32_t>(), 0, len);  
                int32_t constn = dataSize;
                Adds(local2, local3, constn, len);  // 获得下一步的度
                Gather(local2, rowPtrLocal, local2.ReinterpretCast<uint32_t>(), 0, len);
                local2 = local2 - local1;
                //DUMP_TENSOR(local2, 1, len, 39);
                //local2装有度，local1装有row-ptr, 需要通过compare函数生成mask，mask为0的位置表示该节点没有邻居，mask为1的位置表示该节点有邻居。
                //compare直接和零比较，相等即可,得到度为0的位置的mask
                if(i > 0){
                    LocalTensor<int32_t> duplicateTensor = buf4.Get<int32_t>();
                    LocalTensor<int32_t> formerWalkTensor = walksLocal[(i) * len ];
                    Duplicate<int32_t>(duplicateTensor, static_cast<int32_t>(-1), len);
                    Select(formerWalkTensor.ReinterpretCast<float>(), maskTensor, duplicateTensor.ReinterpretCast<float>(), formerWalkTensor.ReinterpretCast<float>(), AscendC::SELMODE::VSEL_CMPMASK_SPR, len);
                }
                maskTensor = maskBuf.Get<uint8_t>();
                Compares(maskTensor, local2, (int32_t)0, AscendC::CMPMODE::EQ, (uint32_t)tileLength);
                //接下来要生成随机数。随机数先用Host侧传来的，后续可以改为Kernel侧生成。随机数作为src0操作数，local2为src1操作数。
                //先调用CAST将int32_t类型转换为float类型。
                LocalTensor<float> deg_fp32 = buf2.Get<float>();
                Cast(deg_fp32, local2, RoundMode::CAST_NONE, len);
                LocalTensor<float> randomNumberLocal_fp32 = buf3.Get<float>();
                LocalTensor<int32_t> randomNumberLine = randomNumberLocal[i * len];
               // DUMP_TENSOR(randomNumberLine, 2, len, 39);
                Cast(randomNumberLocal_fp32, randomNumberLine, RoundMode::CAST_NONE, len);
                //将度中为0的位置替换为1，防止后续Fmod出错。
                AscendC::LocalTensor<float> randomSteps_fp32 = buf4.Get<float>();
                Duplicate<float>(randomSteps_fp32, static_cast<float>(1), len);
                Select(deg_fp32, maskTensor, randomSteps_fp32, deg_fp32, AscendC::SELMODE::VSEL_CMPMASK_SPR, len);
                AscendC::LocalTensor<uint8_t> sharedTensor = buf5.Get<uint8_t>();
                Fmod(randomSteps_fp32, randomNumberLocal_fp32, deg_fp32, sharedTensor, len);
                AscendC::LocalTensor<int32_t> randomSteps = buf4.Get<int32_t>();
                Cast(randomSteps, randomSteps_fp32, RoundMode::CAST_TRUNC, len);
               // DUMP_TENSOR(randomSteps, 1, len, 39);
                local1 = local1 + randomSteps;
                //接下来的gather操作需要考虑len不为32字节对齐的情况。此时，gather的dst向量首地址不对齐会报错。
                LocalTensor<int32_t> walksLocal_line = walksLocal[len * (1+i)];
                Muls(local1, local1, dataSize, len);
                //Gather需要32字节对齐，针对尾块的特殊情况，需要单独处理。此时，len 不等于tileLength.
                Gather(walksLocal_line, colIndLocal, local1.ReinterpretCast<uint32_t>(), 0, len);
                //walksLocal当前的shape为先seeds，后walk，需要调整为先walk后seed。通过scatter重排，由于不支持，用gather代替。
                
            }
           // DUMP_TENSOR(walksLocal, 111, len * (1+numMetapath), 39);
            LocalTensor<int32_t> duplicateTensor = buf4.Get<int32_t>();
            LocalTensor<int32_t> formerWalkTensor = walksLocal[(numMetapath) * len ];
            Duplicate<int32_t>(duplicateTensor, static_cast<int32_t>(-1), len);
            Select(formerWalkTensor.ReinterpretCast<float>(), maskTensor, duplicateTensor.ReinterpretCast<float>(), formerWalkTensor.ReinterpretCast<float>(), AscendC::SELMODE::VSEL_CMPMASK_SPR, len);
            //DUMP_TENSOR(walksLocal, 111, len * (1+numMetapath), 39);
            //用gather实现scatter的功能。
            CreateVecIndex(indexTensor, static_cast<int32_t>(0), 1+numMetapath);
            Muls(indexTensor, indexTensor, static_cast<int32_t>(len * dataSize), 1+numMetapath);
            uint32_t broadcastShapeIn[2] = {static_cast<uint32_t>(1), static_cast<uint32_t>(ceiltoBlock(1+numMetapath, dataSize))};
            uint32_t broadcastShapeOut[2] = {static_cast<uint32_t>(len), static_cast<uint32_t>(ceiltoBlock(1+numMetapath, dataSize))};
            LocalTensor<uint8_t> sharedTensor = colIndLocal.ReinterpretCast<uint8_t>();
            UnPadParams unPadParams{0, static_cast<uint16_t>(ceiltoBlock(1+numMetapath, dataSize) - (1+numMetapath))};
            //在dim=2，axis=0时，要求srcShape[1]必须32B对齐。怎么处理不对齐的状况？  
            //先补齐到32B对齐。再调用unpad。
            BroadCast<int32_t, 2, 0>(indexTensorNew, indexTensor, broadcastShapeOut, broadcastShapeIn, sharedTensor);
            for(uint32_t i = 0; i < len; i++){
                Adds(indexTensorNew[(i+1) * ceiltoBlock(1+numMetapath, dataSize)], indexTensorNew, static_cast<int32_t>((i+1) * dataSize), 1+numMetapath);
            }
            sharedTensor = indexTensor.ReinterpretCast<uint8_t>();
            UnPad(indexTensorNew, indexTensorNew, unPadParams, sharedTensor, this->UnPadTilingData);
            //DUMP_TENSOR(indexTensorNew, 222, len * (1+numMetapath), 39);
            //Gather的api文档并没有禁止源操作数和目的操作数地址相同，然而，复用地址实际上会导致结果出错。
            Gather(indexTensorNew, walksLocal, indexTensorNew.ReinterpretCast<uint32_t>(), 0, len * (1+numMetapath));
            //DUMP_TENSOR(indexTensorNew, 333, len * (1+numMetapath), 39);
            DataCopy(walksLocal, indexTensorNew, len * (1+numMetapath));
            walkQue.EnQue(walksLocal);
            rowPtrQue.FreeTensor<int32_t>(rowPtrLocal);
            colIndQue.FreeTensor<int32_t>(colIndLocal);
            randomNumberQue.FreeTensor<int32_t>(randomNumberLocal);
            metapathQue.FreeTensor<int32_t>(metapathLocal);
        }
    
        __aicore__ inline void CopyOut(int start, int len){
            AscendC::LocalTensor<int32_t> walksLocal = walkQue.DeQue<int32_t>();
            // AscendC::DumpTensor(walksLocal, 130, len);
            // 每个 seed 需要 numMetapath 个节点 id
            AscendC::DataCopy(outputGm[start * (1+numMetapath)], walksLocal, ceiltoBlock(len * (1+numMetapath), dataSize));
            walkQue.FreeTensor<int32_t>(walksLocal);
        }
    
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> walkQue;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> colIndQue;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> rowPtrQue;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> metapathQue;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> randomNumberQue;

        AscendC::TBuf<AscendC::TPosition::VECCALC>     buf1;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     buf2;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     buf3;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     buf4;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     buf5;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     maskBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     indexBuf;
        AscendC::TBuf<AscendC::TPosition::VECCALC>     indexBufNew;
        AscendC::LocalTensor<int32_t>                 local1;
        AscendC::LocalTensor<int32_t>                 local2;
        AscendC::LocalTensor<int32_t>                 local3;
        AscendC::LocalTensor<uint8_t>                 maskTensor;

        AscendC::GlobalTensor<int32_t> rowPtrGm;
        AscendC::GlobalTensor<int32_t> colIndGm;
        AscendC::GlobalTensor<int32_t> metapathGm; 
        AscendC::GlobalTensor<int32_t> seedsGm; 
        AscendC::GlobalTensor<int32_t> outputGm;
        AscendC::GlobalTensor<int32_t> randomNumberGm;
    
        int32_t numSeeds;
        int32_t numRows;
        int32_t numCols;
        int32_t numMetapath;
        int32_t tileLength;
        int32_t dataSize;
        UnPadTiling UnPadTilingData;
};

extern "C" __global__ __aicore__ void random_walk_custom(GM_ADDR seeds_data, GM_ADDR row_ptr, GM_ADDR col_ind, GM_ADDR metapath_data, GM_ADDR randomNumber_data, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tiling_data, tiling);
    const uint32_t dataType = tiling_data.dataType;
    const uint32_t dataSize = tiling_data.dataSize;

    const uint32_t seedsDataLength = tiling_data.seedsDataLength;
    const uint32_t rowPtrLength = tiling_data.rowPtrLength;
    const uint32_t colIndLength = tiling_data.colIndLength;
    const uint32_t metapathDataLength = tiling_data.metapathDataLength;
    const uint32_t randomNumberDataLength = tiling_data.randomNumberDataLength;
    const uint32_t outputLength = tiling_data.outputLength;

    const uint32_t seedsDataLengthAligned = tiling_data.seedsDataLengthAligned;
    const uint32_t rowPtrLengthAligned = tiling_data.rowPtrLengthAligned;
    const uint32_t colIndLengthAligned = tiling_data.colIndLengthAligned;
    const uint32_t metapathDataLengthAligned = tiling_data.metapathDataLengthAligned;
    const uint32_t randomNumberDataLengthAligned = tiling_data.randomNumberDataLengthAligned;
    const uint32_t outputLengthAligned = tiling_data.outputLengthAligned;
    
    const uint32_t baseBlocksPerCore = tiling_data.baseBlocksPerCore;
    const uint32_t ubLeftSize = tiling_data.ubLeftSize;
    uint32_t tileLength = 1024 / dataSize; // 单次核内循环的计算量，剩余量：ubLeftSize / BUFFER_NUM / dataSize / (1 + metapathDataLength), 
    uint32_t blockDim = GetBlockNum();
    uint32_t blockLength = baseBlocksPerCore * 32 / dataSize; // 每个核处理的路径长度
    uint32_t formerBlockNum = (seedsDataLengthAligned * dataSize / 32) % blockDim;
    //给出每个核的GlobalTensor的偏移量，以及处理元素的长度。单位为元素。
    uint32_t blockIdx = GetBlockIdx();
    uint32_t startElement = blockIdx * blockLength + Std::min(blockIdx, formerBlockNum) * 32 / dataSize;
    uint32_t len = Std::min(seedsDataLength - startElement, (blockIdx < formerBlockNum) * 32 / dataSize + blockLength);
    if(len <= 0) return;
    KernelRandomWalk op;
    op.Init(seeds_data, row_ptr, col_ind, metapath_data, output,randomNumber_data,
        seedsDataLength, rowPtrLength, colIndLength, metapathDataLength, randomNumberDataLength, outputLength, tileLength, dataSize, tiling_data.UnPadTilingData);
    op.Process(startElement, len);
}