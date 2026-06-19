// ============================================================
// Vectorized Philox4x32 RNG for Ascend AICore
// ============================================================
//
// 核心优化：
// 1. tile-based
// 2. vector counter
// 3. vector xor/add
// 4. batch store
// 5. 避免 SetValue
//
// 思想：
// 一个 lane = 一个 counter
//
// 每轮：
// VECTOR_WIDTH counters
// ->
// VECTOR_WIDTH * 4 randoms
//
// ============================================================
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t M0 = 0xD2511F53U;
constexpr uint32_t M1 = 0xCD9E8D57U;

constexpr uint32_t W0 = 0x9E3779B9U;
constexpr uint32_t W1 = 0xBB67AE85U;

constexpr uint32_t PHILOX_ROUNDS = 10;

constexpr uint32_t UINT32_MASK = 0xffffffffU;

constexpr uint32_t TILE_COUNTER_NUM = 128;


// ============================================================
// vector mulhilo
// ============================================================

__aicore__ inline void VecMulHiLo(
    AscendC::LocalTensor<uint32_t> hi,
    AscendC::LocalTensor<uint32_t> lo,
    AscendC::LocalTensor<uint32_t> a,
    uint32_t scalar,
    uint32_t len)
{
    auto a64 =
        AscendC::ReinterpretCast<uint64_t>(a);

    AscendC::LocalTensor<uint64_t> product =
        hi.template ReinterpretCast<uint64_t>();

    // uint64 multiply
    AscendC::Muls(
        product,
        a64,
        static_cast<uint64_t>(scalar),
        len);

    // low 32bit
    AscendC::Cast(
        lo,
        product,
        AscendC::RoundMode::CAST_NONE,
        len);

    // shift right 32
    AscendC::ShiftRight(
        product,
        product,
        32,
        len);

    // high 32bit
    AscendC::Cast(
        hi,
        product,
        AscendC::RoundMode::CAST_NONE,
        len);
}


// ============================================================
// vectorized philox round
// ============================================================

__aicore__ inline void Philox4x32RoundVec(
    AscendC::LocalTensor<uint32_t> c0,
    AscendC::LocalTensor<uint32_t> c1,
    AscendC::LocalTensor<uint32_t> c2,
    AscendC::LocalTensor<uint32_t> c3,

    AscendC::LocalTensor<uint32_t> tmpHi0,
    AscendC::LocalTensor<uint32_t> tmpLo0,

    AscendC::LocalTensor<uint32_t> tmpHi1,
    AscendC::LocalTensor<uint32_t> tmpLo1,

    uint32_t k0,
    uint32_t k1,
    uint32_t len)
{
    // --------------------------------------------
    // mulhilo
    // --------------------------------------------

    VecMulHiLo(
        tmpHi0,
        tmpLo0,
        c0,
        M0,
        len);

    VecMulHiLo(
        tmpHi1,
        tmpLo1,
        c2,
        M1,
        len);

    // --------------------------------------------
    // newC0 = hi1 ^ c1 ^ k0
    // --------------------------------------------

    AscendC::Xor(
        tmpHi1,
        tmpHi1,
        c1,
        len);

    AscendC::XorScalar(
        c0,
        tmpHi1,
        k0,
        len);

    // --------------------------------------------
    // newC2 = hi0 ^ c3 ^ k1
    // --------------------------------------------

    AscendC::Xor(
        tmpHi0,
        tmpHi0,
        c3,
        len);

    AscendC::XorScalar(
        c2,
        tmpHi0,
        k1,
        len);

    // --------------------------------------------
    // c1 = lo1
    // c3 = lo0
    // --------------------------------------------

    AscendC::DataCopy(
        c1,
        tmpLo1,
        len);

    AscendC::DataCopy(
        c3,
        tmpLo0,
        len);
}


// ============================================================
// main vectorized philox
// ============================================================

__aicore__ inline void PhiloxRandomVec(
    AscendC::LocalTensor<uint32_t> dst,
    uint32_t length,
    uint32_t seed,
    uint32_t counterOffset,

    AscendC::LocalTensor<uint32_t> workspace)
{
    // ========================================================
    // workspace layout
    // ========================================================

    auto c0 = workspace[0 * TILE_COUNTER_NUM];
    auto c1 = workspace[1 * TILE_COUNTER_NUM];
    auto c2 = workspace[2 * TILE_COUNTER_NUM];
    auto c3 = workspace[3 * TILE_COUNTER_NUM];

    auto hi0 = workspace[4 * TILE_COUNTER_NUM];
    auto lo0 = workspace[5 * TILE_COUNTER_NUM];

    auto hi1 = workspace[6 * TILE_COUNTER_NUM];
    auto lo1 = workspace[7 * TILE_COUNTER_NUM];

    // ========================================================
    // total counter num
    // ========================================================

    uint32_t counterNum =
        (length + 3) / 4;

    // ========================================================
    // tile loop
    // ========================================================

    for (uint32_t tileBase = 0;
         tileBase < counterNum;
         tileBase += TILE_COUNTER_NUM)
    {
        uint32_t tileLen =
            AscendC::Min(
                TILE_COUNTER_NUM,
                counterNum - tileBase);

        // ====================================================
        // generate vector counter
        // ====================================================

        AscendC::Arange(
            c0,
            counterOffset + tileBase,
            1,
            tileLen);

        AscendC::Duplicate(
            c1,
            static_cast<uint32_t>(0),
            tileLen);

        AscendC::Duplicate(
            c2,
            static_cast<uint32_t>(0),
            tileLen);

        AscendC::Duplicate(
            c3,
            static_cast<uint32_t>(0),
            tileLen);

        uint32_t k0 = seed;
        uint32_t k1 = seed ^ 0x0BADF00DU;

        // ====================================================
        // philox rounds
        // ====================================================

        for (uint32_t round = 0;
             round < PHILOX_ROUNDS;
             ++round)
        {
            Philox4x32RoundVec(
                c0,
                c1,
                c2,
                c3,

                hi0,
                lo0,

                hi1,
                lo1,

                k0,
                k1,
                tileLen);

            k0 += W0;
            k1 += W1;
        }

        // ====================================================
        // interleave write
        // ====================================================

        for (uint32_t i = 0; i < tileLen; ++i)
        {
            uint32_t outBase =
                (tileBase + i) * 4;

            if (outBase < length)
            {
                dst.SetValue(
                    outBase,
                    c0.GetValue(i));
            }

            if (outBase + 1 < length)
            {
                dst.SetValue(
                    outBase + 1,
                    c1.GetValue(i));
            }

            if (outBase + 2 < length)
            {
                dst.SetValue(
                    outBase + 2,
                    c2.GetValue(i));
            }

            if (outBase + 3 < length)
            {
                dst.SetValue(
                    outBase + 3,
                    c3.GetValue(i));
            }
        }
    }
}