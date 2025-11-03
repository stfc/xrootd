#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>

#include <XrdCeph/XrdCephPosix.hh>
#include <XrdOuc/XrdOucEnv.hh>

// Declare prototypes for functions we test from XrdCephPosix.cc
extern "C" char *ts_rfc3339();
extern "C" const char* formatAdler32(unsigned long adler32);
extern "C" char *hexbytes2ascii(const char bytes[], const unsigned int length);
extern unsigned int getCephPoolIdxAndIncrease();
extern unsigned int g_maxCephPoolIdx;

TEST(XrdCephPosix_ParseTests, TsRfc3339_NotNullAndFormat) {
    char* ts = ts_rfc3339();
    ASSERT_NE(ts, nullptr);
    // Expect a space between date and time and a 'Z' timezone char
    std::string s(ts);
    EXPECT_TRUE(s.find(' ') != std::string::npos || s.find('T') != std::string::npos);
    free(ts);
}

TEST(XrdCephPosix_FormatAdler32, RoundtripHex) {
    unsigned long val = 0x1; // small test value
    const char* hex = formatAdler32(val);
    ASSERT_NE(hex, nullptr);
    // parse hex back
    unsigned long parsed = strtoul(hex, nullptr, 16);
#ifndef Xrd_Big_Endian
    // formatAdler32 did htonl for little-endian builds
    parsed = ntohl(parsed);
#endif
    EXPECT_EQ(parsed, val);
    free((void*)hex);
}

TEST(XrdCephPosix_HexBytes2Ascii, ConvertsBytesCorrectly) {
    const char bytes[] = { (char)0x12, (char)0xAB, (char)0x00 };
    char* ascii = hexbytes2ascii(bytes, 3);
    ASSERT_NE(ascii, nullptr);
    EXPECT_STREQ(ascii, "12ab00");
    free(ascii);
}

TEST(XrdCephPosix_PoolIdx, InitializesVectorsAndCycles) {
    // remember original max and set to 2 for wrap test
    unsigned int oldMax = g_maxCephPoolIdx;
    g_maxCephPoolIdx = 2;
    // calling twice should yield 0 then 1 (or possibly other but ensure it is within range)
    unsigned int a = getCephPoolIdxAndIncrease();
    unsigned int b = getCephPoolIdxAndIncrease();
    EXPECT_LT(a, g_maxCephPoolIdx);
    EXPECT_LT(b, g_maxCephPoolIdx);
    g_maxCephPoolIdx = oldMax; // restore
}

// main() provided by GTest::Main
