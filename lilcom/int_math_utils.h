#ifndef __LILCOM__INT_MATH_UTILS_H__
#define __LILCOM__INT_MATH_UTILS_H__


#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#ifndef _MSC_VER
#include <strings.h>
#endif
#include <stdlib.h>  /* for abs. */

#ifdef _MSC_VER
#include <intrin.h>

static inline int __builtin_ctz(unsigned x) {
    unsigned long ret;
    _BitScanForward(&ret, x);
    return (int)ret;
}

static inline int __builtin_ctzll(unsigned long long x) {
    unsigned long ret;
    _BitScanForward64(&ret, x);
    return (int)ret;
}

static inline int __builtin_ctzl(unsigned long x) {
    return sizeof(x) == 8 ? __builtin_ctzll(x) : __builtin_ctz((uint32_t)x);
}

static inline int __builtin_clz(unsigned x) {
    return (int)__lzcnt(x);
}

static inline int __builtin_clzll(unsigned long long x) {
    return (int)__lzcnt64(x);
}

static inline int __builtin_clzl(unsigned long x) {
    return sizeof(x) == 8 ? __builtin_clzll(x) : __builtin_clz((uint32_t)x);
}

#ifdef __cplusplus
static inline int __builtin_ctzl(unsigned long long x) {
    return __builtin_ctzll(x);
}

static inline int __builtin_clzl(unsigned long long x) {
    return __builtin_clzll(x);
}
#endif
#endif


namespace int_math {

template <typename I> inline I int_math_min(I a, I b) {
  return (a < b ? a : b);
}

template <typename I> inline I int_math_max(I a, I b) {
  return (a > b ? a : b);
}

template <typename I> inline I int_math_abs(I a) {
  return (a > 0 ? a : -a);
}



/*
   The native_clz functions return the number of leading zeros in the argument;
   HOWEVER, they are undefined for zero input.  (I think this is because of
   peculiarities/differences between processors).
   You shouldn't try to call this with uint16, as with most compilers none
   of these types will be uint16.
*/
inline int native_clz(unsigned int i) {
  return __builtin_clz(i);
}
inline int native_clz(unsigned long int i) {
  return __builtin_clzl(i);
}
inline int native_clz(unsigned long long int i) {
  return __builtin_clzll(i);
}


/* These versions of clz with fixed-size types return the number of leading
   zeros in the argument, and also give the expected results when the
   input is zero. */
inline int clz(uint16_t i) {
  return i == 0 ? 16 : native_clz((uint32_t)i) - 16;
}
inline int clz(uint32_t i) {
  return i == 0 ? 32 : native_clz(i);
}
inline int clz(uint64_t i) {
  return i == 0 ? 64 : native_clz(i);
}


/*
  Returns the number of bits we'd need to write this number down in binary,
  i.e. starting from the most significant nonzero digit, to the least.
 */
inline int num_bits(uint16_t i) {
  return 16 - clz(i);
}
inline int num_bits(uint32_t i) {
  return 32 - clz(i);
}
inline int num_bits(uint64_t i) {
  return 64 - clz(i);
}




}  // namespace int_math

#endif /* include guard */

