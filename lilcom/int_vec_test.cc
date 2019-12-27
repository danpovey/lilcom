#include "stdio.h"
#include "stdlib.h"
#include <assert.h>
#include "int_vec.h"
#include <math.h>
#include <iostream>


namespace int_math {


/*
  Pseudo-randomly initializes the contents of an IntVec
 */
template <typename I> void rand_init_vec(IntVec<I> *s) {
  /* exponent is random between -50 and 50 */
  s->exponent = ((rand() * 33333) % 100) - 50;

  int B = sizeof(I) * 8;


  for (int j = 0; j < s->dim; j++) {
    switch (rand() % 6) {
    case 0:
      // All zeros
      s->data[j] = 0;
      break;
    case 1:
      s->data[j] = (rand() * 103451 * 100313);  /* those are two primes.. aiming to
                                                 * overflow and get negatives. */
      break;
    case 2:
      s->data[j] = 1 << (rand() % B);
      break;
    case 3:
      s->data[j] = (1 << (rand() % B - 1));
      break;
    case 4:
      s->data[j] = -(1 << (rand() % B));
      break;
    case 5:
      s->data[j] = -(1 << (((rand() % B)) - 1));
      break;
    }
  }
  s->set_nrsb();
}


template <typename I> double largest_abs_value(IntVec<I> *s) {
  double largest_abs = 0.0;
  for (int i = 0; i < s->dim; i++) {
    double d = (double)(*s)[i];
    if (-d > largest_abs) largest_abs = -d;
    else if (d > largest_abs) largest_abs = d;
  }
  return largest_abs;
}


/*
  Pseudo-randomly initializes an IntScalar.
 */
template <typename I> void init_scalar(int i,
                                       IntScalar<I> *s) {
  /* exponent is random between -50 and 50 */
  s->exponent = ((i * 33333) % 100) - 50;

  int B = sizeof(I) * 8;

  switch (i % 6) {
    case 0:
      s->elem = 0;
      break;
    case 1:
      s->elem = (i * 103451 * 100313);  /* those are two primes.. aiming to
                                         * overflow and get negatives. */
      break;
    case 2:
      s->elem = 1 << (i % B);
      break;
    case 3:
      s->elem = (1 << (i % B)) - 1;
      break;
    case 4:
      s->elem = -(1 << (i % B));
      break;
    case 5:
      s->elem = -((1 << (i % B)) - 1);
      break;
  }
}



void test_constructor() {
  IntVec<int32_t> a;
  assert(a.dim == 0);
  IntVec<int32_t> b(4);
  assert(b.dim == 4 && data_is_zero(b.data, b.dim) &&
         b.exponent == kExponentOfZero && b.nrsb == 31);
  IntVec<int64_t> c(5);
  rand_init_vec(&c);
  IntVec<int32_t> d(5);
  rand_init_vec(&d);
  /* test copy() */
  copy(&c, &d);
  assert(int_math_abs((largest_abs_value(&c) - largest_abs_value(&d))) <=
         1.0e-05);

  a.check();
  b.check();
  c.check();

  c.resize(6);
  assert(c.dim == 6 && c.data[5] == 0);
  // c.check();  /* Should fail. */
}


template <typename I, typename J, typename K>
void test_compute_dot_product() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<I> v(dim);
    IntVec<J> w(dim);
    IntScalar<K> d, e;
    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        rand_init_vec(&v);
        rand_init_vec(&w);
        compute_dot_product(&v, &w, &d);
        compute_dot_product(&v, &w, &e);
        /* d and e should be identical. */
        assert(d.exponent == e.exponent && d.elem == e.elem);
        double sum = 0.0;
        for (int j = 0; j < dim; j++) {
          sum += (double)v[j] * (double)w[j];
        }
        double df = (double)d;
        /* std::cout << "Dot product of " << (std::string)v
                  << " and " << (std::string) w << " is " << df
                  << ", sum is " << sum << "... "
                  << ", nrsb are " << v.nrsb << ", " << w.nrsb
                  << ", ints are " << v.data[0] << " and " << w.data[0]
                  << ", output int is " << d.elem; */
        if (sum == 0.0) {
          assert(df == 0.0);
        } else {
          double rel_error = (df - sum) / (largest_abs_value(&v) * largest_abs_value(&w));
          /* std::cout << "Rel error is " << rel_error << "\n"; */
          assert(fabs(rel_error) < 1.0e-08);
        }
      }
    }
  }
}


void test_compute_dot_product_with_offset() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<int32_t> v(dim);
    IntVec<int32_t> w(dim);
    IntScalar<int64_t> d;
    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        int offset1 = (i1 + i2) % dim,
            offset2 = ((i1 + i2)*7) % dim,
            sub_dim = 1 + (((i1 + i2)*13) % (dim - int_math_max(offset1, offset2)));

        rand_init_vec(&v);
        rand_init_vec(&w);
        compute_dot_product(sub_dim, &v, offset1, &w, offset2, &d);
        double sum =  0.0;
        for (int j = 0; j < sub_dim; j++) {
          double prod = (double)v[j + offset1] * (double)w[j + offset2];
          sum += prod;
        }
        double den = largest_abs_value(&v) * largest_abs_value(&w);

        double df = (double)d;

        assert(sum - df <= 1.0e-05 * den);
      }
    }
  }
}



void test_compute_dot_product_flip_with_offset() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<int32_t> v(dim);
    IntVec<int32_t> w(dim);
    IntScalar<int64_t> d;
    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        int offset1 = (i1 + i2) % dim,
            offset2 = ((i1 + i2)*7) % dim,
            sub_dim = 1 + (((i1 + i2)*13) % (dim - int_math_max(offset1, offset2)));

        rand_init_vec(&v);
        rand_init_vec(&w);
        compute_dot_product_flip(sub_dim, &v, offset1, &w, offset2, &d);
        double sum =  0.0;
        for (int j = 0; j < sub_dim; j++) {
          double prod = (double)v[j + offset1] * (double)w[(sub_dim - 1) + offset2 - j];
          sum += prod;
        }
        double den = largest_abs_value(&v) * largest_abs_value(&w);

        double df = (double)d;

        assert(sum - df <= 1.0e-05 * den);
      }
    }
  }
}



void test_add_scaled_special() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<int32_t> v(dim);
    IntVec<int32_t> w(dim);
    IntScalar<int32_t> s;

    IntScalar<int64_t> d, e;

    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        rand_init_vec(&v);
        rand_init_vec(&w);
        init_scalar((i1+i2) * 17, &s);

        int v_offset = (i1 + i2) % dim,
            w_offset = ((i1 + i2)*3) % dim,
            sub_dim = 1 + (((i1 + i2)*7) % (dim - int_math_max(v_offset, w_offset)));
        assert(sub_dim > 0 && w_offset + sub_dim <= dim);


        for (int i = 0; i < dim; i++) {
          /* add_scaled_special requires that elements outside the range of
             w that will be written to, to be zero. */
          if (!(i >= w_offset && i < sub_dim + w_offset))
            w.data[i] = 0;
        }
        w.set_nrsb();

        IntVec<int32_t> w_copy(w);
        add_scaled_special(sub_dim, &s, &v, v_offset, &w, w_offset);

        double compare_den = fabs((double)s) * largest_abs_value(&v) + largest_abs_value(&w_copy),
            compare_error = 0.0;
        for (int i = 0; i < dim; i++) {
          double wi = (double)w[i],
              orig_wi = (double)w_copy[i];
          double recomputed_wi = orig_wi;
          if (i >= w_offset && i < w_offset + sub_dim) {
            recomputed_wi += (double)s * (double)v[i + v_offset - w_offset];
          }
          compare_error += fabs(recomputed_wi - wi);
        }
        if (compare_den == 0) {
          assert(compare_error == 0);
        } else {
          double rel_error = compare_error / compare_den;
          assert(rel_error < 1.0e-05);
        }
      }
    }
  }
}



void test_multiply() {
  for (int dim = 1; dim < 10; dim++) {
    IntVec<int32_t> v(dim);
    IntVec<int32_t> w(dim);

    for (int i1 = 0; i1 < 200; i1++) { /* i1,i2 are sources of randomness */
      for (int i2 = 0; i2 < 200; i2++) {
        rand_init_vec(&v);
        rand_init_vec(&w);

        IntVec<int32_t> x(dim);
        rand_init_vec(&x);  /* will be overwritten. */

        multiply(&v, &w, &x);

        double compare_den = largest_abs_value(&v) * largest_abs_value(&w),
            compare_error = 0.0;
        for (int i = 0; i < dim; i++) {
          double prod = (double)v[i] * (double)w[i],
              xi = (double)x[i],
              error = fabs(prod - xi);
          compare_error += error;
        }
        if (compare_den == 0.0) {
          assert(compare_error == 0.0);
        } else {
          assert(compare_error / compare_den < 1.0e-07);
        }
      }
    }
  }
}

void test_powers() {
  int dim = 376;
  for (int p = 1; p < 4; p++) {
    for (int j = 1; j < 25; j++) {
      int num = (j << p),
          den = (j << p) - 1;
      IntScalar<int32_t> a(num), b(den), r;
      divide(&b, &a, &r);

      IntVec<int32_t> powers(dim);
      init_vec_as_powers(&r, &powers);
      powers.check();
      for (int i = 0; i < dim; i++) {
        double ref = pow((double)r, i + 1),
            elem = powers[i],
            error = (ref - elem);
        assert(error < 1.0e-06);  /* absolute error, not relative...
                                     largest elem of this vector is 1. */
      }
    }
  }
}


void test_special_reflection_function() {
  for (int dim = 2; dim < 10; dim++) {
    for (int i = 0; i < 1000; i++) {
      IntVec<int32_t> b(dim);
      rand_init_vec(&b);

      int n = i % dim;
      if (n == 0)
        continue;

      /* special_reflection_function() requires that at entry, elements
         0 through dim - n - 1 are zero. */
      for (int j = 0; j < dim - n; j++)
        b.data[j] = 0;
      b.set_nrsb();

      IntVec<int32_t> b_orig(b);

      double b_double[10], b_double2[10];
      for (int j = 0; j < dim; j++) {
        b_double2[j] = b_double[j] = (double)b[j];
      }

      IntScalar<int32_t> s;
      if (i % 2 == 0) {
        s.elem = 3;
        s.exponent = -2;
        /* s = 3/4 */
      } else {
        s.elem = -1;
        s.exponent = -2;
        /* s = -1/4 */
      }
      int lshift = i % 30;
      s.elem <<= lshift;
      s.exponent -= lshift;


      double den = largest_abs_value(&b);
      special_reflection_function(n, &s, &b);
      b.check();

      /* now the reference version. */
      for (int j = 0; j < n; j++) {
        b_double2[dim - n - 1 + j] += b_double[dim - j - 1] * (double)s;
      }

      double error = 0.0;
      for (int j = 0; j < dim; j++) {
        error += fabs(b_double2[j] - (double)b[j]);
      }

      if (den == 0.0) {
        assert(error == 0.0);
      } else {
        double rel_error = error / den;
        assert(int_math_abs(rel_error) < 1.0e-05);
      }
    }
  }
}


void test_set_elem_to() {
  int dim = 5;
  for (int i = 0; i < 1000; i++) {
    IntVec<int32_t> vec(dim);
    rand_init_vec(&vec);
    IntVec<int32_t> vec_copy(vec);
    IntScalar<int32_t> s;
    init_scalar(i, &s);
    double float_vec[5];
    for (int j = 0; j < dim; j++) {
      float_vec[j] = (double)vec[j];
    }
    int index = i % dim;
    float_vec[index] = (double)s;
    set_elem_to(&s, index, &vec);
    vec.check();
    double den = int_math_abs((double)s) + largest_abs_value(&vec_copy),
        diff = 0.0;
    for (int j = 0; j < dim; j++) {
      diff += int_math_abs((double)float_vec[j] - (double)vec[j]);
    }
    if (den == 0.0) {
      assert(diff == 0.0);
    } else {
      assert(diff / den < 1.0e-07);
    }
  }
}

}

int main() {
  using namespace int_math;
  test_constructor();
  test_compute_dot_product<int32_t, int32_t, int64_t>();
  test_compute_dot_product<int32_t, int16_t, int64_t>();
  test_compute_dot_product_with_offset();
  test_compute_dot_product_flip_with_offset();
  test_add_scaled_special();
  test_multiply();
  test_powers();
  test_special_reflection_function();
  test_set_elem_to();
}

