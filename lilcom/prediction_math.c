#include <stdio.h>
#include <stdlib.h> /* for malloc */
#include <string.h> /* for memset */
#include "fixed_math.h"
#include "prediction_math.h"
#include "lilcom_common.h"  /* for debug_fprintf */

/**
   Below is the working Python code that ToeplitzSolve is based on.
   Please see the documentation in prediction_math.h for the interface of
   this "C" function; this documentation explains its inner workings.

def toeplitz_solve(autocorr, y):
    # The technical report I am looking at deals with things of dimension (N+1),
    # so for consistency I am letting N be the dimension minus one.
    N = autocorr.shape[0] - 1
    assert y.shape[0] == N + 1
    x = np.zeros(N+1)
    b = np.zeros(N+1)
    b_temp = np.zeros(N+1)
    r = autocorr

    b[0] = 1.0
    epsilon = r[0]
    x[0] = y[0] / epsilon

    for n in range(1, N+1):

        # eqn 2.6 for \nu_n.  We are not evaluating \xi_n because it is
        # the same.  Be careful with the indexing of b.  Notice in Eq.
        # (2.3) that the elements of b are in a very strange order,
        # so you have to interpret (2.6) very carefully.
        nu_n = (-1.0 / epsilon) * sum([r[j+1] * b[j] for j in range(n)])

        # next few lines are Eq. 2.7
        b_temp[0] = 0.0
        b_temp[1:n+1] = b[:n]
        b_temp[:n] += nu_n * np.flip(b[:n])
        b[:n+1] = b_temp[:n+1]

        # Eq. 2.8
        epsilon *= (1.0 - nu_n * nu_n)
        assert abs(nu_n) < 1.0

        # The following is an unnumbered formula below Eq. 2.9
        lambda_n = y[n] - sum([ r[n-j] * x[j] for j in range(n)])
        x[:n+1] += (lambda_n / epsilon) * b[:n+1];

    return x
 */
int ToeplitzSolve(const Vector64 *autocorr_in, const Vector64 *y_in, Vector64 *x_in,
                  Vector64 *temp1, Vector64 *temp2) {
  assert(autocorr_in->dim == y_in->dim && autocorr_in->dim == x_in->dim);
  /* All regions should be distinct, except possibly autocorr and y. */
  assert(y_in->region != x_in->region && temp1->region != x_in->region &&
      temp1->region != y_in->region && temp2->region != temp1->region &&
      temp2->region != x_in->region && temp2->region != y_in->region &&
      autocorr_in->region != x_in->region && autocorr_in->region != temp1->region
             && autocorr_in->region != temp2->region);
  /* CAUTION: this N  is not the same as the N mentioned in the header.
     For consistency with the literature, all the vectors are of size
     N+1 (this is done for reasons I don't understand) and I set
     N here accordingly.
   */


  /* Copy the objects to local copies; this may save a few registers.  Also use
     slightly different names internally to in the interface (these names
     correspond to the technical report I am following)..*/
  Vector64 r = *autocorr_in,
      y = *y_in,
      x = *x_in,
      b = *temp1,
      b_temp = *temp2;
  /* Caution: b and b_temp get shallow-swapped inside the function. */

  int N = autocorr_in->dim - 1;

  CopyIntToVector64Elem(0, 1, 1, &b);  /* b[0] = 1.0 */
  Scalar64 epsilon, x0, y0;
  CopyVectorElemToScalar64(&r, 0, &epsilon);  /* epsilon = r[0] */

  CopyVectorElemToScalar64(&y, 0, &y0); /* next 3 lines: x[0] = y[0] / epsilon. */
  DivideScalar64(&y0, &epsilon, &x0);
  CopyScalarToVector64Elem(&x0, 0, &x);

  for (int n = 1; n <= N; n++) {  /* for n in range(1, N+1): */

    /* New few lines:
       nu_n = (-1.0 / epsilon) * sum([r[j+1] * b[j] for j in range(n)])
         ( == (-1.0 / epsilon * (np.dot(r[1:n+1], b[:n]))
    */
    Vector64 r1n1, bn;
    InitSubVector64(&r, 1, n, 1, &r1n1);
    InitSubVector64(&b, 0, n, 1, &bn);
    Scalar64 product, nu_n;
    DotVector64(&r1n1, &bn, &product);
    DivideScalar64(&product, &epsilon, &nu_n);
    NegateScalar64(&nu_n);

    CopyIntToVector64Elem(0, 0, 0, &b_temp);  /* b_temp[0] = 0.0 (3rd arg is size_hint==0). */
    Vector64 b_temp_1n1; /* == b_temp[1:n+1] */
    InitSubVector64(&b_temp, 1, n, 1, &b_temp_1n1);
    CopyVector64(&bn, &b_temp_1n1); /* b_temp[1:n+1] = b[:n] */
    CheckRegion64Size(b_temp.region);
    Vector64 b_temp_n; /* == b[0:n] */
    InitSubVector64(&b_temp, 0, n, 1, &b_temp_n);
    Vector64 b_n_flip; /* == np.flip(b[:n]) */
    InitSubVector64(&b, n - 1, n, -1, &b_n_flip);
    CheckRegion64Size(b_temp.region);
    AddScalarVector64(&nu_n, &b_n_flip, &b_temp_n); /* b_temp[:n] += nu_n * np.flip(b[:n]) */

    /* Shallow-swap the vectors b and b_temp.  In the equations this is
       written as:
         b[:n+1] = b_temp[:n+1]
         but we do it via shallow swap. */
    { Vector64 temp = b; b = b_temp; b_temp = temp; }

    /* The next few lines will do:
         epsilon *= (1.0 - nu_n * nu_n) */
    Scalar64 nu_n2,  /* nu_n^2 */
        epsilon_minus_nu_n2;
    CheckScalar64Size(&nu_n);
    MulScalar64(&nu_n, &nu_n, &nu_n2); /* nu_n2 := nu_n * nu_n */
    CheckScalar64Size(&nu_n2);
    NegateScalar64(&nu_n2);   /* nu_n2 *= -1 */
    CheckScalar64Size(&nu_n2);
    MulScalar64(&epsilon, &nu_n2, &epsilon_minus_nu_n2);
    AddScalar64(&epsilon, &epsilon_minus_nu_n2, &epsilon); /* epsilon -= mu_n*mu_n*epsilon */
    if (epsilon.data <= 0) {
      debug_fprintf(stderr, "Negative or zero epilon %f in Toeplitz computation (n=%d)\n",
                    (float)Scalar64ToDouble(&epsilon), n);
      return 1;
    }

    /* we'll be computing:
       lambda_n = y[n] - sum([ r[n-j] * x[j] for j in range(n)])
       ratio = epsilon / lambda_n
    */
    Scalar64 ratio;
    {
      Scalar64 lambda_n, y_elem_n;
      Vector64 x_n, r_1n1_flip;
      InitSubVector64(&x, 0, n, 1, &x_n);
      InitSubVector64(&r, n, n, -1, &r_1n1_flip);
      DotVector64(&x_n, &r_1n1_flip, &lambda_n);
      NegateScalar64(&lambda_n);
      /* lambda_n is now
         - sum([ r[n-j] * x[j] for j in range(n)]) == np.dot(np.flip(r[1:n+1]), j[:n]) */
      CopyVectorElemToScalar64(&y, n, &y_elem_n);
      AddScalar64(&y_elem_n, &lambda_n, &lambda_n); /* lambda_n += y[n]. */
      DivideScalar64(&lambda_n, &epsilon, &ratio); /* ratio = lambda_n / epsilon */
    }

    Vector64 x_n1, b_n1;
    InitSubVector64(&x, 0, n+1, 1, &x_n1);
    InitSubVector64(&b, 0, n+1, 1, &b_n1);
    /* next line: x[:n+1] += (lambda_n / epsilon) * b[:n+1] */
    AddScalarVector64(&ratio, &b_n1, &x_n1);
  }

  PrintVector64("x is: ", &x);

  return 0;
}



int LpcStatsInit(int lpc_order,
                 int max_block_size,
                 const Scalar64 *eta,
                 struct LpcStats *stats) {
  stats->lpc_order = lpc_order;
  stats->max_block_size = max_block_size;
  stats->T = 0;

  int autocorr_size = (lpc_order + 1),
      x_hat_size = (lpc_order + max_block_size),
      sqrt_scale_size = 2 * (lpc_order > max_block_size ? lpc_order : max_block_size),
      inv_sqrt_scale_size = 2 * lpc_order + 3,
      initial_samples_size = lpc_order,
      tot_size = 2*autocorr_size + x_hat_size + sqrt_scale_size + inv_sqrt_scale_size + initial_samples_size;
  stats->allocated_block = malloc((tot_size * sizeof(int64_t)) +
                                  (lpc_order * sizeof(int16_t)));
  if (!stats->allocated_block)
    return 1;  /* error */
  memset(stats->allocated_block, 0, tot_size * sizeof(int64_t));
  int64_t *data = (int64_t*)stats->allocated_block;
  int exponent = 0, size_hint = -1;
  InitRegionAndVector64(data, autocorr_size, exponent, size_hint,
                        &stats->autocorr_region, &stats->autocorr);
  data += autocorr_size;
  InitRegionAndVector64(data, autocorr_size, exponent, size_hint,
                        &stats->autocorr_tmp_region, &stats->autocorr_tmp);
  data += autocorr_size;
  InitRegionAndVector64(data, x_hat_size, exponent, size_hint,
                        &stats->x_hat_region, &stats->x_hat);
  data += x_hat_size;
  InitRegionAndVector64(data, sqrt_scale_size, exponent, size_hint,
                        &stats->sqrt_scale_region, &stats->sqrt_scale);
  data += sqrt_scale_size;
  InitRegionAndVector64(data, inv_sqrt_scale_size, exponent, size_hint,
                        &stats->inv_sqrt_scale_region, &stats->inv_sqrt_scale);
  data += inv_sqrt_scale_size;
  InitRegionAndVector64(data, initial_samples_size, exponent, size_hint,
                        &stats->initial_samples_region, &stats->initial_samples);
  data += initial_samples_size;

  stats->context_buf = (int16_t*)data;

  stats->eta = *eta;
  MulScalar64(&stats->eta, &stats->eta, &stats->eta_2);
  InvertScalar64(&stats->eta_2, &stats->eta_m2);

  InitScalar64FromInt(1, &stats->eta_2TN);


  { /* Initialize stats->sqrt_scale. */
    Scalar64 cur_eta_power = *eta;

    for (int i = sqrt_scale_size - 1; i >= 0; i--) {
      CopyScalarToVector64Elem(&cur_eta_power, i, &stats->sqrt_scale);
      if (i == sqrt_scale_size - 1) {
        /* shift to a very large size (60) to avoid further left shifts in this
         * loop  */
        ShiftRegion64ToSize(60, &stats->sqrt_scale_region);
      }
      MulScalar64(eta, &cur_eta_power, &cur_eta_power);
    }
    /* The region to shift it to 60 - 15 now, is to ensure that
       when multiplied by a 16-bit number, these data elements
       still have size <= 60, which will avoid extra shifts.
       [note: largest-magnitude 16-bit number is -2^15, which has
       'size' 15].
    */
    ShiftRegion64ToSize(60 - 15, &stats->sqrt_scale_region);
  }

  { /* Initialize stats->inv_sqrt_scale. */
    Scalar64 inv_eta;
    InvertScalar64(eta, &inv_eta);
    Scalar64 cur_inv_eta_power;
    InitScalar64FromInt(1, &cur_inv_eta_power);
    for (int i = 0; i < inv_sqrt_scale_size; i++) {
      CopyScalarToVector64Elem(&cur_inv_eta_power, i, &stats->inv_sqrt_scale);
      if (i == sqrt_scale_size - 1) {
        /* shift to a very large size (60) to avoid further left shifts in this
         * loop  */
        ShiftRegion64ToSize(60, &stats->inv_sqrt_scale_region);
      }
      MulScalar64(&inv_eta, &cur_inv_eta_power, &cur_inv_eta_power);
    }
  }

  return 0;  /* Success */
}

/**
   This function updates stats->initial_samples and stats->context_buf as
   needed.
 */
static void LpcStatsUpdateContext(int block_size,
                                  int16_t *data,
                                  struct LpcStats *stats) {
  int cur_T = stats->T,
      end_T = cur_T + block_size,
      lpc_order = stats->lpc_order,
      t;
  int start_t = end_T - lpc_order;
  if (cur_T > start_t)
    start_t = cur_T;

  for (t = start_t; t < end_T; t++) {
    stats->context_buf[t % lpc_order] = data[t - cur_T];
    fprintf(stderr, "Setting context-buf[%d %% %d = %d] to %d\n",
            t, lpc_order, t % lpc_order, (int)(data[t - cur_T]));
  }
}



/**
   This function updates stats->x_hat to contain `lpc_order` samples of previous
   history followed by `block_size` samples taken from `data`.
   It does not update stats->T.
 */
static void LpcStatsUpdateXhat(int block_size,
                               int16_t *data,
                               struct LpcStats *stats) {
  assert(block_size <= stats->max_block_size);
  int sqrt_scale_dim = stats->sqrt_scale.dim,
      lpc_order = stats->lpc_order;
  int T = stats->T,
      t_start = T - lpc_order,
      t_end = T + block_size;

  /* sqrt_scale_offset is a number that, when added to a 't' value,
     gives us the appropriate index into sqrt_scale so that the
     next sample (at t = t_end) would be indexed `sqrt_scale_dim`. */

  /* The following loops sets up the actual data inside stats->x_hat
     (but not its metadata) yet).  For correctness it relies on the
     fact that we did:
       ShiftRegion64ToSize(60 - 15, &stats->sqrt_scale_region);
   */
  for (int t = t_start; t < t_end; t++) {
    int16_t sample;
    if (t < T) sample = stats->context_buf[t % lpc_order];
    else sample = data[t - T];
    stats->x_hat.data[t - t_start] = sample * stats->sqrt_scale.data[sqrt_scale_dim + t - t_end];
  }

  {  /* This block sets up the metadata of stats->x_hat */
    /* What the following does is to say, treat 16-bit numbers as integers with
       no exponent.   (This won't affect the coefficients, we just need to be consistent).
    */
    stats->x_hat_region.exponent = stats->sqrt_scale_region.exponent;

    int prev_dim = stats->x_hat.dim,
        dim = t_end - t_start;
    assert(dim <= stats->x_hat_region.dim);
    if (dim < prev_dim) {
      /* If this block is smaller than the last one, make sure the rest of the
         region's data is zero so that the 'size' isn't larger than necessary. */
      for (int i = dim; i < prev_dim; i++)
        stats->x_hat_region.data[i] = 0;
    }
    stats->x_hat.dim = dim;
    /* we don't need to change x_hat.data or stride, they will be
     * x_hat_region.data and 1. */

    /* 15/2 is a guess at the size of the residuals (between 0 and 15) */
    int size_hint = stats->sqrt_scale_region.size + (15/2);
    /* Recompute the size exactly to preserve as much precision as possible;
       this may not actually be necessary. */
    SetRegion64Size(size_hint, stats->x_hat.region);
  }
  PrintVector64("Xhat", &stats->x_hat);
}


/*
   This function updates stats->autocorr (the autocorrelation of x_hat, in
   stats->x_hat).  It is analogous to _update_autocorr_stats in class LpcStats
   in ../test/linear_prediction.py.
 */
static void LpcStatsUpdateAutocorrStats(struct LpcStats *stats) {
  int N = stats->lpc_order,
      block_size = stats->x_hat.dim - N;
  /* stats->x_hat contains N samples of left-context then block_size
     samples for which we need the stats. */

  /*
    Python code (simplified) is this:
     for t in range(N, N + block_size):
        for k in range(N+1):
           autocorr_stats[k] += x_hat[t-k] * x_hat[t]
    x_hat_mat's row index will correspond to 'k' and its column index will
    correspond to 't'.  Its rows and columns will overlap.
  */
  Matrix64 x_hat_mat;
  x_hat_mat.region = stats->x_hat.region;
  x_hat_mat.num_rows = N + 1;
  x_hat_mat.num_cols = block_size;
  x_hat_mat.row_stride = -1;
  x_hat_mat.col_stride = 1;
  x_hat_mat.data = stats->x_hat.data + N;

  Vector64 x_hat_vec;
  x_hat_vec.region = stats->x_hat.region;
  x_hat_vec.dim = block_size;
  x_hat_vec.stride = 1;
  x_hat_vec.data = stats->x_hat.data + N;

  Scalar64 old_stats_scale;
  /* eta ** (2 * block_size) */
  CopyVectorElemToScalar64(&stats->sqrt_scale,
                           stats->sqrt_scale.dim - (2 * block_size),
                           &old_stats_scale);
  /* autocorr_temp += (eta ** (2 * block_size)) * autocorr */
  SetScalarVector64(&old_stats_scale, &stats->autocorr, &stats->autocorr_tmp);
  AddMatrixVector64(&x_hat_mat, &x_hat_vec, &stats->autocorr_tmp);

  /* shallow-swap stats->autocorr and stats->autocorr_tmp. */
  { Vector64 tmp = stats->autocorr_tmp; stats->autocorr_tmp = stats->autocorr; stats->autocorr = tmp; }

}

static void LpcStatsUpdateInitialSamples(int block_size,
                                         const int16_t *data,
                                         struct LpcStats *stats)  {
  int N = stats->lpc_order;
  if (stats->T < N) {
    int start_t = stats->T,
        end_t = stats->T + block_size;
    if (end_t >= N)
      end_t = N;
    stats->initial_samples_region.exponent = 0;  /* they are just integers */
    for (int t = start_t; t < end_t; t++) {
      stats->initial_samples.data[t] = data[t - start_t];
    }
    if (end_t >= N) {
      int size_hint = 8;
      SetRegion64Size(size_hint, &stats->initial_samples_region);
    }
  }
}

static void LpcStatsUpdateEta2N(int block_size,
                                struct LpcStats *stats) {
  if (stats->eta_2TN.exponent < -1000)
    return;  /* It's already effectively zero. */
  int N = stats->lpc_order;
  if (stats->T < N) {  /* we skip the first N samples. */
    if (stats->T + block_size <= N)
      return;
    else
      block_size -= (N - stats->T);
  }
  assert(block_size > 0);
  Scalar64 eta_2_block_size;
  CopyVectorElemToScalar64(&stats->sqrt_scale,
                           stats->sqrt_scale.dim - 2 * block_size,
                           &eta_2_block_size);
  MulScalar64(&stats->eta_2TN, &eta_2_block_size, &stats->eta_2TN);
}

void LpcStatsAcceptBlock16(int block_size,
                           int16_t *data,
                           struct LpcStats *stats) {
  LpcStatsUpdateXhat(block_size, data, stats);
  LpcStatsUpdateInitialSamples(block_size, data, stats);
  LpcStatsUpdateAutocorrStats(stats);
  LpcStatsUpdateContext(block_size, data, stats);
  LpcStatsUpdateEta2N(block_size, stats);
  stats->T += block_size;
}

void LpcStatsDestroy(struct LpcStats *stats) {
  free(stats->allocated_block);
}

int LpcStatsAuxInit(struct LpcStats *stats,
                    int lpc_order,
                    struct LpcStatsAux *aux) {
  assert(lpc_order > 0 && lpc_order <= stats->lpc_order);
  aux->stats = stats;
  aux->lpc_order = lpc_order;
  int N = lpc_order, N1 = N + 1,
      tot_size = 3 * (N1 * N1)  +  N1 + N,
      tot_size_bytes = tot_size * sizeof(int64_t);
  stats->allocated_block = malloc(tot_size_bytes);
  if (stats->allocated_block == NULL)
    return 1;  /* Error */
  memset(stats->allocated_block, 0, tot_size_bytes);
  int exponent = 0, size_hint = -1;

  int64_t *data = (int64_t*)stats->allocated_block;
  InitRegionAndMatrix64(data, N1, N1, exponent, size_hint,
                        &aux->A_minus_region, &aux->A_minus);
  data += N1 * N1;
  InitRegionAndMatrix64(data, N1, N1, exponent, size_hint,
                        &aux->A_plus_region, &aux->A_plus);
  data += N1 * N1;
  InitRegionAndMatrix64(data, N1, N1, exponent, size_hint,
                        &aux->A_region, &aux->A);
  data += N1 * N1;
  InitRegionAndVector64(data, N1, exponent, size_hint,
                        &aux->autocorr_reflected_region,
                        &aux->autocorr_reflected);
  data += N1;
  InitRegionAndVector64(data, N, exponent, size_hint,
                        &aux->x_context_region,
                        &aux->x_context);
  data += N;
  assert(data == tot_size + (int64_t*)stats->allocated_block);
  return 0;  /* Success */
}


void LpcStatsAuxDestroy(struct LpcStatsAux *stats_aux){
  free(stats_aux->allocated_block);
}

/* Computes A^+ from the writeup (terms that need to be subtracted due
   to beginning-of-sequence effects). */
void LpcStatsAuxComputeAMinus(struct LpcStatsAux *aux) {
  const struct LpcStats *stats = aux->stats;
  int N = aux->lpc_order;
  ZeroRegion64(&aux->A_minus_region);
  /*
    Python code was as follows, where `x_hat` was actually
    the first samples * (eta ** -np.arange(lpc_order)),
    which is the same as stats->initial_samples[:lpc_order].
            for j in range(N-1, -1, -1):  # for j in [N-1, N-2, .. 0]
                A_minus[j,:N] = ((self.eta ** 2) * A_minus[j+1,1:] +
                                 (self.eta ** -(j+lpc_order)) * self._get_sqrt_scale(N) * x_hat[N-1-j] * np.flip(x_hat))
  */

  Vector64 x_hat;  /* Note this is not really x_hat, it's a scaled version of
                      the first few samples of x_hat.  Need sub-vector in case
                      aux->lpc_order < stats->lpc_order. */
  InitSubVector64(&stats->x_hat, 0, N, 1, &x_hat);

  Vector64 initial_samples_flipped;
  InitSubVector64(&stats->initial_samples, N-1, N, -1, &initial_samples_flipped);

  for (int j = N-1; j >= 0; j--) {
    Vector64 A_minus_j_N, /* A_minus[j,:N] */
        A_minus_j1_1;  /* A_minus[j+1,1:] */
    InitRowVector64(&aux->A_minus, j, 0, N, &A_minus_j_N);
    InitRowVector64(&aux->A_minus, j + 1, 1, N, &A_minus_j1_1);
    /* The next line would be, in Python, A_minus[j,:N] += ((self.eta ** 2) *
     * A_minus[j+1,1:]  */
    SetScalarVector64(&stats->eta_2, &A_minus_j1_1,  &A_minus_j_N);
    /* The next line would be, in Python, A_minus[j,:N] += ((self.eta ** 2) *
     * A_minus[j+1,1:])  */
    SetScalarVector64(&stats->eta_2, &A_minus_j1_1,  &A_minus_j_N);

    Scalar64 product;
    CopyVectorElemToScalar64(&stats->initial_samples, N-1-j, &product);
    MulScalar64(&stats->eta_2, &product, &product);
    /* The next line would be, in Python,
        A_minus[j,:N] += ((self.eta ** 2) *
              initial_samples[N-1-j] * np.flip(initial_samples)). */
    AddScalarVector64(&product, &initial_samples_flipped,  &A_minus_j_N);
  }
  /* We are still missing a factor of eta^2(T-N); we'll add that later. */
}


/* Computes A^+ from the writeup (terms that need to be subtracted due
   to end effects). */
void LpcStatsAuxComputeAPlus(struct LpcStatsAux *aux) {
  const struct LpcStats *stats = aux->stats;
  int N = aux->lpc_order;
  ZeroRegion64(&aux->A_plus_region);
  int T = stats->T;
  assert(T >= N);

  /* Set up aux->x_context. */
  int16_t max_abs_value = 0;
  for (int t = T - N; t < T; t++) {
    /* Put the samples in reversed order because that's how we'll need them. */
    int16_t x = stats->context_buf[t % stats->lpc_order],
        abs_x = (x >= 0 ? x : -x);
    if (abs_x > max_abs_value)
      max_abs_value = abs_x;
    fprintf(stderr, "t=%d, setting x-context data to %d\n", t, x);
    aux->x_context.data[T-1-t] = x;
  }
  aux->x_context_region.exponent = 0;  /* they are just integers */
  int size_hint = 8;
  /* We're finding the size `manually` rather than using SetRegion64Size(),
   * since it's faster this way. */
  aux->x_context_region.size = FindSize(size_hint, (uint64_t)max_abs_value);

  PrintMatrix64("APlus[a] ", &aux->A_plus);
  for (int j = 1; j <= N; j++) {
    /* Python code was: A_plus[j,1:] = (self.eta ** -2 * (A_plus[j-1,:-1]) +  x[T-j] * np.flip(x[T-N:T])) */
    Vector64 A_plus_j_1,  /* A_plus[j,1:] */
        A_plus_jm1; /* A_plus[j-1,:-1] */
    InitRowVector64(&aux->A_plus, j,     1, N, &A_plus_j_1);
    InitRowVector64(&aux->A_plus, j - 1, 0, N, &A_plus_jm1);
    PrintVector64("APlusjm1", &A_plus_jm1);
    PrintVector64("APlusj1", &A_plus_j_1);

    /* do: A_plus[j,1:] = ((self.eta ** -2) * (A_plus[j-1,:-1])) */
    SetScalarVector64(&stats->eta_m2, &A_plus_jm1, &A_plus_j_1);
    Scalar64 x_Tj;  /* in Python this was: x[T-j].  Since aux->x_context is
                       stored in reversed order, this is addressed with index
                       j-1. */
    CopyVectorElemToScalar64(&aux->x_context, j - 1, &x_Tj);
    /* do: A_plus[j,1:] += x[T-j] * np.flip(x[T-N]:T) */
    AddScalarVector64(&x_Tj, &aux->x_context, &A_plus_j_1);
  }
}

static void LpcStatsAuxComputeAAll(struct LpcStatsAux *aux) {
  const struct LpcStats *stats = aux->stats;
  int N = aux->lpc_order;

  Vector64 autocorr;
  /* Take sub-vector in case aux->lpc_order < stats->lpc_order. */
  InitSubVector64(&stats->autocorr, 0, N, 1, &autocorr);

  /* A^all will just go in "A"; we'll later subtract the other parts from it. */
  ZeroRegion64(&aux->A_region);

  for (int j = 0; j <= N; j++) {
    for (int k = 0; k <= N; k++) {
      /* formula in Python is:
         A_all[j,k] = (self.eta ** -(j+k)) * self.autocorr[abs(j-k)] */
      Scalar64 eta_mjk;  /* eta ** -(j+k) */
      CopyVectorElemToScalar64(&stats->inv_sqrt_scale,
                               (j + k), &eta_mjk);
      int diff_index = (j >= k ? j - k : k - j);
      Scalar64 autocorr_element;  /* autocorr[abs(j-k)] */
      CopyVectorElemToScalar64(&stats->autocorr, diff_index,
                               &autocorr_element);
      /* Multiply autocorr_element by eta_mjk; it's now the
         expression we need to put in A_all[j,k]. */
      MulScalar64(&eta_mjk, &autocorr_element, &autocorr_element);
      CopyScalarToMatrix64Elem(&autocorr_element, j, k, &aux->A);
    }
  }
}


static void LpcStatsAuxComputeA(struct LpcStatsAux *aux) {
  const struct LpcStats *stats = aux->stats;
  /* Computes A as A^all - A^+ - A^-.
     Currently, status_aux->A contains A^all. */
  Scalar64 factor;
  InitScalar64FromInt(-1, &factor);
  /* A -= A^+ */
  AddScalarMatrix64(&factor, &aux->A_plus, &aux->A);

  if (stats->eta_2TN.exponent < -1000)
    return;  /* A^- is too small to worry about. */
  MulScalar64(&stats->eta_2TN, &factor, &factor);
  /* next line: A -= eta^2(T-N) * A^-
     Note: aux->a_minus is actually A'^- in the writeup.
   */
  AddScalarMatrix64(&factor, &aux->A_minus, &aux->A);
}


/*
  This is explained more in the python version (../test/linear_prediction.py,
  see get_autocorr_reflected()) and in the writeup.
 */
static void LpcStatsAuxComputeReflectedAutocorr(struct LpcStatsAux *aux) {
  const struct LpcStats *stats = aux->stats;
  /* Computes A as A^all - A^+ - A^-.
     Currently, status_aux->A contains A^all. */
  ZeroRegion64(&aux->autocorr_reflected_region);
  int lpc_order = aux->lpc_order;
  /*
    Python code for this entire function was:
     ans = self.autocorr[0:lpc_order+1].copy()
     for k in range(1, lpc_order + 1):
         ans[k] += 0.5 * np.dot(self.x_hat[-k:], np.flip(self.x_hat[-k:]))
   */
  Scalar64 one_half;
  one_half.data = 1;
  one_half.exponent = -1;
  one_half.size = 0;  /* 1 <= 2^0 */

  for (int k = 1; k <= lpc_order; k++) {
    Vector64 x_hat_tail;
    InitSubVector64(&stats->x_hat, stats->x_hat.dim - k, k, 1, &x_hat_tail);
    Vector64 x_hat_tail_flip;
    InitSubVector64(&stats->x_hat, stats->x_hat.dim - 1, k, -1, &x_hat_tail_flip);

    Scalar64 dot_prod;
    DotVector64(&x_hat_tail, &x_hat_tail_flip, &dot_prod);
    MulScalar64(&dot_prod, &one_half, &dot_prod);
    CopyScalarToVector64Elem(&dot_prod, k, &aux->autocorr_reflected);
  }
  Vector64 autocorr_part;  /* may be shorter than stats->autocorr, in case
                              aux->lpc_order < stats->lpc_order */
  InitSubVector64(&stats->autocorr, 0, lpc_order + 1, 1, &autocorr_part);
  AddVector64(&autocorr_part, &aux->autocorr_reflected);
}


void LpcStatsAuxCompute(struct LpcStatsAux *aux) {
  if (aux->A_minus.data[0] == 0)
    LpcStatsAuxComputeAMinus(aux);
  LpcStatsAuxComputeAPlus(aux);
  LpcStatsAuxComputeAAll(aux);
  PrintMatrix64("A^plus", &aux->A_plus);
  PrintMatrix64("A^all", &aux->A);
  LpcStatsAuxComputeA(aux);
  LpcStatsAuxComputeReflectedAutocorr(aux);
}


#ifdef PREDICTION_MATH_TEST
#include <math.h>

/**
   Construct a matrix `mat` with elements
     mat[i,j] = vec[abs(i-j)]
   Requires that `mat` occupy its entire region (so that
   we can freely set its exponent and size).
 */
void ConstructToeplitzMatrix(Vector64 *vec,
                             Matrix64 *mat) {
  int size = mat->num_rows * mat->num_cols;
  assert(size == mat->region->dim &&
         "Matrix must occupy its whole region");
  mat->region->size = vec->region->size;
  mat->region->exponent = vec->region->exponent;
  int64_t *vec_data = vec->data,
      *mat_data = mat->data;
  int dim = vec->dim;
  assert(mat->num_rows == dim && mat->num_cols == dim);
  int vec_stride = vec->stride,
      mat_row_stride = mat->row_stride,
      mat_col_stride = mat->col_stride;
  for (int r = 0; r < dim; r++) {
    for (int c = 0; c < dim; c++) {
      int i = (r > c ? r - c : c - r);
      mat_data[r * mat_row_stride + c * mat_col_stride] = vec_data[i * vec_stride];
    }
  }
}

void test_toeplitz_solve() {
  /* Note: this mirrors code in ../test/linear_prediction.py, in
   * test_toeplitz_solve_compare().  I'm using the same exact
   * numbers in testing, to make it easier to debug.
   */

  int64_t autocorr_array[4] = { 10, 5, 2, 1 },
      y_array[4]  = { 1, 2, 3, 4 },
      x_array[4] = { 0, 0, 0, 0 },
      temp1_array[4] = { 0, 0, 0, 0 },
      temp2_array[4] = { 0, 0, 0, 0 };
  Region64 r1, r2, r3, r4, r5;
  Vector64 autocorr, y, x, temp1, temp2;
  int size_hint = 2;
  InitRegionAndVector64(autocorr_array, 4, 0, size_hint,
                        &r1, &autocorr);
  InitRegionAndVector64(y_array, 4, 0, size_hint,
                        &r2, &y);
  InitRegionAndVector64(x_array, 4, 0, size_hint,
                        &r3, &x);
  InitRegionAndVector64(temp1_array, 4, 0, size_hint,
                        &r4, &temp1);
  InitRegionAndVector64(temp2_array, 4, 0, size_hint,
                        &r5, &temp2);
  ToeplitzSolve(&autocorr, &y, &x, &temp1, &temp2);


  CheckRegion64Size(temp1.region);
  CheckRegion64Size(temp2.region);



  /* Now check that the solution is correct. */
  int64_t mat_array[16];
  for (int i = 0; i < 16; i++) mat_array[i] = 0;
  Region64 mat_region;
  Matrix64 mat;
  InitRegionAndMatrix64(mat_array, 4, 4, 0, 0,
                        &mat_region, &mat);

  CheckRegion64Size(temp1.region);
  /* temp1 := mat * x. */
  PrintVector64("x is ", &x);
  ConstructToeplitzMatrix(&autocorr, &mat);
  PrintMatrix64("mat is ", &mat);
  SetMatrixVector64(&mat, &x, &temp1);
  CheckRegion64Size(temp1.region);

  PrintVector64("temp1", &temp1);
  PrintVector64("y", &y);

  /* temp1 -= y.   Now temp1 is the error/residual. */
  AddIntVector64(-1, &y, &temp1);
  double rel_error = DotVector64AsDouble(&temp1, &temp1) /
      DotVector64AsDouble(&y, &y);
  fprintf(stderr, "Relative error in Toeplitz inversion is %g\n", (float)sqrt(rel_error));
  assert(rel_error < pow(2.0, -20));
}

void test_lpc_stats() {
  /* Test LPC stats accumulation.  Mirrors code in ../test/linear_prediction.py
     These numbers are the same ones used there.
   */
  int16_t signal[10] = { 1,2,3,4,5,7,9,11,13,15 };
  Scalar64 eta;
  InitScalar64FromInt(2, &eta);
  InvertScalar64(&eta, &eta);  /* now it's 0.5. */

  int lpc_order = 4;
  struct LpcStats stats;
  int max_block_size = 5;
  LpcStatsInit(lpc_order, max_block_size, &eta, &stats);
  struct LpcStatsAux aux;
  int est_lpc_order = 4;  /* it can be any number >= 1 and <= 4. */
  LpcStatsAuxInit(&stats, est_lpc_order, &aux);

  LpcStatsAcceptBlock16(5, signal + 0, &stats);
  PrintVector64("Autocorr-coeffs[1]: ", &stats.autocorr);

  LpcStatsAuxCompute(&aux);
  PrintVector64("Autocorr-coeffs-reflected[1]:", &aux.autocorr_reflected);

  LpcStatsAcceptBlock16(5, signal + 5, &stats);
  PrintVector64("Autocorr-coeffs[2]: ", &stats.autocorr);

  LpcStatsAuxCompute(&aux);
  PrintVector64("Autocorr-coeffs-reflected[2]:", &aux.autocorr_reflected);

  PrintMatrix64("A'^-", &aux.A_minus);

  PrintMatrix64("A^+", &aux.A_plus);

  PrintMatrix64("A", &aux.A);


  /* TODO */
}

int main() {
  test_toeplitz_solve();
  test_lpc_stats();
}


#endif  /* #ifdef PREDICTION_MATH_TEST */
