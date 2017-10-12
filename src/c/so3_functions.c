#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "fftw3.h"
#include "ssht.h"
#include <omp.h>
#include "so3.h"

void _so3_core_inverse_via_ssht(complex double * f, const complex double * flmn, 
		const so3_parameters_t * parameters)
{
  SO3_ERROR_MEM_ALLOC_CHECK(flmn);
  SO3_ERROR_MEM_ALLOC_CHECK(f);

  so3_core_inverse_via_ssht(f, flmn, parameters);  

}

void _so3_core_inverse_via_ssht_real(double * f_real, const complex double * flmn, 
		const so3_parameters_t * parameters)
{
  SO3_ERROR_MEM_ALLOC_CHECK(flmn);
  SO3_ERROR_MEM_ALLOC_CHECK(f_real);

  so3_core_inverse_via_ssht_real(f_real, flmn, parameters);  

}

void _so3_sampling_ind2elmn(int *el, int *m, int *n, int ind,
				 const so3_parameters_t *parameters)
{
  so3_sampling_ind2elmn(el, m, n, ind, parameters);

}

void _so3_sampling_ind2elmn_real(int *el, int *m, int *n, int ind,
				 const so3_parameters_t *parameters)
{
  so3_sampling_ind2elmn_real(el, m, n, ind, parameters);

}

void _so3_sampling_elmn2ind(int *ind, int el, int m, int n, 
			   const so3_parameters_t *parameters)
{
  so3_sampling_elmn2ind(ind, el, m, n, parameters);
}

void _so3_sampling_elmn2ind_real(int *ind, int el, int m, int n, 
				const so3_parameters_t *parameters)
{
  so3_sampling_elmn2ind_real(ind, el, m, n, parameters);
}

int _so3_sampling_f_size(const so3_parameters_t *parameters)
{
  return so3_sampling_f_size(parameters);
}

int _so3_sampling_n(const so3_parameters_t *parameters)
{
  return so3_sampling_n(parameters);
}

int _so3_sampling_nalpha(const so3_parameters_t *parameters)
{
  return so3_sampling_nalpha(parameters);
}

int _so3_sampling_nbeta(const so3_parameters_t *parameters)
{
  return so3_sampling_nbeta(parameters);
}

int _so3_sampling_ngamma(const so3_parameters_t *parameters)
{
  return so3_sampling_ngamma(parameters);
}

double _so3_sampling_a2alpha(int a, const so3_parameters_t *parameters)
{
  return so3_sampling_a2alpha(a, parameters);
}

double _so3_sampling_b2beta(int b, const so3_parameters_t *parameters)
{
  return so3_sampling_b2beta(b, parameters);
}

double _so3_sampling_g2gamma(int g, const so3_parameters_t *parameters)
{
  return so3_sampling_g2gamma(g, parameters);
}

int _so3_sampling_flmn_size(const so3_parameters_t *parameters)
{
  return so3_sampling_flmn_size(parameters);
}

void _ssht_dl_beta_risbo_full_table(double *dl, double beta, int L,
	           ssht_dl_size_t dl_size, int el, double *sqrt_tbl)
{
  return ssht_dl_beta_risbo_full_table(dl, beta, L, dl_size, el, sqrt_tbl);
} 

void _ssht_core_mw_lb_inverse_sov_sym(complex double *f, const complex double *flm,
				      int L0, int L, int spin,
				      ssht_dl_method_t dl_method,
				      int verbosity)
{
  return ssht_core_mw_lb_inverse_sov_sym(f, flm, L0, L, spin,
					 dl_method, verbosity);
}

void _ssht_core_mw_lb_inverse_sov_sym_real(double *f, const complex double *flm,
				      int L0, int L,
				      ssht_dl_method_t dl_method,
				      int verbosity)
{
  return ssht_core_mw_lb_inverse_sov_sym_real(f, flm, L0, L,
					 dl_method, verbosity);
}

void _ssht_core_mwdirect_inverse(complex double *f, const complex double *flm,
			   int L, int spin, int verbosity)
{
  return ssht_core_mwdirect_inverse(f, flm, L, spin, verbosity);
}

void _ssht_core_mwdirect_inverse_sov(complex double *f, const complex double *flm,
			   int L, int spin, int verbosity)
{
  return ssht_core_mwdirect_inverse_sov(f, flm, L, spin, verbosity);
}
