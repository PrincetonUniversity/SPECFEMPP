/*
!=====================================================================
!
!                       S p e c f e m 3 D  G l o b e
!                       ----------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, April 2014
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================
*/

/* b-splines on arbitrary spacing

   parameterization used for Berkeley model SEMUCB-WM1

   CHM 12/97
*/

#include "config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>     // for malloc/free

#define U (unsigned)
#define TOL 1.5f

float *farray1(int, int);
void free_farray1(float *, int);
void stop(char *);

void fill_hh_A3d(float *, float *, int);
float spl_A3d(int, int, float *, float *, float);

/* ----------------------------------------------------------------------------- */

/* wrapper for call from fortran 90 code */

//#ifdef _IBM_
//void fspl(int *ord, int *nknots, double *knot, double *xi, double *rho)
//#else
//void fspl_(int *ord, int *nknots, double *knot, double *xi, double *rho)
//#endif

void
FC_FUNC(fspl,FSPL)(int *ord, int *nknots, float *knot_s, float *knot_hh, double *xi, double *rho)
{
  float xi_s,val;
  int ord_s;

  ord_s = *ord - 1; /* change from fortran to c array convention */
  xi_s = (float) (*xi);

  // allocate temporary float array
  //float *knot_s;
  //knot_s = farray1(0,*nknots - 1);
  //for(int i=0; i < *nknots; i++)
  //  knot_s[i] = (float) knot[i];

  // spline value
  val = spl_A3d(ord_s, *nknots, knot_s, knot_hh, xi_s);

  // return value
  *rho = (double) val;

  // free temporary array
  //free_farray1(knot_s,0);

  return;
}

/* ----------------------------------------------------------------------------- */

float spl_A3d(int ord, int nknots, float *knot, float *hh, float xi)
{
/* ord: number of rho(x)
   nknots : # of knkots = Nx+1 (Nx=index of highest spline)
   xi: point of interest
   splines defined as :
   f_i(x) = a_i(x-x_i)^3 + b_i(x-x_i)^2 + c_i(x-x_i) + d_i
*/
  int Nx;
  float rho_x;
  float coefa,coefb,coefc,coefd;
  float dxi;

  Nx = nknots - 1;

  /* Compute vector hh of spacings
     note: this is done already in the fortran routine.
           the spline node radii won't change and so do their spacings hh
            between different spline evaluation calls.
            thus, there is no need to re-allocate and compute this array for each spline call.
  */
  //float *hh;
  //hh = farray1(0,Nx - 1);
  //fill_hh_A3d(hh,knot,Nx);

  /* Consistency checks */
  if ((xi - TOL) > knot[Nx]) {
    printf("xi=%g / knot[%d]=%g",xi,Nx,knot[Nx]);
    stop("spl: xi>knot[Nx]");
  }
  else if ((xi + TOL) < knot[0]) {
    printf("xi=%g / knot[0]=%g",xi,knot[0]);
    stop("spl: xi<knot[0]");
  }
  else if (ord > Nx)
    stop("spl: order > Nx");

  if (ord == 0) {	/* LHS */
    float hh0 = hh[ord];
    float hhp1 = hh[ord+1];

    float denom = 3.0f * hh0 * hh0 + 3.0f * hh0 * hhp1 + hhp1 * hhp1;

    if (xi >= knot[ord] && xi <= knot[ord+1]) {			/* x0<=x<=x1 */
      coefa = 4.0f / (hh0 * (hh0 + hhp1) * denom);
      coefb = 0.0f;
      coefc = -12.0f / denom;
      coefd = 4.0f * (2.0f * hh0 + hhp1) / denom;

      dxi = xi - knot[ord];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else if (xi > knot[ord+1] && xi <= knot[ord+2]){		/* x1<=x<=x2 */
      coefa = -4.0f / (hhp1 * (hh0 + hhp1) * denom);
      coefb = 12.0f / ((hh0 + hhp1) * denom);
      coefc = -12.0f * hhp1 / ((hh0 + hhp1) * denom);
      coefd = 4.0f * hhp1 * hhp1 / ((hh0 + hhp1) * denom);

      dxi = xi - knot[ord+1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else						/* x>x2 */
      rho_x = 0.0f;
  }

  else if (ord == 1) {	/* LHS+1 */
    float hh0 = hh[ord];
    float hhp1 = hh[ord+1];
    float hhm1 = hh[ord-1];

    float denom = 3.0f * hhm1 * hhm1 + 4.0f * hhm1 * hh0 + hh0 * hh0 + 2.0f * hhm1 * hhp1 + hh0 * hhp1;
    float denomsum = hhm1 + hh0 + hhp1;
    float dd = denomsum * denom;

    if (xi >= knot[ord-1] && xi <= knot[ord]) {			/* x0<=x<=x1 */
      coefa = -4.0f * (3.0f * hhm1 + 2.0f * hh0 + hhp1) /
      	          (hhm1 * (hhm1 + hh0) * dd);
      coefb = 0.0f;
      coefc = 12.0f / denom;
      coefd = 0.0f;

      dxi = xi - knot[ord-1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else if (xi >= knot[ord] && xi <= knot[ord+1]) {			/* x1<=x<=x2 */
      coefa = 4.0f * (2.0f * hhm1 * hhm1 + 6.0f * hhm1 * hh0 + 3.0f * hh0 * hh0 + 3.0f * hhm1 * hhp1 +
              3.0f * hh0 * hhp1 + hhp1 * hhp1) /
                (hh0 * (hhm1 + hh0) * (hh0 + hhp1) * dd);
      coefb = -12.0f * (3.0f * hhm1 + 2.0f * hh0 + hhp1) /
      	       ((hhm1 + hh0) * dd);
      coefc = 12.0f * (-2.0f * hhm1 * hhm1 + hh0 * hh0 + hh0 * hhp1) /
      	       ((hhm1 + hh0) * dd);
      coefd = 4.0f * hhm1 * (4.0f * hhm1 * hh0 + 3.0f * hh0 * hh0 + 2.0f * hhm1 * hhp1 + 3.0f * hh0 * hhp1) /
      	       ((hhm1 + hh0) * dd);

      dxi = xi - knot[ord];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else if (xi >= knot[ord+1] && xi <= knot[ord+2]) {			/* x2<=x<=x3 */
      dd *= (hh0 + hhp1);
      coefa = -4.0f * (2.0f * hhm1 + hh0) / (hhp1 * dd);
      coefb = 12.0f * (2.0f * hhm1 + hh0) / dd;
      coefc = -12.0f * (2.0f * hhm1 + hh0) * hhp1 / dd;
      coefd = 4.0f * (2.0f * hhm1 + hh0) * hhp1 * hhp1 / dd;

      dxi = xi - knot[ord+1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else						/* x>x3 */
      rho_x = 0.0f;
  }

  else if (ord==Nx-1) {		/* RHS-1 */
    float hh0 = hh[ord];
    float hhm1 = hh[ord-1];
    float hhm2 = hh[ord-2];

    float denom = hhm2 * hhm1 + hhm1 * hhm1 + 2.0f * hhm2 * hh0 + 4.0f * hhm1 * hh0 + 3.0f * hh0 * hh0;
    float denomsum = hhm2 + hhm1 + hh0;
    float dd = denomsum * denom;

    if (xi >= knot[ord-2] && xi <= knot[ord-1]) {	/* x0<=x<=x1 */
      coefa = 4.0f * (hhm1 + 2.0f * hh0) / (hhm2 * (hhm2 + hhm1) * dd);
      coefb = coefc = coefd = 0.0f;

      dxi = xi - knot[ord-2];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else if (xi >= knot[ord-1] && xi <= knot[ord]) {	/* x1<=x<=x2 */
      coefa = -4.0f * (hhm2 * hhm2 + 3.0f * hhm2 * hhm1 + 3.0f * hhm1 * hhm1 + 3.0f * hhm2 * hh0 +
               6.0f * hhm1 * hh0 + 2.0f * hh0 * hh0) /
                 (hhm1 * (hhm2 + hhm1) * (hhm1 + hh0) * dd);
      coefb = 12.0f * (hhm1 + 2.0f * hh0) / ((hhm2 + hhm1) * dd);
      coefc = 12.0f * hhm2 * (hhm1 + 2.0f * hh0) / ((hhm2 + hhm1) * dd);
      coefd = 4.0f * hhm2 * hhm2 * (hhm1 + 2.0f * hh0) / ((hhm2 + hhm1) * dd);

      dxi = xi - knot[ord-1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else if (xi >= knot[ord] && xi <= knot[ord+1]) {	/* x2<=x<=x3 */
      dd *= (hhm1 + hh0);
      coefa = 4.0f * (hhm2 + 2.0f * hhm1 + 3.0f * hh0) / (hh0 * dd);
      coefb = -12.0f * (hhm2 + 2.0f * hhm1 + 3.0f * hh0) / dd;
      coefc = 12.0f * (-hhm2 * hhm1 - hhm1 * hhm1 + 2.0f * hh0 * hh0) / dd;
      coefd = 4.0f * hh0 * (3.0f * hhm2 * hhm1 + 3.0f * hhm1 * hhm1 + 2.0f * hhm2 * hh0 + 4.0f * hhm1 * hh0) / dd;

      dxi = xi - knot[ord];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else						/* x>x4 */
      rho_x = 0.0f;
  }

  else if (ord==Nx) {		/* RHS */
    float hhm1 = hh[ord-1];
    float hhm2 = hh[ord-2];

    float denom = (hhm2 + hhm1) * (hhm2 * hhm2 + 3.0f * hhm2 * hhm1 + 3.0f * hhm1 * hhm1);

    if (xi >= knot[ord-2] && xi <= knot[ord-1]) {	/* x0<=x<=x1 */
      coefa = 4.0f / (hhm2 * denom);
      coefb = coefc = coefd = 0.0f;

      dxi = xi - knot[ord-2];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else if (xi >= knot[ord-1] && xi <= knot[ord]) {	/* x1<=x<=x2 */
      coefa = -4.0f / (hhm1 * denom);
      coefb = 12.0f / denom;
      coefc = 12.0f * hhm2 / denom;
      coefd = 4.0f * hhm2 * hhm2 / denom;

      dxi = xi - knot[ord-1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else						/* x>x2 */
      rho_x = 0.0f;
  }

  else {
    /* Away from borders */
    float hh0 = hh[ord];
    float hhp1 = hh[ord+1];
    float hhm1 = hh[ord-1];
    float hhm2 = hh[ord-2];

    float denom1 = hhm2 + hhm1 + hh0 + hhp1;

    if (xi >= knot[ord-2] && xi <= knot[ord-1]) {	/* x0<=x<=x1 */
      coefa = 4.0f / (hhm2 * (hhm2 + hhm1) * (hhm2 + hhm1 + hh0) * denom1);
      coefb = coefc = coefd = 0.0f;

      dxi = xi - knot[ord-2];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }
    else if (xi >= knot[ord-1] && xi <= knot[ord]) {	/* x1<=x<=x2 */
      float denom2 = (hhm2 + hhm1) * (hhm2 + hhm1 + hh0);
      float denom = denom1 * denom2;

      coefa = -4.0f * (hhm2 * hhm2 + 3.0f * hhm2 * hhm1 + 3.0f * hhm1 * hhm1 + 2.0f * hhm2 * hh0 +
               4.0f * hhm1 * hh0 + hh0 * hh0 + hhm2 * hhp1 + 2.0f * hhm1 * hhp1 + hh0 * hhp1) /
                  (hhm1 * (hhm1 + hh0) * (hhm1 + hh0 + hhp1) * denom);
      coefb = 12.0f / denom;
      coefc = 12.0f * hhm2 / denom;
      coefd = 4.0f * hhm2 * hhm2 / denom;

      dxi = xi - knot[ord-1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else if (xi >= knot[ord] && xi <= knot[ord+1]) {	/* x2<=x<=x3 */
      float denom2 = (hhm1 + hh0) * (hhm2 + hhm1 + hh0) * (hhm1 + hh0 + hhp1);
      float denom = denom1 * denom2;

      coefa = 4.0f * (hhm2 * hhm1 + hhm1 * hhm1 + 2.0f * hhm2 * hh0 + 4.0f * hhm1 * hh0 + 3.0f * hh0 * hh0 +
      	      hhm2 * hhp1 + 2.0f * hhm1 * hhp1 + 3.0f * hh0 * hhp1 + hhp1 * hhp1) /
                (hh0 * (hh0 + hhp1) * denom);
      coefb = -12.0f * (hhm2 + 2.0f * hhm1 + 2.0f * hh0 + hhp1) / denom;
      coefc = 12.0f * (-hhm2 * hhm1 - hhm1 * hhm1 + hh0 * hh0 + hh0 * hhp1) / denom;
      coefd = 4.0f * (2.0f * hhm2 * hhm1 * hh0 + 2.0f * hhm1 * hhm1 * hh0 + hhm2 * hh0 * hh0 +
              2.0f * hhm1 * hh0 * hh0 + hhm2 * hhm1 * hhp1 + hhm1 * hhm1 * hhp1 +
              hhm2 * hh0 * hhp1 + 2.0f * hhm1 * hh0 * hhp1) / denom;

      dxi = xi - knot[ord];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else if (xi >= knot[ord+1] && xi <= knot[ord+2]) {	/* x3<=x<=x4 */
      float denom2 = (hh0 + hhp1) * (hhm1 + hh0 + hhp1);
      float denom = denom1 * denom2;

      coefa = -4.0f / (hhp1 * denom);
      coefb = 12.0f / denom;
      coefc = -12.0f * hhp1 / denom;
      coefd = 4.0f * hhp1 * hhp1 / denom;

      dxi = xi - knot[ord+1];
      rho_x = ((coefa * dxi + coefb) * dxi + coefc) * dxi + coefd;
    }

    else						/* x>x4 */
      rho_x = 0.0f;
  }

  // frees temporary array
  //free_farray1(hh,0);

  return (rho_x);
}

/* ----------------------------------------------------------------------------- */

void fill_hh_A3d(float *hh, float *knot, int Nx)
{
  int ii;

  for(ii=0; ii<Nx; ii++)
    hh[ii] = knot[ii+1] - knot[ii];
}

/* ----------------------------------------------------------------------------- */

float *farray1(int n11, int n12)
{
  float *m;
  m = (float *) malloc( U (n12-n11+1) * sizeof(float) );
  if(!m) {
    stop("allocation error in farray1");
  }
  return (m-n11);
}

/* ----------------------------------------------------------------------------- */

void free_farray1(float *a, int n11)
{
  free(&a[n11]);
}

/* ----------------------------------------------------------------------------- */

void stop(char *message){
  printf("\n\a <error> %s\n",message);
  exit(-1);
}
