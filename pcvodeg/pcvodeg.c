#include <stdlib.h>
#include <stdio.h>
#include <cvode/cvode.h>                          /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_openmp.h>             /* access to openmp N_Vector              */
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h> /* access to the fixed point SUNNonlinearSolver */
#include <sundials/sundials_types.h>              /* definition of type realtype                  */
#include "pcvodeg.h"
#include <omp.h>

pcv_pars *p_glob;

static int fnpvodeg(realtype t, N_Vector y, N_Vector dydt, void *fdata){
  pcv_pars *p=(pcv_pars*)fdata;
  p->fnpy(t,NV_DATA_OMP(y),NV_DATA_OMP(dydt));
  return 0;
}

void init_solver(int N,int nthreads, double *y, double t0, void (*fnpy)(double,double *,double *), double atol, double rtol, int mxsteps){
  SUNNonlinearSolver NLS;
  int state;
  pcv_pars *p;
  SUNContext sunctx;
  p=malloc(sizeof(pcv_pars));
  p->N=N;
  p->nthreads=nthreads;
  p->y=y;
  p->t0=t0;
  p->fnpy=fnpy;

  /* Create the SUNDIALS context */
  state = SUNContext_Create(NULL, &(sunctx));
  p->uv=N_VMake_OpenMP(N,y,nthreads,sunctx);
  p->solver=CVodeCreate(CV_ADAMS,sunctx);
  state = CVodeSetUserData(p->solver, p);
  state = CVodeSetMaxNumSteps(p->solver, mxsteps);
  state = CVodeInit(p->solver, fnpvodeg,t0,p->uv);
  state = CVodeSStolerances(p->solver, rtol, atol);
  NLS = SUNNonlinSol_FixedPoint(p->uv, 0,sunctx);
  state = CVodeSetNonlinearSolver(p->solver, NLS);
  p_glob=p;
};

void integrate_to(double tnext, double *t, int *state){
  pcv_pars *p=p_glob;
  *state=CVode(p->solver, tnext, p->uv, &(p->t), CV_NORMAL);
  *t=p->t;
}
