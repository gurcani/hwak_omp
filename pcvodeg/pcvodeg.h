#include <cvode/cvode.h>                          /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_openmp.h>             /* access to MPI-parallel N_Vector              */
#include <sunnonlinsol/sunnonlinsol_fixedpoint.h> /* access to the fixed point SUNNonlinearSolver */
#include <sundials/sundials_types.h>              /* definition of type realtype                  */

typedef struct pcv_pars{
  double *y;//, *dydt;
  double t0,t;
  int N,nthreads;
  void *solver;
  N_Vector uv;
  void (*fnpy)(double, double *, double *);
}pcv_pars;

void init_solver(int N,int nthreads, double *y, double t0, void (*fnpy)(double,double *,double *), double atol, double rtol, int mxsteps);
void integrate_to(double tnext, double *t, int *state);
