/* -*-mode:c; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// original Burak & Fiete indentation
// entorhinal5.c
// Yoram Burak and Ila Fiete
// Christoph Schmidt-Hieber

#include <complex.h>
#ifdef CUDA
#include <cufftw.h>
#include <helper_cuda.h>
#else
#include "fftw/fftw3.h"
#endif //CUDA
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#ifdef PROFILER
#include <gperftools/profiler.h>
#endif

// Precision Settings for the fftw3 library
// you must have installed single precision 
// for this code to work.
// For installation instructions, view the associated ReadMe.
// #define DOUBLE_PRECISION 1
#ifndef DOUBLE_PRECISION

typedef float real;
#define REAL "%f"
#define REALREAL "%f %f"
#define fftwp_complex fftwf_complex
#define fftwp_plan fftwf_plan
#define fftwp_plan_dft_r2c_2d fftwf_plan_dft_r2c_2d
#define fftwp_plan_dft_c2r_2d fftwf_plan_dft_c2r_2d
#define fftwp_destroy_plan fftwf_destroy_plan
#define fftwp_execute fftwf_execute
#endif

//=========================================================================
// Parameters and Default Value Declarations
//=========================================================================

#define PI 3.141592654
#define UNDEF -999

// Default Network Parameters
#define MAXN 128
#define MAXNP (MAXN+MAXN/2)
#define MAXNLR 100000
int n = 128;                // number of neurons per side
real bvalue = 1;            // value of b
real clip = UNDEF;			// clipping in nonlinear transfer function
real tau = 10;               // neuron time-constant in ms
real tau_e = 2.5;
real tau_i = 2.5;
real noise_amp = 0;
int gaussian = 0;
int bipolar = 0;
int constant = 0;
int ornstein = 0;
real rarg_min = 1e9;
real rarg_max = -1e9;

// network: weights

int filtn = UNDEF;

// network: weights 
real wamp = 5;
real abar = 1;
real alphabar = 1.02;
real blobspacing = 13;
real beta;
int notau = 0;
int nonlinear = 0;
int nonlinear_cont = 0;
int nonlinear_step = 0;
int nonlinear_equiv = 0;
real nl_start = 0.0;
real nl_cutoff = 1.0;
real nl_ceiling = 3.0;
real nl_factor = 1.5;

real nl_cont_amp1 = 0.9;
real nl_cont_xshift1 = -0.3;
real nl_cont_slope1 = 167.0;
real nl_cont_amp2 = 1.4;
real nl_cont_xshift2 = 2.5;
real nl_cont_slope2 = 0.13;

int nmda = 0;
int nmda_nl = 0;
int nmda_ampa = 0;
int nmda_mul = 0;
int nmda_add = 0;

real tau_nmda = 100.0;
real k_nmda = 0.1;

real center_nmda = 0.0;
real slope_nmda = 0.1;

real tau_ampa = 10.0;

int fi = 0;
real fi_amp = 0.0;
real fi_freq = 0.001;

real mercedes = 0;

int wtphase = 2;
real v_gain = .0825;

int scale_weights = 0;   // scale weights together with the input scaling

// feedforward input

// gaussian falloff
real falloff = 0.;
int falloff_band = UNDEF;  // in neurons

// square falloff
int sq_falloff = 0;

// network: long-range connections
int longrange = 0;
int Nlr = 1000;
int naddlr = 50;
real lramp = 5;
int ndrawlr = 10000;
real minlr = 1;
int refresh_lr = 0;
int normalise_lr = 0;
real thres_lr = 0.1;

// Enclosure and Trajectory settings
#define MAXD 200
int d = 180;                // size of enclusure
real initial_theta = UNDEF; // UNDEF designates random value.
real init_v = 0.16/2;       // 0.16: (d/1000) max v, 1m/s

int initxpos = 60 ;         // set to d/2
int initypos = 145;
int Nminstay = 15;          // min time until turn; 10 corresponds to
                            // 5 ms = membrane time constant
							
int Nmaxstay = 600;         // max time until next turn
int smooth = 1;             // smooth turns flag (1 = smooth)
int det = 0;                // deterministic circle trajctory

int vstep = 0;
int vstep_clear = 0;
int vstep_nsteps;
real vstep_vstep;

real min_rate = 0.01;

real climit = UNDEF;
real slimit = UNDEF;

int external_trajectory = 1;
char trajectory_filename[1024] = "trajectory";
FILE *vfile;

// Velocity gradient

real dvx = UNDEF;
real dvy = UNDEF;

// smooth trajectory parameters

real accel_fact = 0.5;      // determines acceleration in units of v/tau

// Simulation Parameters

int periodic = 0;
real dt = 0.5;              // in ms
int niter = 100000;
//print after every niterprint iterations
int niterprint = 1000;
int seed = UNDEF;
int nflow = 300;
real flow_theta = UNDEF;
real init_noise = 1e-3;

// input and output

char rfile[1024] = "r";      // set s to "" for random initial condition
char popfile[1024] = "pop";
char snfile[1024] = "sn";
char trackfile[1024] = "track";
char il_trackfile[1024] = "il_track";
char lrfile[1024] = "lr";
int ndumplr = 1000;
int ndump_population = -1;  // -1: print only at the end
int ndump_single = -1;      // -1: print only at the end
int nclear_single = -1;
int special_dumps = 0;
int out_decomp = 0;

int njump_limit = UNDEF;
int njump_steps = 20001;

// tracking

// parameters of version1
int initxcm = 97;
int initycm = 127;
int ntrack = 17;            // make sure that only one blob goes into a patch of this size

// inverse lattice tracking

int il_track = 0;
int il_initial_search_radius = 16; // ~1.5 times expected radius.
int il_search_radius = 5;
int il_cm_radius = 8;
real il_cm_thres = 0;
int il_max_radius = 20;
int il_reset = 2000;
real il_min_value = 10;

double il_cm[3][2];
int il_track_center[3][2];

real fftmatrix[MAXNP][MAXNP];

// spiking simulation

int spiking = 0;
int spikeout = 0;
char spikename[1024] = "spike";
int spikesubcounter[MAXN][MAXN];

// single neuron recording

#define MAXNRECORD 25
int nrecord = 10;           // number of neurons to record
real sn_radius = 80;




// FFT Plans
fftwp_complex *fto, *ftr, *ftl, *ftd, *ftu;
fftwp_plan prf,plf,pdf,puf;
fftwp_plan prr,plr,pdr,pur;
real *inr, *inl, *ind, *inu;
real *outr, *outl, *outd, *outu;
fftwp_complex *conv_temp;
fftwp_complex *fft_storage;
#if 0
Complex *d_fto, *d_ftr, *d_ftl, *d_ftd, *d_ftu;
real *d_inr, *d_inl, *d_ind, *d_inu;
real *d_outr, *d_outl, *d_outd, *d_outu;
Complex *d_conv_temp;
#endif
int nlr;
int wlr[MAXNLR][4];

int *neuron_type;
int np;                 // size of fft arrays with padding
int npad;                  // size of padding on each side

typedef struct smooth_params_struct {
  real accel;
  int accel_sign;
  int now_turning;
  int steps_in_turn;
  real time_to_stop;
  int steps_to_stop;
} smooth_params;


//=========================================================================
// END Parameters and Default Value Declarations
//=========================================================================

void get_parameters(int argc, char *argv[])
{
    // char s[1024];
  int narg = 1;
  while (narg < argc) {
    if (!strcmp(argv[narg],"-h")) {
      printf("%s -h: Print this help message.\n", argv[0]);
      printf("  Simulation:   -dt -niter -seed -nprint\n");
      printf("  Network:      -n -tau -clip -noise -notau -nonlinear -nonlinear_cont -nonlinear_step -nonlinear_equiv -nl_start -nl_cutoff -nl_ceiling -nl_factor -mercedes\n");
      printf("  Network:      -nl_cont_amp1 -nl_cont_xshift1 -nl_cont_slope1 -nl_cont_amp2 -nl_cont_xshift2 -nl_cont_slope2\n");
      printf("  Network:      -nmda -nmda_nl -nmda_mul -nmda_add -nmda_ampa -tau_nmda -k_nmda -center_nmda -slope_nmda\n");
      printf("  Network:      -fi -fi_amp -fi_freq\n");
      printf("  Spiking:      -spike -spikeout -spikename\n");
      printf("  Weights:      -wamp -abar -alphabar -bs  -wtphase -vgain -bvalue\n");
      printf("  Boundary:     -climit -slimit -scaleweights -falloff -sqfall -falloff_band -periodic\n");
      printf("  Initialize:   -rfile -randomr -initnoise -nflow flow_theta\n");
      printf("  Long-range:   -lr -lramp -ndrawlr -minlr -Nlr naddlr -normaliselr -refreshlr -threslr -lrfil -ndumplre\n");
      printf("  Trajectory:   -d -det -smooth -sharp -af -Nminstay -Nmaxstay -initth -initv -initpos\n");
      printf("                -trj_file -external -vstep -vstepclear\n");
      printf("  Vel. Grad:    -gradv <dvx dvy>\n");
      printf("  S-N rec:      -nrec snradius -ndumps -nclears\n");
      printf("  Pop dump:     -ndumpp -popfile\n");
      printf("  Abort:        -nj_limit -nj_steps\n");
      printf("  Track:        -ntrack -initcm -trackfile\n");
      printf("                -minrate\n");
      printf("  FFT track:    -il_track -il_rinit -il_rsearch -il_rcm -il_cm_thres -il_rmax -il_min_val -il_reset -il_trackfile\n");
      printf("  Special:      -specialdumps -outdecomp\n");
      exit(0);
    } else if (!strcmp(argv[narg],"-n")) {
      sscanf(argv[narg+1],"%d",&n);
      narg += 2;
    } else if (!strcmp(argv[narg],"-d")) {
      sscanf(argv[narg+1],"%d",&d);
      narg += 2;
    } else if (!strcmp(argv[narg],"-clip")) {
      sscanf(argv[narg+1],REAL,&clip);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tau")) {
      sscanf(argv[narg+1],REAL,&tau);
      narg += 2;
    } else if (!strcmp(argv[narg],"-notau")) {
      notau = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nonlinear")) {
      nonlinear = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nonlinear_cont")) {
      nonlinear_cont = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nonlinear_step")) {
      nonlinear_step = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nonlinear_equiv")) {
      nonlinear_equiv = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nmda")) {
      nmda = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nmda_nl")) {
      nmda_nl = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nmda_mul")) {
      nmda_mul = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nmda_add")) {
      nmda_add = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nmda_ampa")) {
      nmda_ampa = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-tau_nmda")) {
      sscanf(argv[narg+1],REAL,&tau_nmda);
      narg += 2;
    } else if (!strcmp(argv[narg],"-k_nmda")) {
      sscanf(argv[narg+1],REAL,&k_nmda);
      narg += 2;
    } else if (!strcmp(argv[narg],"-center_nmda")) {
      sscanf(argv[narg+1],REAL,&center_nmda);
      narg += 2;
    } else if (!strcmp(argv[narg],"-slope_nmda")) {
      sscanf(argv[narg+1],REAL,&slope_nmda);
      narg += 2;
    } else if (!strcmp(argv[narg],"-fi")) {
      fi = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-fi_amp")) {
      sscanf(argv[narg+1],REAL,&fi_amp);
      narg += 2;
    } else if (!strcmp(argv[narg],"-fi_freq")) {
      sscanf(argv[narg+1],REAL,&fi_freq);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_start")) {
      sscanf(argv[narg+1],REAL,&nl_start);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cutoff")) {
      sscanf(argv[narg+1],REAL,&nl_cutoff);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_ceiling")) {
      sscanf(argv[narg+1],REAL,&nl_ceiling);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_factor")) {
      sscanf(argv[narg+1],REAL,&nl_factor);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cont_amp1")) {
      sscanf(argv[narg+1],REAL,&nl_cont_amp1);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cont_xshift1")) {
      sscanf(argv[narg+1],REAL,&nl_cont_xshift1);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cont_slope1")) {
      sscanf(argv[narg+1],REAL,&nl_cont_slope1);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cont_amp2")) {
      sscanf(argv[narg+1],REAL,&nl_cont_amp2);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cont_xshift2")) {
      sscanf(argv[narg+1],REAL,&nl_cont_xshift2);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nl_cont_slope2")) {
      sscanf(argv[narg+1],REAL,&nl_cont_slope2);
      narg += 2;
    } else if (!strcmp(argv[narg],"-mercedes")) {
      sscanf(argv[narg+1],REAL,&mercedes);
      narg += 2;
    } else if (!strcmp(argv[narg],"-noise")) {
      sscanf(argv[narg+1],REAL,&noise_amp);
      narg += 2;
    } else if (!strcmp(argv[narg],"-gaussian")) {
      gaussian = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-bipolar")) {
      bipolar = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-constant")) {
      constant = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-ornstein")) {
      ornstein = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-spike")) {
      spiking = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-spikeout")) {
      spikeout = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-spikename")) {
      sscanf(argv[narg+1],"%s",spikename);
      narg+=2;
    } else if (!strcmp(argv[narg],"-wamp")) {
      sscanf(argv[narg+1],REAL,&wamp);
      narg += 2;
    } else if (!strcmp(argv[narg],"-abar")) {
      sscanf(argv[narg+1],REAL,&abar);
      narg += 2;
    } else if (!strcmp(argv[narg],"-climit")) {
      sscanf(argv[narg+1],REAL,&climit);
      narg += 2;
    } else if (!strcmp(argv[narg],"-slimit")) {
      sscanf(argv[narg+1],REAL,&slimit);
      narg += 2;
    } else if (!strcmp(argv[narg],"-alphabar")) {
      sscanf(argv[narg+1],REAL,&alphabar);
      narg += 2;
    } else if (!strcmp(argv[narg],"-bs")) {
      sscanf(argv[narg+1],REAL,&blobspacing);
      // wtphase = blobspacing/5;
      narg += 2; 
    } else if (!strcmp(argv[narg],"-lramp")) {
      sscanf(argv[narg+1],REAL,&lramp);
      narg += 2; 
    } else if (!strcmp(argv[narg],"-lr")) {
      longrange = 1;
      narg ++;
    } else if (!strcmp(argv[narg],"-ndrawlr")) {
      sscanf(argv[narg+1],"%d",&ndrawlr);
      narg += 2;
    } else if (!strcmp(argv[narg],"-Nlr")) {
      sscanf(argv[narg+1],"%d",&Nlr);
      narg += 2;  
    } else if (!strcmp(argv[narg],"-naddlr")) {
      sscanf(argv[narg+1],"%d",&naddlr);
      narg += 2; 
    } else if (!strcmp(argv[narg],"-lrfile")) {
      sscanf(argv[narg+1],"%s",lrfile);
      narg += 2; 
    } else if (!strcmp(argv[narg],"-minlr")) {
      sscanf(argv[narg+1],REAL,&minlr);
      narg +=2 ;
    } else if (!strcmp(argv[narg],"-threslr")) {
      sscanf(argv[narg+1],REAL,&thres_lr);
      narg +=2 ;
    } else if (!strcmp(argv[narg],"-refreshlr")) {
      refresh_lr = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-normaliselr")) {
      normalise_lr = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-wtphase")) {
      sscanf(argv[narg+1],"%d",&wtphase);
      narg += 2;
    } else if (!strcmp(argv[narg],"-vgain")) {
      sscanf(argv[narg+1],REAL,&v_gain);
      narg += 2;
    } else if (!strcmp(argv[narg],"-bvalue")) {
      sscanf(argv[narg+1],REAL,&bvalue);
      narg += 2;
    } else if (!strcmp(argv[narg],"-scaleweights")) {
      scale_weights = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-falloff")) {
      sscanf(argv[narg+1],REAL,&falloff);
      narg +=2 ;
    } else if (!strcmp(argv[narg],"-falloff_band")) {
      sscanf(argv[narg+1],"%d",&falloff_band);
      narg +=2 ;
    } else if (!strcmp(argv[narg],"-d")) {
      sscanf(argv[narg+1],"%d",&d);
      narg += 2;
    } else if (!strcmp(argv[narg],"-det")) {
      det = 1;
      narg ++;
    } else if (!strcmp(argv[narg],"-initth")) {
      sscanf(argv[narg+1],REAL,&initial_theta);
      narg += 2;
    } else if (!strcmp(argv[narg],"-initv")) {
      sscanf(argv[narg+1],REAL,&init_v);
      narg += 2;
    } else if (!strcmp(argv[narg],"-initpos")) {
      sscanf(argv[narg+1],"%d",&initxpos);
      sscanf(argv[narg+2],"%d",&initypos);
      narg += 3;
    } else if (!strcmp(argv[narg],"-dt")) {
      sscanf(argv[narg+1],REAL,&dt);
      narg += 2;
    } else if (!strcmp(argv[narg],"-niter")) {
      sscanf(argv[narg+1],"%d",&niter);
      narg += 2;
    } else if (!strcmp(argv[narg],"-Nminstay")) {
      sscanf(argv[narg+1],"%d",&Nminstay);
      narg +=2;
    } else if (!strcmp(argv[narg],"-Nmaxstay")) {
      sscanf(argv[narg+1],"%d",&Nmaxstay);
      narg +=2;
    } else if (!strcmp(argv[narg],"-nrec")) {
      sscanf(argv[narg+1],"%d",&nrecord);
      narg += 2;
    } else if (!strcmp(argv[narg],"-initcm")) {
      sscanf(argv[narg+1],"%d",&initxcm);
      sscanf(argv[narg+2],"%d",&initycm);
      narg += 3;
    } else if (!strcmp(argv[narg],"-ntrack")) {
      sscanf(argv[narg+1],"%d",&ntrack);
      narg += 2;
    } else if (!strcmp(argv[narg],"-ndumpp")) {
      sscanf(argv[narg+1],"%d",&ndump_population);
      narg += 2;
    } else if (!strcmp(argv[narg],"-ndumps")) {
      sscanf(argv[narg+1],"%d",&ndump_single);
      narg += 2;
    } else if (!strcmp(argv[narg],"-specialdumps")) {
      special_dumps = 1;
      narg ++;
    } else if (!strcmp(argv[narg],"-nj_limit")) {
      sscanf(argv[narg+1],"%d",&njump_limit);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nj_steps")) {
      sscanf(argv[narg+1],"%d",&njump_steps);
      narg += 2;
    } else if (!strcmp(argv[narg],"-outdecomp")) {
      out_decomp = 1;
      narg ++;
    } else if (!strcmp(argv[narg],"-snradius")) {
      sscanf(argv[narg+1],REAL,&sn_radius);
      narg += 2;
    } else if (!strcmp(argv[narg],"-nclears")) {
      sscanf(argv[narg+1],"%d",&nclear_single);
      narg += 2;
    } else if (!strcmp(argv[narg],"-ndumplr")) {
      sscanf(argv[narg+1],"%d",&ndumplr);
      narg += 2;
    } else if (!strcmp(argv[narg],"-rfile")) {
      sscanf(argv[narg+1],"%s",rfile);
      narg += 2;
    } else if (!strcmp(argv[narg],"-randomr")) {
      strcpy(rfile,"");
      narg++;
    } else if (!strcmp(argv[narg],"-initnoise")) {
      sscanf(argv[narg+1],REAL,&init_noise);
      narg += 2;
    } else if (!strcmp(argv[narg],"-snfile")) {
      sscanf(argv[narg+1],"%s",snfile);
      narg += 2;
    } else if (!strcmp(argv[narg],"-trackfile")) {
      sscanf(argv[narg+1],"%s",trackfile);
      narg += 2;
    } else if (!strcmp(argv[narg],"-popfile")) {
      sscanf(argv[narg+1],"%s",popfile);
      narg += 2;
    } else if (!strcmp(argv[narg],"-smooth")) {
      smooth = 1;
      narg ++;
    } else if (!strcmp(argv[narg],"-sharp")) {
      smooth = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-af")) {
      sscanf(argv[narg+1],REAL, &accel_fact);
      narg += 2;
    } else if (!strcmp(argv[narg],"-seed")) {
      sscanf(argv[narg+1],"%d",&seed);
      narg += 2;
    } else if (!strcmp(argv[narg],"-periodic")) {
      periodic = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-nflow")) {
      sscanf(argv[narg+1],"%d",&nflow);
      narg += 2;
    } else if (!strcmp(argv[narg],"-flow_theta")) {
      sscanf(argv[narg+1],REAL,&flow_theta);
      narg+=2;
    } else if (!strcmp(argv[narg],"-vstep")) {
      sscanf(argv[narg+1],"%d",&vstep_nsteps);
      sscanf(argv[narg+2],REAL,&vstep_vstep);
      vstep = 1;
      narg += 3;
    } else if (!strcmp(argv[narg],"-vstepclear")) {
      vstep_clear = 1;
      narg++;
    } else if (!strcmp(argv[narg], "-gradv")) {
      sscanf(argv[narg+1],REAL,&dvx);
      sscanf(argv[narg+2],REAL,&dvy);
      narg += 3;
    } else if (!strcmp(argv[narg],"-sqfall")) {
      sq_falloff = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-minrate")) {
      sscanf(argv[narg+1],REAL,&min_rate);
      narg+=2;
    } else if (!strcmp(argv[narg],"-trj_file")) {
      sscanf(argv[narg+1],"%s",trajectory_filename);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_track")) {
      il_track = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-il_reset")) {
      sscanf(argv[narg+1],"%d",&il_reset);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_rinit")) {
      sscanf(argv[narg+1],"%d",&il_initial_search_radius);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_rsearch")) {
      sscanf(argv[narg+1],"%d",&il_search_radius);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_rcm")) {
      sscanf(argv[narg+1],"%d",&il_cm_radius);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_cm_thres")) {
      sscanf(argv[narg+1], REAL ,&il_cm_thres);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_rmax")) {
      sscanf(argv[narg+1],"%d", &il_max_radius);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_min_val")) {
      sscanf(argv[narg+1],REAL,&il_min_value);
      narg+=2;
    } else if (!strcmp(argv[narg],"-il_trackfile")) {
      sscanf(argv[narg+1],"%s",il_trackfile);
      narg += 2;
    } else if (!strcmp(argv[narg],"-external")) {
      external_trajectory = 1;
      smooth = 0;
      if (NULL == (vfile = fopen(trajectory_filename,"r"))) {
	printf("error opening trajectory file %s\n", trajectory_filename);
	fflush(stdout);
	exit(0);
      }
      printf("trajectory file opened successfully\n");
      narg++;
    } else {
      printf("unkown option: %s\n", argv[narg]);
      exit(0);
    }
  }  
}


void print_parameters()
{
  printf("n               = %d\n", n);
  printf("d               = %d\n", d);
  printf("tau             = %f\n", tau);
  printf("noise_amp       = %f\n", noise_amp);
  printf("gaussian        = %d\n", gaussian);
  printf("bipolar         = %d\n", bipolar);
  printf("constant        = %d\n", constant);
  printf("ornstein        = %d\n", ornstein);
  printf("notau           = %d\n", notau);
  printf("nonlinear       = %d\n", nonlinear);
  printf("nonlinear_cont  = %d\n", nonlinear_cont);
  printf("nonlinear_step  = %d\n", nonlinear_step);
  printf("nonlinear_equiv = %d\n", nonlinear_equiv);
  printf("nl_start        = %f\n", nl_start);
  printf("nl_cutoff       = %f\n", nl_cutoff);
  printf("nl_ceiling      = %f\n", nl_ceiling);
  printf("nl_factor       = %f\n", nl_factor);
  printf("mercedes        = %f\n", mercedes);
  printf("periodic        = %d\n", periodic);
  printf("nflow           = %d\n", nflow);
  printf("blobspacing     = %f\n", blobspacing);
  printf("abar            = %f\n", abar);
  printf("alphabar        = %f\n", alphabar);
  printf("beta            = %f\n", beta);
  printf("vgain           = %f\n", v_gain);
  printf("d               = %d\n", d);
  printf("flow_theta      = %f\n", flow_theta);
  printf("initial_theta   = %f\n", initial_theta);
  printf("initial v       = %f\n", init_v);
  printf("initial x pos   = %d\n", initxpos);
  printf("initial y pos   = %d\n", initypos);
  printf("dvx             = %f\n", dvx);
  printf("dvy             = %f\n", dvy);    
  printf("Nminstay        = %d\n", Nminstay);
  printf("Nmaxstay        = %d\n", Nmaxstay);
  printf("dt              = %f\n", dt);
  printf("spiking         = %d\n", spiking);
  printf("niter           = %d\n", niter);
  printf("nrecord         = %d\n", nrecord);
  printf("init xcm        = %d\n", initxcm);
  printf("init ycm        = %d\n", initycm);
  printf("ntrack          = %d\n", ntrack);
  printf("minrate         = %le\n", min_rate);
  printf("rfile           = %s\n", rfile);
  printf("snfile          = %s\n", snfile);
  printf("trackfile       = %s\n", trackfile);
  printf("smooth          = %d\n", smooth);
  printf("accel_fact      = %f\n", accel_fact);
  printf("clip            = %f\n", clip);
  printf("seed            = %d\n", seed);
  printf("det             = %d\n", det);
  printf("scale weights   = %d\n", scale_weights);
  printf("falloff         = %f\n", falloff);
  printf("falloff band    = %d\n", falloff_band);
  printf("longrange       = %d\n", longrange);
  printf("nclear_single   = %d\n", nclear_single);
  printf("climit          = %f\n", climit);
  printf("slimit          = %f\n", slimit);
  printf("sqfall          = %d\n", sq_falloff);
  printf("il_rinit        = %d\n", il_initial_search_radius);
  printf("il_rsearch      = %d\n", il_search_radius);
  printf("il_cm_radius    = %d\n", il_cm_radius);
  printf("il_cm_thres     = %f\n", il_cm_thres);
  printf("il_max_radius   = %d\n", il_max_radius);
  printf("il_trackfile    = %s\n", il_trackfile);
  printf("init_noise      = %f\n", init_noise);
  if (longrange == 1) {
    printf("Nlr             = %d\n", Nlr);
    printf("naddlr          = %d\n", naddlr);
    printf("lramp           = %f\n", lramp);
    printf("ndrawlr         = %d\n", ndrawlr);
    printf("minlr           = %f\n", minlr); 
    printf("normalise_lr    = %d\n", normalise_lr);
    printf("refresh_lr      = %d\n", refresh_lr);
    printf("thres_lr        = %f\n", thres_lr);
    printf("lrfile          = %s\n", lrfile);
  }
  fflush(stdout);
}

double gaussrand() {
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if(phase == 0) {
        do {
            double U1 = drand48();
            double U2 = drand48();

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    return X;
}

double orn_uhl_A(double D, double tau, double dt) {
    return sqrt((D*tau/2.0) * (1-exp(-2.0*dt/tau)));
}

double orn_uhl_rand(double s0, double st, double dt, double tau, double A) {
    return s0 + (st-s0)*exp(-dt/tau) + A*gaussrand();
}

real fsigm(real x, real center, real slope) {
  return 1.0 - 1.0/(1.0+exp((x-center)/slope));
}

real nonlinf_cont(real x) {
    return nl_cont_amp1 * (1.0 - 1.0/(1.0+exp(nl_cont_slope1*x+nl_cont_xshift1))) *
        (1.0 - nl_cont_amp2 * exp(nl_cont_slope2*x+nl_cont_xshift2));
}


real nonlinf(real x) {
    // return x * (1-fsigm(x, 0.5, 0.5)) + 0.3*fsigm(x, 0.18, 0.04);
    // return x*nl_factor > nl_cutoff ? nl_cutoff : x*nl_factor;
    /*
    y[x<start] = 0.0
    y[(x>=start) & (x<cutoff)] = x[(x>=start) & (x<cutoff)]-start
    y[x>=cutoff] = factor * x[x>=cutoff] + cutoff - start - factor*cutoff
    y[y>=ceiling] = ceiling
    */
    real val = 0.0;
    if (x >= nl_cutoff) {
        val = nl_factor*x + nl_cutoff - nl_start - nl_factor*nl_cutoff;
    } else if (x >= nl_start) {
        val = x-nl_start;
    }
    return val > nl_ceiling? nl_ceiling : val;
}

real nonlinf_step(real x) {
    if (x > nl_start) {
        real val = nl_factor*x + nl_cutoff - nl_factor*nl_start;
        return val > nl_ceiling? nl_ceiling : val;
    } else {
        return 0;
    }
}

real nonlinf_equiv(real x) {
    real A = nl_factor;
    real offset = nl_start;
    real step = nl_cutoff;
    real sigma = nl_ceiling;

    return
        (A*exp(-((offset-x)*(offset-x)/(2.0*sigma*sigma)))*sigma)/sqrt(2.0*PI) +
        0.5*A*(-offset+step+x)*erfc((offset-x)/(sqrt(2)*sigma));
}

static inline real nonlinf_static(real x) {
    real val =
        x >= nl_start ?
        (x >= nl_cutoff ?
         nl_factor*x + nl_cutoff - nl_start - nl_factor*nl_cutoff : x-nl_start) : 0;
    
    return val > nl_ceiling? nl_ceiling : val;
}

// Used in the construction of the center surround weight matrix
// Returns the weight of neuron (x,y) 
real filter_value(real x, real y)
{
  real ret = 0;

  ret = wamp*abar*exp(-alphabar*beta*(x*x+y*y));
  ret -= wamp*exp(-beta*(x*x+y*y));

  if (mercedes != 0) {
    real theta = atan2(y,x);
    ret *= 1 + mercedes*cos(3*theta);
  }

  return ret;
}

// Setup weight matricies and creates the execution plans for
// the fourier transforms
// All filters and arrays are declared as global variables
void setup_weights()
{


  int i,j;
  fftwp_plan p;
  real x[2*MAXN];
  
  beta = 3./(blobspacing*blobspacing);

  if (periodic == 1)
  {
    npad = 0;
    np = n;  
  } 
  else   // non-periodic boundary conditions
  {    
    if (filtn == UNDEF)
      npad = n/2;
    else
      npad = filtn;
    if (npad % 2 == 1)
      npad++;
    np = n+npad;
  }
	
	printf("n = %d, npad = %d, np = %d, v_gain = %f\n", n, npad, np, v_gain);
	fflush(stdout);
 
        real* filt = (real *)malloc(np*np*sizeof(real));

  for (i = 0; i < np/2; i++)
    x[i] = i;
  for (; i < np; i++)
    x[i] = i-np;

// Create the center surround weight matrix

  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      if (periodic)
	filt[np*i+j] = filter_value(x[i],x[j]);
      else if ((abs(x[i]) < npad-wtphase)&&(abs(x[j]) < npad-wtphase))
	filt[np*i+j] = filter_value(x[i],x[j]);
      else
	filt[np*i+j] = 0;

  real *frshift, *flshift, *fdshift, *fushift;
  frshift = (real *)malloc(np*np*sizeof(real));
  flshift = (real *)malloc(np*np*sizeof(real));
  fdshift = (real *)malloc(np*np*sizeof(real));
  fushift = (real *)malloc(np*np*sizeof(real));

  for (i = 0; i < np; i++){
    for (j = 0; j < np; j++) {

      frshift[np*i+j] = filt[np*i+((j-wtphase+np)%np)];
      flshift[np*i+j] = filt[np*i+((j+wtphase+np)%np)];
      fdshift[np*i+j] = filt[np*((i-wtphase+np)%np)+j];
      fushift[np*i+j] = filt[np*((i+wtphase+np)%np)+j];
    }

	}

  // fourier transforms to center a weight matrix on every neuron
 
  fto = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));
  ftr = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));
  ftl = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));
  ftu = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));
  ftd = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));

  p = fftwp_plan_dft_r2c_2d(np,np,filt,fto,FFTW_MEASURE);
  fftwp_execute(p);
  fftwp_destroy_plan(p);
  p = fftwp_plan_dft_r2c_2d(np,np,frshift,ftr,FFTW_MEASURE);
  fftwp_execute(p);
  fftwp_destroy_plan(p);
  p = fftwp_plan_dft_r2c_2d(np,np,flshift,ftl,FFTW_MEASURE);
  fftwp_execute(p);
  fftwp_destroy_plan(p);
  p = fftwp_plan_dft_r2c_2d(np,np,fdshift,ftd,FFTW_MEASURE);
  fftwp_execute(p);
  fftwp_destroy_plan(p);
  p = fftwp_plan_dft_r2c_2d(np,np,fushift,ftu,FFTW_MEASURE);
  fftwp_execute(p);
  fftwp_destroy_plan(p);




//========================================================
//	SET GLOBAL VARIABLES
//========================================================

// prepair the fft templates for all neuron types
  
  conv_temp = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));
  fft_storage = (fftwp_complex *)malloc(sizeof(fftwp_complex)*np*(np/2+1));
 
// Allocate space for input arrays
  inr = (real *)malloc(np*np*sizeof(real));
  inl = (real *)malloc(np*np*sizeof(real));
  ind = (real *)malloc(np*np*sizeof(real));
  inu = (real *)malloc(np*np*sizeof(real));

  // set to zero all elements (including the padding elements)
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) {
      int index = i*np+j;
      inr[index] = inl[index] = ind[index] = inu[index] = 0;
    }

// Allocate space for output arrays
  outr = (real *)malloc(np*np*sizeof(real));
  outl = (real *)malloc(np*np*sizeof(real));
  outd = (real *)malloc(np*np*sizeof(real));
  outu = (real *)malloc(np*np*sizeof(real));
  

// Forward Plans
  prf = fftwp_plan_dft_r2c_2d(np, np, inr, conv_temp, FFTW_MEASURE);
  plf = fftwp_plan_dft_r2c_2d(np, np, inl, conv_temp, FFTW_MEASURE);
  pdf = fftwp_plan_dft_r2c_2d(np, np, ind, conv_temp, FFTW_MEASURE);
  puf = fftwp_plan_dft_r2c_2d(np, np, inu, conv_temp, FFTW_MEASURE);

// Reverse Plans
  prr = fftwp_plan_dft_c2r_2d(np, np, conv_temp, outr, FFTW_MEASURE);
  plr = fftwp_plan_dft_c2r_2d(np, np, conv_temp, outl, FFTW_MEASURE);
  pdr = fftwp_plan_dft_c2r_2d(np, np, conv_temp, outd, FFTW_MEASURE);
  pur = fftwp_plan_dft_c2r_2d(np, np, conv_temp, outu, FFTW_MEASURE);


  
// Cleanup
  free(filt);
  free(frshift);
  free(flshift);
  free(fdshift);
  free(fushift);

}

//  Creates a block matrix wwith uniformly distributed blocks representing
//  each head direction type (Up, Down, Left, Right)

void setup_type_distribution(char neuron_type[MAXN][MAXN])
{

  // 2x2 pattern, no zero type neurons
  int i,j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      switch (2*(i%2)+(j%2)) {
      case 0:
	neuron_type[i][j] = 'L'; break;
      case 1:
	neuron_type[i][j] = 'U'; break;
      case 2:
	neuron_type[i][j] = 'D'; break;
      case 3:
	neuron_type[i][j] = 'R'; break;
      }
    }
}

// Setup the files to record neural activity
void setup_rec(int nrecord, 
	       int rec_neurons[][2],
	       real single_neuron_histogram[][MAXD][MAXD],
	       int has_been[MAXD][MAXD])
{
  int i,j,k;
  char s[1024];
  FILE *outf;
  int m;
  int good;
  real nx, ny;

  for (i = 0; i < nrecord; i++) {
    good = 0;
    while (good == 0) {
      nx = (int)(n*drand48());
      ny = (int)(n*drand48());
      if ((nx-n/2.)*(nx-n/2)+(ny-n/2.)*(ny-n/2.) < sn_radius*sn_radius) 
	good = 1;
    }
    rec_neurons[i][0] = (int)nx;
    rec_neurons[i][1] = (int)ny;
    for (j = 0; j < d; j++)
      for (k = 0; k < d; k++)
	single_neuron_histogram[i][j][k] = 0;	
  }
  
  sprintf(s,"%s_legend.dat",snfile);
  if (NULL == (outf = fopen(s, "w"))) {
    printf("error opening single neuron output file %s\n", s);
    printf("no output produced\n");
    return;
  }
  for (m = 0; m < nrecord; m++)
    fprintf(outf, "%5d  %5d  %5d\n", m, rec_neurons[m][0], rec_neurons[m][1]);
  fclose(outf);
  
  for (i = 0; i < d; i++)
    for (j = 0; j < d; j++)
      has_been[i][j] = 0;
}


// Setup global input with specified falloff type (square or gaussian)

void setup_initial_input(real input[MAXN][MAXN])
{

  int i,j;
  real x[MAXN];
  real cx, cy;
  real c;
  real fact;
  real half;

  half = (n-1.)/2.;
  for (i = 0; i < n; i++)
    x[i] = i-half;

  if (sq_falloff == 1) { // Square Falloff
		
    if (slimit == UNDEF) {
      printf("error: sq_falloff with undefined slimit\n");
      exit(0);
    }
    if (falloff != 0) 
      printf("warning: sq_falloff => ignoring non-zero gaussian falloff\n");
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {
        cx = fabs(x[i]);
	cy = fabs(x[j]);
	c = (cx > cy)?cx:cy;
	if (c > slimit)
	  fact = (half - c)/(half-slimit);
	else 
	  fact = 1;
	input[i][j] = bvalue * fact;
	}

  } 
  else  // gaussian falloff
  {               
 
   for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {
	double r = sqrt(x[i]*x[i] + x[j]*x[j]);
	if (falloff_band == UNDEF) 
	  input[i][j] = bvalue * exp(-falloff*r*r/((n/2.)*(n/2.)));
	else
          if (r <= (n/2.-falloff_band))
            input[i][j] = bvalue;
          else
            input[i][j] = bvalue * exp(-falloff*(r-n/2.+falloff_band)*(r-n/2.+falloff_band)/(falloff_band*falloff_band));
      }
    
    if (climit != UNDEF) 
      for (i = 0; i < n; i++)
	for (j = 0; j < n; j++)
	  if (sqrt(x[i]*x[i]+x[j]*x[j]) > climit)
	    input[i][j] = 0;
    
    if (slimit != UNDEF)
      for (i = 0; i < n; i++)
	for (j = 0; j < n; j++)
	  if ((fabs(x[j]) > slimit)||(fabs(x[i]) > slimit))
	    input[i][j] = 0;
    
  }

  printf("input[1][1] = %le\n", input[1][1]);
}


void zero_initial_firing_rates(real r[MAXN][MAXN])
{
  int i,j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      r[i][j] = init_noise*drand48();
} 

// Load initial firing rates if they are available. If not, randomize 
// the population
void setup_initial_firing_rates(real r[MAXN][MAXN])
{
  int i,j;
  FILE *f;
  char s[1024];

  if (strcmp(rfile,"")==0)
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
	r[i][j] = 1.e-3*drand48();
  else {
    sprintf(s,"%s.dat",rfile);
    if (NULL == (f = fopen(s,"r"))) {
      printf("error opening r file %s\n", s);
      exit(0);
    }
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
	fscanf(f, REAL, &(r[j][i]));
  }
  fclose(f);
} 

void setup_initial_orn(real orn_e[MAXN][MAXN], real orn_i[MAXN][MAXN]) {
  int i,j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      orn_e[i][j] = 0.0;
      orn_i[i][j] = 0.0;
    }
  }
}

//===========================================================================
// Tracking population activity and spiking and saving data
// to .dat files
//===========================================================================

void setup_tracking1(double *xcm, double *ycm, FILE **outf)
{
    // int i,j;
  char s[1024];
  
  *xcm = initxcm;
  *ycm = initycm;

  if (strcmp(trackfile,"")==0)
    outf = NULL;
  else {
    sprintf(s,"%s.dat", trackfile);
    *outf = fopen(s,"w");  
  }
}


void update_tracking1(real r[MAXN][MAXN], double *xcm, double *ycm, int *jump)
{
  int ref_x, ref_y; // the centers of our patch
  real xsum, ysum;
  real xrsum, yrsum;
  int i,j;

  ref_x = floor(*xcm);
  ref_y = floor(*ycm);

  xsum = ysum = 0;
  xrsum = yrsum = 0;
  for (i = -ntrack; i <= ntrack; i++)
    for (j = -ntrack; j <= ntrack; j++) {
      xsum += i*r[(ref_x+i+n)%n][(ref_y+j+n)%n];
      ysum += j*r[(ref_x+i+n)%n][(ref_y+j+n)%n];
      xrsum += r[(ref_x+i+n)%n][(ref_y+j+n)%n];  
      yrsum += r[(ref_x+i+n)%n][(ref_y+j+n)%n];
    }
  xsum /= xrsum;
  ysum /= yrsum;

  *xcm = ref_x+xsum;
  if (*xcm >= n) {
    *xcm -= n;
    *jump = 1;
  } else if (*xcm < 0) {
    *xcm += n;
    *jump = 1;
  } else
    *jump = 0;

  *ycm = ref_y+ysum;
  if (*ycm >= n) 
    *ycm -= n;
  else if (*ycm < 0)
    *ycm += n;
}



void output_population(int iter, char *fname, real r[MAXN][MAXN])
{
  char name[1024];

  if (iter >= 0)
    sprintf(name,"%s_%d.dat", fname, iter);
  else
    sprintf(name,"%s_m%d.dat", fname, -iter);
  FILE *outf = fopen(name,"w");

  fwrite(&r[0][0], sizeof(real), MAXN*MAXN, outf);
  /* for (int i = 0; i < n; i++) { */
  /*   for (int j = 0; j < n; j++)  */
  /*     fprintf(outf, "%f ", r[i][j]);  */
  /*   fprintf(outf,"\n"); */
  /* } */
  fclose(outf);
}


void output_population_decomp(int iter, char *fname, real r[MAXN][MAXN],
                              real Ws[MAXN][MAXN], real B[MAXN][MAXN],
                              FILE* outf)
{
  if (out_decomp == 0) {
    output_population(iter, fname, r);
    return;
  }
  int newf = outf==NULL;
  if (newf==1) {
    char name[1024];
    if (iter < 0)  
      sprintf(name,"%s_m%d.dat", fname, -iter);
    else
      sprintf(name,"%s_%d.dat", fname, iter);
    outf = fopen(name,"w");
  }
  fwrite(&r[0][0], sizeof(real), MAXN*MAXN, outf);
  fwrite(&Ws[0][0], sizeof(real), MAXN*MAXN, outf);
  fwrite(&B[0][0], sizeof(real), MAXN*MAXN, outf);

  if (newf==1) {
    fclose(outf);
  }
}


void calc_cm(real r[MAXN][MAXN], int track_center[2], int radius, double cm[2])
{
  int i,j;

  double sum[2];
  double norm;
  int is,js;

  sum[0] = sum[1] = 0;
  norm = 0;

  for (i = -radius; i <= radius; i++)
    for (j = -radius; j <= radius; j++) {
      if (i*i+j*j > radius*radius)
	continue;
      is = track_center[0]+i;
      js = track_center[1]+j;
      sum[0] += is*r[is][js];
      sum[1] += js*r[is][js];
      norm += r[is][js];
    }
  
  cm[0] = sum[0]/norm;
  cm[1] = sum[1]/norm;
}


int search_in_circle(real r[MAXN][MAXN], int old_center[2], int radius, int new_center[2])
{
  int i,j;
  int min_dist_sqr;
  int is,js;
  min_dist_sqr = MAXD*MAXD;
  for (i = -radius; i <= radius; i++) {
    for (j = -radius; j <= radius; j++) {
      if ((i*i+j*j) > radius*radius)
	continue;
      if (i*i+j*j >= min_dist_sqr)
	continue;
      is = old_center[0]+i;
      js = old_center[1]+j;
      if ((r[is][js] > min_rate)&&
	  	  (r[is][js] > r[is+2][js])&&(r[is][js] > r[is-2][js])&&(r[is][js] > r[is][js+2])&&(r[is][js] > r[is][js-2])) {
	new_center[0] = is;
	new_center[1] = js;
	min_dist_sqr = i*i+j*j;
      }
    }
  }
  if (min_dist_sqr == MAXD*MAXD) {
    return 0;
  } else {
    return 1;
  }
}





void il_calc_cm(real r[MAXNP][MAXNP], int track_center[2], int radius, double thres, double cm[2])
{
  int i,j;

  double sum[2];
  double norm;
  int is,js;

  sum[0] = sum[1] = 0;
  norm = 0;

  for (i = -radius; i <= radius; i++)
    for (j = -radius; j <= radius; j++) {
      if (i*i+j*j > radius*radius)
	continue;
      is = track_center[0]+i;
      js = track_center[1]+j;
      if (r[is][js] < thres)
	continue;
      sum[0] += is*r[is][js];
      sum[1] += js*r[is][js];
      norm += r[is][js];
    }
  
  cm[0] = sum[0]/norm;
  cm[1] = sum[1]/norm;
}


int il_search_in_circle(real r[MAXNP][MAXNP], int old_center[2], int radius, int new_center[2])
{
  int i,j;
  int min_dist_sqr;
  int is,js;
  min_dist_sqr = MAXD*MAXD;
  for (i = -radius; i <= radius; i++) {
    for (j = -radius; j <= radius; j++) {
      if ((i*i+j*j) > radius*radius)
	continue;
      if (i*i+j*j >= min_dist_sqr)
	continue;
      is = old_center[0]+i;
      js = old_center[1]+j;
      /* if (is >= MAXNP || js >= MAXNP) */
      /*   return 0; */
      if ((r[is][js] > il_min_value)&&
	  (r[is][js] >= r[is+1][js])&&(r[is][js] >= r[is-1][js])&&(r[is][js] >= r[is][js+1])&&(r[is][js] >= r[is][js-1])) {
	new_center[0] = is;
	new_center[1] = js;
	min_dist_sqr = i*i+j*j;
      }
    }
  }
  if (min_dist_sqr == MAXD*MAXD) {
    return 0;
  } else {
    return 1;
  }
}


void fft_storage_convert()
{
  int i,j;
  int ip,jp;
  int index;

  for(i = 0; i < np; i++)
    for(j = 0; j < np; j++) {
      ip = i-np/2;
      jp = j-np/2;
      
      if (jp < 0) {
	jp = -jp;
	ip = -ip;
      }
      if(ip < 0)
	ip = np+ip;
      
      index = ip*(np/2+1)+jp;
#ifdef CUDA
      real rv = fft_storage[index][0];
      real iv = fft_storage[index][1];
#else
      real rv = crealf(fft_storage[index]);
      real iv = cimagf(fft_storage[index]);
#endif
      fftmatrix[i][j] = sqrt(rv*rv+iv*iv);
    }
}

// checks to see if two points are the same for
// the purpose of tracking spiking
int il_same_point(double cm1[2], double cm2[2])
{
  if ((fabs(cm1[0]-cm2[0]) <= 2)&&(fabs(cm1[1]-cm2[1]) <= 2))
    return 1;
  if ((fabs(cm1[0]+cm2[0]) <= 2)&&(fabs(cm1[1]+cm2[1]) <= 2))
    return 1;
  return 0;
}


void init_il_tracking()
{
  int nfound;
  int offset;
  int i,j;
  int m;
  FILE *fftf;
  char s[1024];

  offset = np/2;
  int c[2];
  double cm[2];

  fft_storage_convert();

  sprintf(s,"%s.dat", il_trackfile);
  fftf = fopen(s, "w");
  for (i = 0; i < np; i++) {
    for (j = 0; j < np; j++)
      fprintf(fftf, "%13e ", fftmatrix[i][j]);
    fprintf(fftf,"\n");
  }
  fflush(fftf);
  fclose(fftf);

  nfound = 0;
  for (i = offset-il_initial_search_radius; i <= offset+il_initial_search_radius; i++)
    for (j = offset-il_initial_search_radius; j <= offset+il_initial_search_radius; j++) {
      if (i>MAXNP || j>MAXNP || i<0 || j<0) {
        return;
      }
      if ((i == offset)&&(j == offset)) 
	continue;
      if (((i-offset)*(i-offset) + (j-offset)*(j-offset)) > (il_initial_search_radius*il_initial_search_radius))
	continue;
      if ((fftmatrix[i][j] > il_min_value)&&
	  (fftmatrix[i][j] >= fftmatrix[i+1][j])&&(fftmatrix[i][j] >= fftmatrix[i-1][j])&&
	  (fftmatrix[i][j] >= fftmatrix[i][j+1])&&(fftmatrix[i][j] >= fftmatrix[i][j-1])) {
	c[0] = i; c[1] = j;
	il_calc_cm(fftmatrix, c, il_cm_radius, il_cm_thres, cm);
	cm[0] -= offset;
	cm[1] -= offset;
	for (m = 0; m < nfound; m++)
	  if(il_same_point(il_cm[m], cm) == 1) {
            // printf("il found same.\n");
	    break;
	  }
	if (m == nfound)
	  if (nfound < 3) {
	    il_cm[nfound][0] = cm[0];
	    il_cm[nfound][1] = cm[1];
	    nfound++;
	  } 
	// printf("il found new: %3d,%3d -> %13lf  %13lf\n", i, j, cm[0], cm[1]);
      } 
    }
}
	

void update_il_tracking(int iter)
{
  int offset;
  int il_prev_center[2];
  int i;
  int res;

  offset = np/2;

  fft_storage_convert();

  for (i = 0; i < 3; i++) {
    il_prev_center[0] = rint(il_cm[i][0]+offset);
    il_prev_center[1] = rint(il_cm[i][1]+offset);
  
    res = il_search_in_circle(fftmatrix, il_prev_center, il_search_radius, il_track_center[i]);
    il_calc_cm(fftmatrix, il_track_center[i], il_cm_radius, il_cm_thres, il_cm[i]);
       il_cm[i][0] = il_cm[i][0]-offset;
    il_cm[i][1] = il_cm[i][1]-offset;

    if ((res != 1)||(fabs(il_cm[i][0]) > il_max_radius)||(fabs(il_cm[i][1]) > il_max_radius)) 
      if (iter > il_reset) {
        // printf("warning: im reset tracking, iter = %10d, i = %d\n", iter, i);
	init_il_tracking();
	return;
      }
  }
}


//===========================================================================
// END Tracking
//===========================================================================


//===========================================================================
// Sets up trajectory data and
// creates velocity and head direction
// data from position data depending on imput parameters
//===========================================================================

// Get velocities from file if they exist
int get_velocity(FILE *vfile, real *v_x, real *v_y)
{
  if (fscanf(vfile, REALREAL, v_x, v_y) != 2)
    return 0;
  return 1;
}

// reads initial trajectory data from a file
void setup_trajectory_external(real pos[],
			       real *theta_v,
			       real *v,
			       real *up, 
			       real *left)
{
  real v_x, v_y;

  pos[0] = initxpos;
  pos[1] = initypos;

  printf("setup external:\n"); fflush(stdout);

  if (get_velocity(vfile, &v_x, &v_y) == 0) {
    printf("error reading velocity file\n"); fflush(stdout);
    exit(0);
  }
  *theta_v = atan2(v_x, v_y);
  *v = sqrt(v_x*v_x + v_y*v_y);
  *left = -sin(*theta_v);
  *up = -cos(*theta_v);

  printf("done setup.\n"); fflush(stdout);
}



void setup_trajectory_vstep(real pos[], 
			     real *theta_v, 
			     real *v, 
			     real *up,
			     real *left, 
			     real *time_since_turn)
{
  pos[0] = initxpos;
  pos[1] = initypos;
  
  *v = vstep_vstep;
  *theta_v = initial_theta;
  *left = -sin(*theta_v);
  *up = -cos(*theta_v);

  printf("v = %13le: %13le, %13le\n", *v, *left, *up);
}



void setup_trajectory(real pos[], real *theta_v, real *v,
		      real *up, real *left, real *time_since_turn)
{
  pos[0] = initxpos;
  pos[1] = initypos;

  if (initial_theta == UNDEF) {
    *theta_v = 2*PI*drand48();
    printf("initial theta -> %f\n", *theta_v);
  } else 
    *theta_v = initial_theta;
  *left = -sin(*theta_v);
  *up = -cos(*theta_v);
  printf("initialize: theta = %le, left = %le, up = %le\n", *theta_v, *left, *up);
  *v = init_v;
}


void setup_trajectory_smooth(real pos[], real *theta_v, real *v,
			     real *up, real *left, 
			     real *time_since_turn,
			     smooth_params *sp)
{
  pos[0] = initxpos;
  pos[1] = initypos;

  if (initial_theta == UNDEF) {
    *theta_v = 2*PI*drand48();
    printf("initial theta -> %f\n", *theta_v);
  } else 
    *theta_v = initial_theta;
  *left = -sin(*theta_v);
  *up = -cos(*theta_v);
  printf("initialize: theta = %le, left = %le, up = %le\n", *theta_v, *left, *up);
  *v = init_v;

  sp->accel = accel_fact*init_v/tau;
  sp->accel_sign = -1;
  sp->now_turning = 0;
  sp->steps_in_turn = 1;
  sp->time_to_stop = init_v/(sp->accel);
  sp->steps_to_stop = (int)((sp->time_to_stop)/dt);

  printf("accel: %f, sign: %d, time_to_stop: %f, steps_to_stop: %d\n", 
	 sp->accel, sp->accel_sign, sp->time_to_stop, sp->steps_to_stop);
}

#ifndef CUDA
static inline real min(real x, real y)
{
  return (x<y)?x:y;
}


static inline real max(real x, real y)
{
  return (x>y)?x:y;
}
#endif
//===========================================================================
// END Setup
//===========================================================================


//===========================================================================
// Update Postion based on some trajectory
// Updates with use either external data from file or a random trajectory
//===========================================================================
void update_position(real *pos, 
		     real *v, 
		     real *theta_v, 
		     int *vel, 
		     real *left, 
		     real *up, 
		     real *time_since_turn, 
		     int iter)
{

  real theta_step = 0.002; 
  real *allowed_th;
  int nth;

  real future_pos[2];
  real nmin;
  real minx, maxx, miny, maxy;
  int turn;

  allowed_th = (real *)malloc((int)(2*PI/theta_step+10)*sizeof(real));

  future_pos[0] = min((float)d,max(1.0,pos[0]-3*(*vel)*(*v)*dt*(*up)));
  future_pos[1] = min((float)d,max(1.0,pos[1]-3*(*vel)*(*v)*dt*(*left)));

  if (*time_since_turn > Nminstay) 
    turn = ((int)((Nmaxstay-Nminstay)*drand48())==0);
  else 
    turn = 0;

  if (turn||(future_pos[0] >= d-2)||(future_pos[0] <= 2)||
      (future_pos[1] >= d-2)||(future_pos[1] <= 2)) {
    
    fprintf(stderr,"turning: iter = %d\n", iter);

    // determine allowed values for theta

      nmin = (*v)*dt*Nminstay;
      minx = (3.-pos[0])/nmin;
      maxx = (d-2.-pos[0])/nmin;
      miny = (3.-pos[1])/nmin;
      maxy = (d-2.-pos[1])/nmin;

      nth = 0;
      for (real t = 0; t < 2*PI; t+= theta_step) 
	if ((((maxx <= 1)&&(t > acos(maxx))&&(t < 2*PI-acos(maxx)))||(maxx > 1))&&
	    (((minx >=-1)&&((t < acos(minx))||(t > 2*PI-acos(minx))))||(minx <-1))&&
	    (((maxy <= 1)&&((t < asin(maxy))||(t > PI-asin(maxy))))||(maxy>1))&&
	    (((miny >=-1)&&((t < PI-asin(miny))||(t > 2*PI+asin(miny))))||(miny<-1)))
	  allowed_th[nth++] = t;
      
      int index = floor(nth*drand48());
      if (index == nth)       // theoretically this has zero probability
	index = nth-1;
      *theta_v = allowed_th[index];
      *left = -sin(*theta_v);
      *up = -cos(*theta_v);
      *time_since_turn = 1;
  }
  (*time_since_turn) += dt;
  pos[0] = pos[0]-(*vel)*(*v)*dt*(*up);
  pos[1] = pos[1]-(*vel)*(*v)*dt*(*left);

  free(allowed_th);
}


int update_position_external(real *pos, real *v, real *theta_v, real *left, real *up)
{
  real v_x, v_y;
  
  pos[0] = pos[0] - (*v)*dt*(*up);
  pos[1] = pos[1] - (*v)*dt*(*left);

  if (get_velocity(vfile, &v_x, &v_y) == 0) 
    return 0;

  *theta_v = atan2(v_x, v_y);
  *v = sqrt(v_x*v_x + v_y*v_y);
  *left = -sin(*theta_v);
  *up = -cos(*theta_v);

  return 1;
}


void update_position_vstep(real *pos, real *v, real *theta_v, int *vel, real *left, real *up, real *time_since_turn, int iter)
{
  if (iter % vstep_nsteps == 0) {
    *v += vstep_vstep; 
    printf("update velocity: iter = %d -> %f\n", iter, *v);
  }
  pos[0] = pos[0]-(*vel)*(*v)*dt*(*up);
  pos[1] = pos[1]-(*vel)*(*v)*dt*(*left);
}


void update_position_det(real *pos, real *v, real *theta_v, int *vel, real *left, real *up, real *time_since_turn, int iter)
{

  real theta_step = 0.002; 
  real *allowed_th;
  int nth;

  real future_pos[2];
  real nmin;
  real minx, maxx, miny, maxy;

  allowed_th = (real *)malloc((int)(2*PI/theta_step+10)*sizeof(real));

  future_pos[0] = min((float)d,max(1.0,pos[0]-3*(*vel)*(*v)*dt*(*up)));
  future_pos[1] = min((float)d,max(1.0,pos[1]-3*(*vel)*(*v)*dt*(*left)));

  if ((*time_since_turn == 200)||(future_pos[0] >= d-2)||(future_pos[0] <= 2)||
      (future_pos[1] >= d-2)||(future_pos[1] <= 2)) {
    
    printf("turning: iter = %d\n", iter);

    // determine allowed values for theta

      nmin = (*v)*dt*Nminstay;
      minx = (3.-pos[0])/nmin;
      maxx = (d-2.-pos[0])/nmin;
      miny = (3.-pos[1])/nmin;
      maxy = (d-2.-pos[1])/nmin;

      nth = 0;
      for (real t = 0; t < 2*PI; t+= theta_step) 
	if ((((maxx <= 1)&&(t > acos(maxx))&&(t < 2*PI-acos(maxx)))||(maxx > 1))&&
	    (((minx >=-1)&&((t < acos(minx))||(t > 2*PI-acos(minx))))||(minx <-1))&&
	    (((maxy <= 1)&&((t < asin(maxy))||(t > PI-asin(maxy))))||(maxy>1))&&
	    (((miny >=-1)&&((t < PI-asin(miny))||(t > 2*PI+asin(miny))))||(miny<-1)))
	  allowed_th[nth++] = t;
      
      int index = iter%nth;
      *theta_v = allowed_th[index];
      printf("index = %d, theta = %f\n", index, *theta_v);
      *left = -sin(*theta_v);
      *up = -cos(*theta_v);
      *time_since_turn = 1;
  }
  (*time_since_turn) += dt;
  pos[0] = pos[0]-(*vel)*(*v)*dt*(*up);
  pos[1] = pos[1]-(*vel)*(*v)*dt*(*left);

  free(allowed_th);
}


void update_position_smooth(real *pos, 
			    real *v, 
			    real *theta_v, 
			    int *vel, 
			    real *left, 
			    real *up, 
			    real *time_since_turn, 
			    int iter,
			    smooth_params *sp)
{

  real theta_step = 0.002; 
  real *allowed_th;
  int nth;

  real future_pos[2];
  real nmin;
  real minx, maxx, miny, maxy;
  int turn;

  allowed_th = (real *)malloc((int)(2*PI/theta_step+10)*sizeof(real));

  future_pos[0] = min((float)d,max(1.0,pos[0]+init_v*cos(*theta_v)*(sp->time_to_stop)));
  future_pos[1] = min((float)d,max(1.0,pos[1]+init_v*sin(*theta_v)*(sp->time_to_stop)));

  if (*time_since_turn > Nminstay) {
    turn = ((int)((Nmaxstay-Nminstay)*drand48())==0);
    if (turn == 1)
      printf("spontaneous turn\n");
  }
  else 
    turn = 0;

  if ((!sp->now_turning)&&(turn||(future_pos[0] >= d-2)||(future_pos[0] <= 3)||
      (future_pos[1] >= d-2)||(future_pos[1] <= 3))) {    
    sp->now_turning = 1;
    sp->steps_in_turn = 0;
    *v = init_v;
    sp->accel_sign = -1;
    printf("turning: iter = %d, turn = %d, position: %f, %f\n", iter, turn, pos[0], pos[1]);
  }

  if (sp->now_turning) {
    *v = min(init_v, max(0.0,(*v)+(sp->accel_sign)*(sp->accel)*dt));
    sp->steps_in_turn++;
    if ((sp->steps_in_turn)==(sp->steps_to_stop)+1) {
      //choose new direction      
      nmin = init_v*dt*((sp->steps_to_stop)+2);
      minx = (3.-pos[0])/nmin;
      maxx = (d-2.-pos[0])/nmin;
      miny = (3.-pos[1])/nmin;
      maxy = (d-2.-pos[1])/nmin;

      nth = 0;
      for (real t = 0; t < 2*PI; t+= theta_step) 
	if ((((maxx <= 1)&&(t > acos(maxx))&&(t < 2*PI-acos(maxx)))||(maxx > 1))&&
	    (((minx >=-1)&&((t < acos(minx))||(t > 2*PI-acos(minx))))||(minx <-1))&&
	    (((maxy <= 1)&&((t < asin(maxy))||(t > PI-asin(maxy))))||(maxy>1))&&
	    (((miny >=-1)&&((t < PI-asin(miny))||(t > 2*PI+asin(miny))))||(miny<-1)))
	  allowed_th[nth++] = t;
      
      int index = floor(nth*drand48());
      if (index == nth)       // theoretically this has zero probability
	index = nth-1;
      *theta_v = allowed_th[index];
      printf("stoped!, index = %d, theta = %f\n", index, *theta_v);
      *left = -sin(*theta_v);
      *up = -cos(*theta_v);
      sp->accel_sign = 1;
    } else if ((sp->steps_in_turn)==2*(sp->steps_to_stop)+2) {
      sp->now_turning = 0;
      *v = init_v;
      (*time_since_turn = 0);
    }
  } else // not turning
    (*time_since_turn) += dt;

  pos[0] = pos[0]-(*vel)*(*v)*dt*(*up);
  pos[1] = pos[1]-(*vel)*(*v)*dt*(*left);
  
  free(allowed_th);
}


void init_longrange()
{
  nlr = 0;
}
    


void add_longrange(real r[MAXN][MAXN], real input[MAXN][MAXN])
{
  // real score;
  // real best_score;
  int i,j;
  int i1, j1, i2, j2;
  int nadd;

  // best_score = -1;

  real maxr = 0;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      if (r[i][j] > maxr)
	maxr = r[i][j];

  real r1 = 0;
  while (r1 < maxr*thres_lr) {
    i1 = (int)( n * drand48() );
    j1 = (int)( n * drand48() );
    r1 = r[i1][j1];
    if (normalise_lr == 1)
      r1 /= input[i1][j1];
  }

  real r2 = 0;
  i2 = i1; 
  j2 = j1;
  while ( (r2 < maxr*thres_lr) || (sqrt((i1-i2)*(i1-i2)+(j1-j2)*(j1-j2)) < blobspacing*minlr) ) {
    i2 = (int)( n * drand48() );
    j2 = (int)( n * drand48() );
    r2 = r[i2][j2];
    if (normalise_lr == 1)
      r2 /= input[i2][j2];
  }

  if (nlr < Nlr)
    nadd = nlr;
  else
    nadd = (int)(Nlr*drand48());

  wlr[nadd][0] = i1;
  wlr[nadd][1] = j1;
  wlr[nadd][2] = i2;
  wlr[nadd][3] = j2;

  printf("added longrange %4d: %d, %d   ; %d, %d\n", nadd, i1, j1, i2, j2);

  if (nlr < Nlr)
    nlr++;
}


void apply_long_range(real r[MAXN][MAXN], real rfield[MAXN][MAXN])
{
  int m;
  int i1, j1;
  int i2, j2;

  for (m = 0; m < nlr; m++) {
    i1 = wlr[m][0];
    j1 = wlr[m][1];
    i2 = wlr[m][2];
    j2 = wlr[m][3];
    rfield[i1][j1] += lramp*r[i2][j2];
    rfield[i2][j2] += lramp*r[i1][j1];
  }
}


void output_longrange(int iter)
{
  char s[1024];

  sprintf(s,"%s_%d.dat", lrfile, iter);
  FILE *outf = fopen(s,"w");
  int m;

  for (m = 0; m < nlr; m++) {
    for (int i = 0; i < 4; i++) 
      fprintf(outf, "%5d  ", wlr[m][i]); 
    fprintf(outf,"\n");
  }
  fclose(outf);
}

//===========================================================================
// END Update Postion
//===========================================================================


// Perform convolutions on the population activity with the filter matrix to get 
// new population activity
void convolve(fftwp_plan plf, fftwp_plan plr, fftwp_complex *ftl, int clear_storage)
{
  int i;

  fftwp_execute(plf);              // forward transform for the neuron data
  if (clear_storage == 1)
    for (i = 0; i < np*(np/2+1); i++) {
#ifdef CUDA
      fft_storage[i][0] = conv_temp[i][0];
      fft_storage[i][1] = conv_temp[i][1];
      conv_temp[i][0] = conv_temp[i][0]*ftl[i][0]/(np*np) - conv_temp[i][1]*ftl[i][1]/(np*np);
      conv_temp[i][1] = conv_temp[i][0]*ftl[i][1]/(np*np) + conv_temp[i][1]*ftl[i][0]/(np*np);
#else
      fft_storage[i] = conv_temp[i];
      conv_temp[i] *= ftl[i]/(np*np);  // multiply with fft of filter
#endif
    } else 
    for (i = 0; i < np*(np/2+1); i++) {
#ifdef CUDA
      fft_storage[i][0] += conv_temp[i][0];
      fft_storage[i][1] += conv_temp[i][1];
      conv_temp[i][0] = conv_temp[i][0]*ftl[i][0]/(np*np) - conv_temp[i][1]*ftl[i][1]/(np*np);
      conv_temp[i][1] = conv_temp[i][0]*ftl[i][1]/(np*np) + conv_temp[i][1]*ftl[i][0]/(np*np);
#else
      fft_storage[i] += conv_temp[i];
      conv_temp[i] *= ftl[i]/(np*np);  // multiply with fft of filter
#endif
    }
  fftwp_execute(plr);
}

// Update neuron activity
void update_neuron_activity(real r[MAXN][MAXN],
                            real r_nmda[MAXN][MAXN],                             
                            real r_ampa[MAXN][MAXN],                             
                            real r_tmp[MAXN][MAXN],                             
                            real input[MAXN][MAXN],
                            real Ws[MAXN][MAXN],
                            real B[MAXN][MAXN],
                            real orn_e[MAXN][MAXN],
                            real orn_i[MAXN][MAXN],
                            char neuron_type[MAXN][MAXN],
                            int vel, real v, real left, real up,
                            int iter,
                            real dvx, real dvy,
                            char spike[MAXN][MAXN])
{
  real rfield[MAXN][MAXN];
  int i,j;
  int index;
  real gradfact;
  // real sp;
  real D_e = noise_amp*noise_amp * 2.0/tau_e;
  real A_e = orn_uhl_A(D_e, tau_e, dt);
  real s0_e = noise_amp*6.0;
  real D_i = noise_amp*noise_amp * 2.0/tau_i;
  real A_i = orn_uhl_A(D_i, tau_i, dt);
  real s0_i = noise_amp*6.0;
  int m_sub = round(1.0/(noise_amp*noise_amp));
  
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) {
      int index = i*np+j;
      inr[index] = inl[index] = ind[index] = inu[index] = 0;
    }

  gradfact = 1;
  for (i = 0; i < n; i++) {
    for(j = 0; j < n; j++) {
      index = np*i+j;
      if (dvx != UNDEF)	
        gradfact = 1+(dvx*(i-n/2.) + dvy*(j-n/2.))/((real)n);
      switch(neuron_type[i][j]) {
       case 'L':
         rfield[i][j] = vel*(v/init_v)*v_gain*gradfact*left;
         inl[index] = r[i][j]; inr[index] = 0; ind[index] = 0; inu[index] = 0;
         break;
       case 'R':
         rfield[i][j] = -vel*(v/init_v)*v_gain*gradfact*left;
         inr[index] = r[i][j]; inl[index] = 0; ind[index] = 0; inu[index] = 0;
         break;
       case 'U':
         rfield[i][j] = vel*(v/init_v)*v_gain*gradfact*up;
         inu[index] = r[i][j]; inr[index] = 0; ind[index] = 0; inl[index] = 0;
         break;
       case 'D':
         rfield[i][j] = -vel*(v/init_v)*v_gain*gradfact*up;
         ind[index] = r[i][j]; inu[index] = 0; inr[index] = 0; inl[index] = 0;
         break;
      }

      rfield[i][j] *= input[i][j];

    }
  }
  if (longrange==1) 
    apply_long_range(r, rfield);

  convolve(plf,plr,ftl,1);
  convolve(prf,prr,ftr,0);
  convolve(pdf,pdr,ftd,0);
  convolve(puf,pur,ftu,0);


  // update population activity according to the specified update rule;
  // spiking or non-spiking and clipping or non-clipping
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      index = np*i+j;
      if (fi == 0) {
        B[i][j] = rfield[i][j] + input[i][j];  
        Ws[i][j] = outl[index]+outr[index]+outd[index]+outu[index];
        if (scale_weights == 0)
          rfield[i][j] += Ws[i][j];
        else
          rfield[i][j] += (input[i][j]/bvalue)*Ws[i][j];
      } else {
        rfield[i][j] = 0.0;
        input[i][j] = fi_amp * (sin(2.0*PI*fi_freq*iter*dt + 2.0*PI*(i*n+j)/(n*n)) + 1.0);
      }
      
      real rarg = rfield[i][j]+input[i][j];

      if (noise_amp > 0 && spiking == 0) {
        if (gaussian == 1) {
          rarg += noise_amp*gaussrand();
        } else {
          if (bipolar == 1) {
            rarg += noise_amp*(drand48()-0.5);
          } else {
            if (constant == 1) {
              rarg += noise_amp*0.5;
            } else {
              if (ornstein == 1) {
                orn_e[i][j] = orn_uhl_rand(s0_e, orn_e[i][j], dt, tau_e, A_e);
                orn_i[i][j] = orn_uhl_rand(s0_i, orn_i[i][j], dt, tau_i, A_i);
                rarg += (orn_e[i][j] - orn_i[i][j]);
              } else {
                rarg += noise_amp*drand48();
              }
            }
          }
        }
      }
      if (nmda == 1) {
        r_nmda[i][j] += -(dt/tau_nmda)*r_nmda[i][j] + (dt/tau_nmda)*rarg;
        rarg += k_nmda*r_nmda[i][j];
      } else {
        if (nmda_nl == 1) {
          if (nmda_add == 1) {
            if (tau_nmda != 0) {
              r_nmda[i][j] += -(dt/tau_nmda)*r_nmda[i][j] + k_nmda*(dt/tau_nmda)*rarg;
            } else {
              r_nmda[i][j] = k_nmda*rarg;
            }
          } else {
            if (tau_nmda != 0) {
              r_nmda[i][j] += -(dt/tau_nmda)*r_nmda[i][j] + (dt/tau_nmda)*fsigm(rarg, center_nmda, slope_nmda);
            } else {
              r_nmda[i][j] = fsigm(rarg, center_nmda, slope_nmda);
            }
            if (nmda_mul == 0) {
              rarg += k_nmda*r_nmda[i][j];
            } else {
              rarg *= (1.0 + k_nmda*(r_nmda[i][j]-0.5));
            }
          }
        } else {
          if (nmda_ampa == 1) {
            r_nmda[i][j] += -(dt/tau_nmda)*r_nmda[i][j] + (dt/tau_nmda)*fsigm(rarg, center_nmda, slope_nmda)*r[i][j];
            r_ampa[i][j] += -(dt/tau_ampa)*r_ampa[i][j] + (dt/tau_nmda)*r[i][j];
          }
        }
      }
      if (nmda_nl == 0) {
        if (nonlinear == 1) {
          if (rarg > 0) {
            rarg = nonlinf_static(rarg);
          }
        } else {
          if (nonlinear_cont == 1) {
            rarg = nonlinf_cont(rarg);
          } else {
            if (nonlinear_step == 1) {
              rarg = nonlinf_step(rarg);
            } else {
              if (nonlinear_equiv == 1) {
                rarg = nonlinf_equiv(rarg);
              }
            }
          }
        }
        if (rarg < rarg_min)
          rarg_min = rarg;
        if (rarg > rarg_max)
          rarg_max = rarg;

        if (spiking == 1) {
          spike[i][j] = 0;
          if (rarg > 0) {
            for (int nsub=0; nsub < m_sub; nsub++) {
              int spikesub = (drand48() < rarg*dt)? 1:0;
              if (spikesub != 0) {
                if (spikesubcounter[i][j] == m_sub-1) {
                  spike[i][j] = 1.0;
                  spikesubcounter[i][j] = 0;
                } else {
                  spikesubcounter[i][j]++;
                }
              }
            }
          }
          r[i][j] += -(dt/tau)*r[i][j] + dt*spike[i][j]; 

        } else if (notau == 1) {
          if (rarg > 0) {
            r[i][j] += -(dt/tau)*r[i][j]+dt*rarg;
          } else
            r[i][j] += -(dt/tau)*r[i][j];

        } else { // older version
          if (rarg > 0) {
            r[i][j] += -(dt/tau)*r[i][j]+(dt/tau)*rarg;
          } else
            r[i][j] += -(dt/tau)*r[i][j];
        }
      } else { /* nmda_nl == 1 */
        r_tmp[i][j] += -(dt/tau)*r_tmp[i][j]+(dt/tau)*rarg;
        real r_nmda_add = 0;
        if (nmda_add == 1) {
          r_nmda_add = r_nmda[i][j];
        }
        if (fi == 0) {
          if (nonlinear == 1) {
            if (r_tmp[i][j] > 0) {
              r[i][j] = nonlinf_static(r_tmp[i][j]+r_nmda_add);
            }
          } else {
            if (nonlinear_cont == 1) {
              r[i][j] = nonlinf_cont(r_tmp[i][j]+r_nmda_add);
            } else {
              if (nonlinear_step == 1) {
                r[i][j] = nonlinf_step(r_tmp[i][j]+r_nmda_add);
              } else {
                if (nonlinear_equiv == 1) {
                  r[i][j] = nonlinf_equiv(r_tmp[i][j]+r_nmda_add);
                }
              }
            }
          }
        } else {
          // Use subthreshold V_m for fi curve estimation (no recurrent connections anyway)
          r[i][j] = r_tmp[i][j];
        }
      }
      if (clip != UNDEF)
        if (r[i][j] > clip)
          r[i][j] = clip;
    }
}


void clear_single_neuron(int rec_neurons[][2], 
		    real single_neuron_histogram[][MAXD][MAXD], 
		    int has_been[MAXD][MAXD])
{
  int m,i,j;

  for (i = 0; i < d; i++)
    for (j = 0; j < d; j++) {
      for (m = 0; m < nrecord; m++)
	single_neuron_histogram[m][i][j] = 0;
      has_been[i][j] = 0;
    }
}

// recording single neuron activity
void update_rec(real r[MAXN][MAXN],
		real pos[2],
		int rec_neurons[][2],
		real single_neuron_histogram[][MAXD][MAXD],
		int has_been[MAXD][MAXD])
{
  int m,i,j;
  int pos0, pos1;

  pos0 = (int)round(pos[0]);
  pos1 = (int)round(pos[1]);
  
  if ((pos0 < 0)||(pos0 >= MAXD)||(pos1 < 0)||(pos1 >= MAXD))

    printf("warning: trajectory out of rec bound: %3d, %3d\n", pos0, pos1);

  else {

    for (m = 0; m < nrecord; m++) {
      i = rec_neurons[m][0];
      j = rec_neurons[m][1];
      single_neuron_histogram[m][pos0][pos1] += 
	r[i][j];
    }
    (has_been[pos0][pos1])++;

  }
}

// Setup spikeout files
void setup_spikeout(int nrecord, FILE *spikefiles[])
{
  int i;
  char s[1024];

  for (i = 0; i < nrecord; i++) {
    sprintf(s, "%s_%d.dat", spikename, i);
    spikefiles[i] = fopen(s, "w");
  }
}


void close_spikeout(int nrecord, FILE *spikefiles[])
{
  int i;

  for (i = 0; i < nrecord; i++)
    fclose(spikefiles[i]);
}


// record single neuron activity
void output_single_neuron(int iter, 
			  int has_been[MAXD][MAXD], 
			  int rec_neurons[][2],
			  real single_neuron_histogram[][MAXD][MAXD])
{
  int i,j;
  FILE *outf;
  char s[1024];
  int m;
  int norm;
  if (iter < 0)
    sprintf(s,"%s.dat", snfile);
  else 
    sprintf(s,"%s_%d.dat", snfile, iter);
  if (NULL == (outf = fopen(s, "w"))) {
    printf("error opening single neuron output file %s\n", s);
    printf("no output produced\n");
    return;
  }
  for (m = 0; m < nrecord; m++) {
    /* if (iter < 0) */
    /*   sprintf(s,"%s_%03d.dat", snfile, m); */
    /* else  */
    /*   sprintf(s,"%s_%d_%03d.dat", snfile, iter, m); */
    /* if (NULL == (outf = fopen(s, "w"))) { */
    /*   printf("error opening single neuron output file %s\n", s); */
    /*   printf("no output produced\n"); */
    /*   return; */
    /* } */
    for (i = 0; i < d; i++) {
      for (j = 0; j < d; j++) {
	norm = has_been[i][j];
	if (norm == 0)
	  norm = 1;
	// fprintf(outf, "%f  ", single_neuron_histogram[m][i][j]/norm);
        real rnormed = single_neuron_histogram[m][i][j]/norm;
	fwrite(&rnormed, sizeof(real), 1, outf);
      }
      // fprintf(outf,"\n");
    }
    // fclose(outf);
  }
  fclose(outf);

  if (iter < 0)
    sprintf(s,"%s_w.dat", snfile);
  else 
    sprintf(s,"%s_%d_w.dat", snfile, iter);
  if (NULL == (outf = fopen(s, "w"))) {
    printf("error opening single neuron output file %s\n", s);
    printf("no output produced\n");
    return;
  }
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++) {
      fprintf(outf, "%d  ", has_been[i][j]);
    }
    fprintf(outf,"\n");
  }

  fclose(outf);
  
}


void output_track(int iter, real pos[], double xcm, double ycm, int jump, FILE *outf)
{
  int i;

  if (outf == NULL)
    return;

  real riter = (real)iter;
  real rxcm = (real)xcm;
  real rycm = (real)ycm;
  real rjump = (real)jump;
  
  fwrite(&riter, sizeof(real), 1, outf);
  fwrite(&pos[0], sizeof(real), 2, outf);
  fwrite(&rxcm, sizeof(real), 1, outf);
  fwrite(&rycm, sizeof(real), 1, outf);
  fwrite(&rjump, sizeof(real), 1, outf);

  // fprintf(outf, "%5d  %13le  %13le  %13le  %13le  %1d  ", 
  //  	  iter, pos[0], pos[1], xcm, ycm, jump);

  if (il_track == 1) {
    for (i = 0; i < 3; i++) {
      real ril_cm0 = (real)il_cm[i][0];
      real ril_cm1 = (real)il_cm[i][1];
      fwrite(&ril_cm0, sizeof(real), 1, outf);
      fwrite(&ril_cm1, sizeof(real), 1, outf);
      // fprintf(outf, "%13le  %13le  ", il_cm[i][0], il_cm[i][1]);
    }
  }

  // fprintf(outf, "\n");
}


void flow(int vel, real v, real theta_v, int n, int nphase, 
	  char neuron_type[MAXN][MAXN], real input[MAXN][MAXN],
          real r[MAXN][MAXN], real r_nmda[MAXN][MAXN], real r_ampa[MAXN][MAXN], real r_tmp[MAXN][MAXN], real Ws[MAXN][MAXN], real B[MAXN][MAXN],
          real orn_e[MAXN][MAXN], real orn_i[MAXN][MAXN])
{
  real left, up;
  int iter;
  char spike[MAXN][MAXN];
 
  left = -sin(theta_v);
  up = -cos(theta_v);
  for (iter = 1; iter <= n; iter++) {
    update_neuron_activity(r,r_nmda,r_ampa,r_tmp,input,Ws,B,orn_e,orn_i,neuron_type,vel,v,left,up,1,UNDEF,UNDEF,spike);
    if (iter %niterprint == 0)
      printf("flow part %d: %d\n", nphase, iter); fflush(stdout);
  }
}


int main(int argc, char *argv[])
{
#ifdef PROFILER
  ProfilerStart("gperf.out");
#endif
  int has_been[MAXD][MAXD];
  real r[MAXN][MAXN];  // Container to hold population firing rates
  real r_nmda[MAXN][MAXN];
  real r_ampa[MAXN][MAXN];
  real r_tmp[MAXN][MAXN];
  real Ws[MAXN][MAXN];  // recurrent input
  real B[MAXN][MAXN];  // feedforward input
  char spike[MAXN][MAXN];
  real input[MAXN][MAXN];
  real orn_e[MAXN][MAXN];
  real orn_i[MAXN][MAXN];
  char neuron_type[MAXN][MAXN];
  int rec_neurons[MAXNRECORD][2];
  FILE *spikefiles[MAXNRECORD];
  real single_neuron_histogram[MAXNRECORD][MAXD][MAXD];
  // real b;  
  real pos[2];
  real left, up;
  // real track;
  int jump;
  int iter;
  int vel;
  real time_since_turn;
  real v;
  real theta_v;
  double xcm, ycm;
  FILE *troutf;
  smooth_params sp;
  int in;
  int njumps = 0;

  // for vstepclear
  int i,j;
  real vstep_initr[MAXN][MAXN];
  real vstep_initxcm, vstep_initycm;
  real vstep_initpos[2];

#ifdef CUDA
  findCudaDevice(argc, (const char **)argv);
#endif
  
  theta_v = initial_theta;
  left = -sin(theta_v);
  up = -cos(theta_v);

  get_parameters(argc, argv);

  if (seed !=UNDEF)
    srand48(seed);
  
  // Setup the network wiring

  setup_weights();
  
  // Setup neuron types
  
  setup_type_distribution(neuron_type);
  
  // long-range network
  if (longrange == 1)
    init_longrange();

  // output options
  if (vstep != 1) {
    setup_rec(nrecord, rec_neurons, single_neuron_histogram, has_been);
    if (spikeout == 1)
	setup_spikeout(nrecord, spikefiles);
  }

  // tracking
  // print_parameters();



// Setup initial feedforward network input
  setup_initial_input(input);

  if (ornstein)
    setup_initial_orn(orn_e, orn_i);
  
// Save initial network state
  output_population_decomp(-3, popfile, input, Ws, B, NULL);


  if (periodic == 1)
    setup_initial_firing_rates(r);
  else
    zero_initial_firing_rates(r);

  if (nmda == 1 || nmda_nl == 1) {
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        r_nmda[i][j] = 0;
        r_ampa[i][j] = 0;
        r_tmp[i][j] = 0;
      }
    }
  }

  if (spiking == 1)
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
        spikesubcounter[i][j] = 0;

  if (flow_theta == UNDEF) {
    flow(0, init_v, 0, nflow, 1, neuron_type, input, r, r_nmda, r_ampa, r_tmp, Ws, B, orn_e, orn_i);
    output_population_decomp(-2, popfile, r, Ws, B, NULL);
    flow(1, init_v, PI/5, nflow, 2, neuron_type, input, r, r_nmda, r_ampa, r_tmp, Ws, B, orn_e, orn_i);
    output_population_decomp(-1, popfile, r, Ws, B, NULL);
    flow(1, init_v, PI/2-PI/5, nflow, 3, neuron_type, input, r, r_nmda, r_ampa, r_tmp, Ws, B, orn_e, orn_i);
  } else {
    flow(1, init_v, flow_theta, nflow, 3, neuron_type, input, r, r_nmda, r_ampa, r_tmp, Ws, B, orn_e, orn_i);
  }

  FILE* ppopfile = NULL;
  if (out_decomp == 1) {
    char name[1024];
    sprintf(name,"%s.dat", popfile);
    FILE* ppopfile = fopen(name, "wb");
  }
  
  output_population_decomp(0, popfile, r, Ws, B, ppopfile);

  setup_tracking1(&xcm, &ycm, &troutf);


  if (il_track == 1)
    init_il_tracking(); 

  if (vstep == 1)
    setup_trajectory_vstep(pos, &theta_v, &v, &up, &left, &time_since_turn);
  else if (smooth == 1)
    setup_trajectory_smooth(pos, &theta_v, &v, &up, &left, &time_since_turn, &sp);
  else if (external_trajectory == 1) {
    setup_trajectory_external(pos, &theta_v, &v, &up, &left);
    printf("external trajectory: %d\n", external_trajectory); fflush(stdout);
  } else
    setup_trajectory(pos, &theta_v, &v, &up, &left, &time_since_turn);

  vel = 1;

  
  if (vstep == 1)
    if (vstep_clear == 1) {
      for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
          vstep_initr[i][j] = r[i][j];
      vstep_initxcm = xcm;
      vstep_initycm = ycm;
      vstep_initpos[0] = pos[0];
      vstep_initpos[1] = pos[1];

      if (il_track == 1)
        init_il_tracking(); 
    }
  //loop through all positions
  for (iter = 1; iter <= niter; iter++) {

    if (iter%niterprint == 0) {
      // printf("%d\n", iter);
      printf("%d", (int)(iter*10.0/niter));
      // printf(".");
      fflush(stdout);
    }

    if (longrange == 1) {
      if (iter%naddlr == 0)
        for (int m = 0; m < 1; m++)
          if ((nlr < Nlr)||(refresh_lr==1)) {
            add_longrange(r, input);
            if ((nlr == Nlr)&&(refresh_lr==0))
              output_longrange(iter);
          }
      if (((nlr < Nlr)||(refresh_lr == 1))&&(iter%ndumplr == 0))
        output_longrange(iter);
    }

    vel = (iter <= 80)?0:1;
    if (vstep == 1)
      update_position_vstep(pos, &v, &theta_v, &vel, &left, &up, &time_since_turn, iter);
    else if (det == 1)
      update_position_det(pos, &v, &theta_v, &vel, &left, &up, &time_since_turn, iter);
    else if (smooth == 1)
      update_position_smooth(pos, &v, &theta_v, &vel, &left, &up, &time_since_turn, iter, &sp);
    else if (external_trajectory == 1) {
      if (update_position_external(pos, &v, &theta_v, &left, &up) == 0) {
        printf("no more data.\n"); fflush(stdout);
        iter = niter+1;
      }
    } else
      update_position(pos, &v, &theta_v, &vel, &left, &up, &time_since_turn, iter);

    if (vstep == 1)
      if (vstep_clear == 1)
        if (iter % vstep_nsteps == 0) {  // reset to initial state
          for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
              r[i][j] = vstep_initr[i][j];
          xcm = vstep_initxcm;
          ycm = vstep_initycm;
          pos[0] = vstep_initpos[0];
          pos[1] = vstep_initpos[1];
        }
    
    update_neuron_activity(r,r_nmda,r_ampa,r_tmp,input,Ws,B,orn_e,orn_i,neuron_type,vel,v,left,up,iter,dvx,dvy,spike);


    if (spikeout == 1) 
      for (in = 0; in < nrecord; in++)
        if (spike[rec_neurons[in][0]][rec_neurons[in][1]] == 1)
		{
          fprintf(spikefiles[in], "%8d  %13le  %13le\n", iter, pos[0], pos[1]);
		  }


    update_tracking1(r, &xcm, &ycm, &jump);
    njumps += jump;
    if (njump_limit != UNDEF) {
      if (iter == njump_steps) {
        if (njumps > njump_limit) {
          printf("Maximum number of jumps reached; aborting now\n"); fflush(stdout);
          if (out_decomp == 1)
            fclose(ppopfile);
          if (spikeout == 1)
            close_spikeout(nrecord, spikefiles);
          if (troutf != NULL)
            fclose(troutf);
          exit(0);
        }
      }
    }

    if (il_track == 1) 
      if (iter== il_reset)
	init_il_tracking();
      else
	update_il_tracking(iter);

    if (vstep != 1)
      update_rec(r, pos, rec_neurons, single_neuron_histogram, has_been);

    output_track(iter, pos, xcm, ycm, jump, troutf);

    if (ndump_population > 0)
      if (iter%ndump_population == 0)
        output_population_decomp(iter, popfile, r, Ws, B, ppopfile);
    
    if (ndump_single > 0)
      if (iter%ndump_single == 0)
	if (vstep != 1)
	  output_single_neuron(iter, has_been, rec_neurons, single_neuron_histogram);
    
    if (special_dumps == 1) {
      if ((iter >= 1000000)&&(iter <= 1002500)&&(iter%50 == 0)) 
        output_population_decomp(iter, popfile, r, Ws, B, NULL);
      
      if ((iter >= 1000000)&&(iter <= 1050000)&&(iter%1000 == 0)) 
        output_population_decomp(iter, popfile, r, Ws, B, NULL);
    }

    if (nclear_single > 0)
      if (iter%nclear_single == 0)
	if (vstep != 1)
	  clear_single_neuron(rec_neurons, single_neuron_histogram, has_been);

  }

  output_population_decomp(-1, popfile, r, Ws, B, NULL);
  if (out_decomp == 1)
    fclose(ppopfile);
  if (vstep != 1)
    output_single_neuron(-1, has_been,rec_neurons,single_neuron_histogram);

  if (spikeout == 1)
    close_spikeout(nrecord, spikefiles);

  if (troutf != NULL)
    fclose(troutf);

  printf("rarg_max = %f\n", rarg_max);
  printf("rarg_min = %f\n", rarg_min);
#ifdef PROFILER
  ProfilerStop();
#endif
}

