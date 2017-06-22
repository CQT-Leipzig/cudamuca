#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <sys/time.h>
#include "Random123/philox.h"
#include "Random123/examples/uniform.hpp"
#include "muca.hpp"

#include <errno.h>

// choose random number generator
typedef r123::Philox4x32_R<7> RNG;

// This includes my_uint64 type
#include "ising2D_io.hpp"

// 256 threads per block ensures the possibility of full occupancy
// for all compute capabilities if thread count small enough
#define WORKERS_PER_BLOCK 256
#define WORKER (blockIdx.x * blockDim.x + threadIdx.x)

// launch bounds depend on compute capability
#if __CUDA_ARCH__ >= 300
    #define MY_KERNEL_MIN_BLOCKS   2048/WORKERS_PER_BLOCK
#elif __CUDA_ARCH__ >= 200
    #define MY_KERNEL_MIN_BLOCKS   1536/WORKERS_PER_BLOCK
#else
    #define MY_KERNEL_MIN_BLOCKS   0
#endif

// random access to textures in global memory is faster due to caching
// this texture holds the logarithmic weights for MUCA
texture<float, 1, cudaReadModeElementType> t_log_weights;

using namespace std;

// calculate bin index from energy E
__device__ inline unsigned EBIN(int E)
{
  return (E + (d_N << 1)) >> 2;
}

// calculate energy difference of one spin flip
__device__ __forceinline__ int localE(unsigned idx, int8_t* lattice)
{ 
  int right = idx + 1;
  int left = static_cast<int>(idx) - 1;
  int up = idx + d_L;
  int down = static_cast<int>(idx) - d_L;
  
  // check periodic boundary conditions
  if (right % d_L == 0) right -= d_L;
  if (idx % d_L == 0) left += d_L;
  if (up > static_cast<int>(d_N - 1) ) up -= d_N;
  if (down < 0 ) down += d_N;
   
   return -lattice[idx * d_NUM_WORKERS + WORKER] *
     ( lattice[right * d_NUM_WORKERS + WORKER] +
       lattice[left * d_NUM_WORKERS + WORKER] +
       lattice[up * d_NUM_WORKERS + WORKER] + 
       lattice[down * d_NUM_WORKERS + WORKER] );
}

// calculate total energy
__device__ int calculateEnergy(int8_t* lattice)
{
  int sum = 0;

  for (size_t i = 0; i < d_N; i++) {
    sum += localE(i, lattice);
  }
  // divide out double counting
  return (sum >> 1); 
}

// multicanonical Markov chain update (single spin flip)
__device__ __forceinline__ bool mucaUpdate(float rannum, int* energy, int8_t* d_lattice, unsigned idx)
{
  // precalculate energy difference
  int dE = -2 * localE(idx, d_lattice);

  // flip with propability W(E_new)/W(E_old)
  // weights are stored in texture memory for faster random access
  if (rannum < expf(tex1Dfetch(t_log_weights, EBIN(*energy + dE)) - tex1Dfetch(t_log_weights, EBIN(*energy)))) {
    d_lattice[idx * d_NUM_WORKERS + WORKER] = -d_lattice[idx * d_NUM_WORKERS + WORKER];
    *energy += dE;
    return true;
  }
  return false;
}

// initial calculation of total energy per worker 
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
computeEnergies(int8_t *d_lattice, int* d_energies)
{
  d_energies[WORKER] = calculateEnergy(d_lattice);
}

// multicanonical iteration including initial thermalization
__global__ void
__launch_bounds__(WORKERS_PER_BLOCK, MY_KERNEL_MIN_BLOCKS)
mucaIteration(int8_t* d_lattice, my_uint64* d_histogram, int* d_energies, unsigned iteration, unsigned seed, my_uint64 d_NUPDATES_THERM, my_uint64 d_NUPDATES)
{
  // initialize two RNGs
  // one for acceptance propability (k1)
  // and one for selection of a spin (same for all workers) (k2)
  RNG rng;
  RNG::key_type k1 = {{WORKER, 0xdecafbad}};
  RNG::key_type k2 = {{0xC001CAFE, 0xdecafbad}};
  RNG::ctr_type c = {{0, seed, iteration, 0xBADC0DED}};//0xBADCAB1E
  RNG::ctr_type r1, r2; 
 
  // reset global histogram
  for (size_t i = 0; i < ((d_N + 1) / d_NUM_WORKERS) + 1; i++) {
    if (i*d_NUM_WORKERS + WORKER < d_N + 1) {
      d_histogram[i * d_NUM_WORKERS + WORKER] = 0;
    }
  }
  __syncthreads();

  int energy;
  energy = d_energies[WORKER];

  // thermalization
  for (size_t i = 0; i < d_NUPDATES_THERM; i++) {
    if(i%4 == 0) {
      ++c[0];
      r1 = rng(c, k1); r2 = rng(c, k2);
    }
    unsigned idx = static_cast<unsigned>(r123::u01fixedpt<float>(r2.v[i%4]) * d_N);
    mucaUpdate(r123::u01fixedpt<float>(r1.v[i%4]), &energy, d_lattice, idx);
  }

  // estimate current propability distribution of W(E)
  for (my_uint64 i = 0; i < d_NUPDATES; i++) {
    if(i%4 == 0) {
      ++c[0];
      r1 = rng(c, k1); r2 = rng(c, k2);
    }
    unsigned idx = static_cast<unsigned>(r123::u01fixedpt<float>(r2.v[i%4]) * d_N);
    mucaUpdate(r123::u01fixedpt<float>(r1.v[i%4]), &energy, d_lattice, idx);
    // add to global histogram 
    atomicAdd(d_histogram + EBIN(energy), 1);
  }

  d_energies[WORKER] = energy;
}

int main(int argc, char** argv)
{
  // read command line arguments and initialize constants (see ising2D_io.hpp)
  parseArgs(argc, argv);

  if (NUM_WORKERS % WORKERS_PER_BLOCK != 0) {
    cerr << "ERROR: NUM_WORKERS must be multiple of " << WORKERS_PER_BLOCK << endl;
  }

  // select device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if(REQUESTED_GPU >= 0 and REQUESTED_GPU < deviceCount) cudaSetDevice(REQUESTED_GPU);

  // prefer cache over shared memory
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // figure out optimal execution configuration
  // based on GPU architecture and generation
  int currentDevice;
  cudaGetDevice(&currentDevice);
  int maxresidentthreads, totalmultiprocessors;
  cudaDeviceGetAttribute(&maxresidentthreads, cudaDevAttrMaxThreadsPerMultiProcessor, currentDevice);
  cudaDeviceGetAttribute(&totalmultiprocessors, cudaDevAttrMultiProcessorCount, currentDevice);
  int optimum_number_of_workers = maxresidentthreads*totalmultiprocessors;
  if (NUM_WORKERS == 0) {
    NUM_WORKERS = optimum_number_of_workers;
  }
 
  // copy constants to GPU
  cudaMemcpyToSymbol(d_N, &N, sizeof(unsigned));
  cudaMemcpyToSymbol(d_L, &L, sizeof(unsigned));
  cudaMemcpyToSymbol(d_NUM_WORKERS, &NUM_WORKERS, sizeof(unsigned));

  // initialize NUM_WORKERS (LxL) lattices
  RNG rng;
  vector<int8_t> h_lattice(NUM_WORKERS * N);
  int8_t* d_lattice;
  cudaMalloc((void**)&d_lattice, NUM_WORKERS * N * sizeof(int8_t));
  for (unsigned worker=0; worker < NUM_WORKERS; worker++) {
    RNG::key_type k = {{worker, 0xdecafbad}};
    RNG::ctr_type c = {{0, seed, 0xBADCAB1E, 0xBADC0DED}};
    RNG::ctr_type r;
    for (size_t i = 0; i < N; i++) {
      if (i%4 == 0) {
        ++c[0];
        r = rng(c, k);
      }
      h_lattice.at(i*NUM_WORKERS+worker) = 2*(r123::u01fixedpt<float>(r.v[i%4]) < 0.5)-1;
    }
  }
  cudaMemcpy(d_lattice, h_lattice.data(), NUM_WORKERS * N * sizeof(int8_t), cudaMemcpyHostToDevice);

  // initialize all energies
  int* d_energies;
  cudaMalloc((void**)&d_energies, NUM_WORKERS * sizeof(int));
  computeEnergies<<<NUM_WORKERS / WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice, d_energies);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cout << "Error: " << cudaGetErrorString(err) << " in " << __FILE__ << __LINE__ << endl;
    exit(err);
  }

  // initialize ONE global weight array
  vector<float> h_log_weights(N + 1, 0.0f);
  float* d_log_weights;
  cudaMalloc((void**)&d_log_weights, (N + 1) * sizeof(float));
  // texture for weights
  cudaBindTexture(NULL, t_log_weights, d_log_weights, (N + 1) * sizeof(float));

  // initialize ONE global histogram
  vector<my_uint64> h_histogram((N + 1), 0);
  my_uint64* d_histogram;
  cudaMalloc((void**)&d_histogram, (N + 1) * sizeof(my_uint64));

  // timing and statistics
  vector<long double> times;
  timespec start, stop;
  ofstream iterfile;

  iterfile.open("run_iterations.dat");
  // initial estimate of width at infinite temperature 
  // (random initialization requires practically no thermalization)
  unsigned width = 10;
  double nupdates_run = 1;
  // heuristic factor that determines the number of statistic per iteration
  // should be related to the integrated autocorrelation time
  double z = 2.25;
  // main iteration loop
  for (size_t k=0; k < MAX_ITER; k++) {
    // start timer
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    // copy global weights to GPU
    cudaMemcpy(d_log_weights, h_log_weights.data(), (N + 1) * sizeof(float), cudaMemcpyHostToDevice);
    // acceptance rate and correlation time corrected "random walk"
    // in factor 30 we adjusted acceptance rate and >L range requirement of our present Ising situation
    NUPDATES_THERM = 30*width;
    if(width<N) {
      // 6 is motivated by the average acceptance rate of a multicanonical simulation ~0.45 -> (1/0.45)**z~6
      nupdates_run = 6*pow(width,z)/NUM_WORKERS;
    }
    else{
      // for a flat spanning histogram, we assume roughly equally distributed
      // walkers and reduce the thermalization time
      // heuristic modification factor;
      // (>1; small enough to introduce statistical fluctuations on the convergence measure)
      nupdates_run *= 1.1;
    }
    NUPDATES = static_cast<my_uint64>(nupdates_run)+1;
    // local iteration on each thread, writing to global histogram
    mucaIteration<<<NUM_WORKERS / WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice, d_histogram, d_energies, k, seed, NUPDATES_THERM, NUPDATES);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      cout << "Error: " << cudaGetErrorString(err) << " in " << __FILE__ << __LINE__ << endl;
      exit(err);
    }
    // copy global histogram back to CPU
    cudaMemcpy(h_histogram.data(), d_histogram, (N + 1) * sizeof(my_uint64), cudaMemcpyDeviceToHost);
    // stop timer
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    long double elapsed = 1e9* (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
    times.push_back(elapsed);
    TOTAL_THERM   += NUPDATES_THERM;
    TOTAL_UPDATES += NUPDATES;
   
    // flatness in terms of kullback-leibler-divergence; 
    // requires sufficient thermalization!!! 
    double dk  = d_kullback(h_histogram);
    iterfile << "#NITER = " << k  << " dk=" << dk << endl;
    writeHistograms(h_log_weights, h_histogram, iterfile);
    if (dk<1e-4) {
      break;
    }

    // measure width of the current histogram
    size_t start,end;
    getHistogramRange(h_histogram, start, end);
    unsigned width_new = end-start;
    if (width_new > width) width=width_new;

    // update logarithmic weights with basic scheme if not converged
    updateWeights(h_log_weights, h_histogram);
  }
  iterfile.close();

  ofstream sout;
  sout.open("stats.dat");
  writeStatistics(times, sout);
  sout << "total number of thermalization steps/Worker : " << TOTAL_THERM << "\n";
  sout << "total number of iteration updates   /Worker : " << TOTAL_UPDATES << "\n";
  sout << "total number of all updates         /Worker : " << TOTAL_THERM+TOTAL_UPDATES << "\n";
  sout.close();

  if (production) {
    std::cout << "start production run ..." << std::endl;
    // copy global weights to GPU
    cudaMemcpy(d_log_weights, h_log_weights.data(), (N + 1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // thermalization
    NUPDATES_THERM = pow(N,z);
    mucaIteration<<<NUM_WORKERS / WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice, d_histogram, d_energies, 0, seed+1000, NUPDATES_THERM, 0);
    // set jackknife  
    size_t JACKS = 100;
    NUPDATES = NUPDATES_PRODUCTION/JACKS;
    // loop over Jackknife bins
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    for (size_t k = 0; k < JACKS; k++) {
      cudaDeviceSynchronize();
      // local production on each thread, writing to global histogram
      mucaIteration<<<NUM_WORKERS / WORKERS_PER_BLOCK, WORKERS_PER_BLOCK>>>(d_lattice, d_histogram, d_energies, k, seed+2000, 0, NUPDATES);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        cout << "Error: " << cudaGetErrorString(err) << " in " << __FILE__ << __LINE__ << endl;
        exit(err);
      }
      // copy global histogram back to CPU
      cudaMemcpy(h_histogram.data(), d_histogram, (N + 1) * sizeof(my_uint64), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      std::stringstream filename;
      filename << "production" << std::setw(3) << std::setfill('0') << k << ".dat";
      iterfile.open(filename.str().c_str());
      writeHistograms(h_log_weights, h_histogram, iterfile);
      iterfile.close();
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    std::cout << "production run updates  JACK: " << NUPDATES     << "*WORKER \n";
    std::cout << "production run updates total: " << NUPDATES*100 << "*WORKER \n";
    std::cout << "production run time total   : " << (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1e-9 << "s\n"; 
    sout.open("stats.dat", std::fstream::out | std::fstream::app);
    sout << "production run updates  JACK: " << NUPDATES     << "*WORKER \n";
    sout << "production run updates total: " << NUPDATES*100 << "*WORKER \n";
    sout << "production run time total   : " << (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1e-9 << "s\n"; 
    sout.close();
  }

  return 0;
}
