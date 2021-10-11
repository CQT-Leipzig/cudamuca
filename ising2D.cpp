#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <stdint.h>
#include <cmath>
#include <vector>
#include <limits>
#include <sys/time.h>
#include <mpi.h>
#include "Random123/philox.h"
#include "Random123/examples/uniform.hpp"
#include "muca.hpp"

// choose random number generator
typedef r123::Philox4x32_R<7> RNG;

// This includes my_uint64 type
#include "ising2D_io.hpp"

using namespace std;

// calculate bin index from energy E
inline unsigned EBIN(int E)
{
  return (E + (N << 1)) >> 2;
}

// calculate energy difference of one spin flip
int localE(unsigned idx, vector<int>& lattice)
{
  int right = idx + 1;
  int left = static_cast<int>(idx) - 1;
  int up = idx + L;
  int down = static_cast<int>(idx) - L;

  // check periodic boundary conditions
  if (right % L == 0) right -= L;
  if (idx % L == 0) left += L;
  if (up > static_cast<int>(N) - 1 ) up -= N;
  if (down < 0 ) down += N;

   return -lattice.at(idx) *
     ( lattice.at(right) +
       lattice.at(left) +
       lattice.at(up) +
       lattice.at(down) );
}

// calculate total energy
int calculateEnergy(vector<int>& lattice)
{
  int sum = 0;

  for (size_t i = 0; i < N; i++) {
    sum += localE(i, lattice);
  }

  // divide out double counting
  return (sum >> 1);
}

// multicanonical Markov chain update (single spin flip)
void mucaUpdate(float rannum, int& energy, vector<int>& lattice, vector<float>& h_log_weights, unsigned idx)
{
  // precalculate energy change
  int dE = -2 * localE(idx, lattice);

  // flip with propability W(E_new)/W(E_old)
  if (rannum < expf(h_log_weights.at(EBIN(energy+dE))-h_log_weights.at(EBIN(energy)))) {
    lattice.at(idx) = -lattice.at(idx);
    energy += dE;
  }
}

// multicanonical iteration including initial thermalization
void mucaIteration(vector<int>& h_lattice, vector<my_uint64>& h_histograms, vector<float>& h_log_weights, int& energy, unsigned worker, unsigned iteration, unsigned seed, my_uint64 NUPDATES_THERM, my_uint64 NUPDATES)
{
  // initialize two RNGs
  // one for acceptance propability (k1)
  // and one for selection of a spin (same for all workers) (k2)
  RNG rng;
  RNG::key_type k1 = {{worker, 0xdecafbad}};
  RNG::key_type k2 = {{0xC001CAFE, 0xdecafbad}};
  RNG::ctr_type c = {{0, seed, iteration, 0xBADC0DED}};//0xBADCAB1E
  RNG::ctr_type r1, r2;

  // reset local histogram
  for (size_t i = 0; i < h_histograms.size(); i++) {
    h_histograms.at(i) = 0;
  }

  // thermalization
  for (my_uint64 i = 0; i < NUPDATES_THERM; i++) {
    if(i%4 == 0) {
      ++c[0];
      r1 = rng(c, k1); r2 = rng(c, k2);
    }
    unsigned idx = static_cast<unsigned>(r123::u01fixedpt<float>(r2.v[i%4]) * N);
    mucaUpdate(r123::u01fixedpt<float>(r1.v[i%4]), energy, h_lattice, h_log_weights, idx);
  }

  // estimate current probability distribution of W(E)
  for (my_uint64 i = 0; i < NUPDATES; i++) {
    if(i%4 == 0) {
      ++c[0];
      r1 = rng(c, k1); r2 = rng(c, k2);
    }
    unsigned idx = static_cast<unsigned>(r123::u01fixedpt<float>(r2.v[i%4]) * N);
    mucaUpdate(r123::u01fixedpt<float>(r1.v[i%4]), energy, h_lattice, h_log_weights, idx);
    // add to local histogram
    h_histograms.at(EBIN(energy)) += 1;
  }
}

int main(int argc, char* argv[])
{
  // read command line arguments and initialize constants (see ising2D_io.hpp)
  parseArgs(argc, argv);

  // initialize MPI
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  unsigned WORKER = static_cast<unsigned>(rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  NUM_WORKERS = static_cast<unsigned>(mpi_size);

  // initialize local (LxL) lattice
  RNG rng;
  RNG::key_type k = {{WORKER, 0xdecafbad}};
  RNG::ctr_type c = {{0, seed, 0xBADCAB1E, 0xBADC0DED}};
  RNG::ctr_type r; 
  vector<int> h_lattice(N);
  for (size_t i = 0; i < h_lattice.size(); i++) {
    if(i%4 == 0) {
      ++c[0];
      r = rng(c, k);
    }
    h_lattice.at(i) = 2*(r123::u01fixedpt<float>(r.v[i%4]) < 0.5)-1;
  }

  // initialize local energy
  int energy = calculateEnergy(h_lattice);

  // initialize local weights
  vector<float> h_log_weights(N + 1, 0.0f);
  // initialize local histogram
  vector<my_uint64> h_histograms(N + 1, 0);
  // initialize global histogram
  vector<my_uint64> mpi_hist(N + 1, 0);

  // timing and statistics
  vector<long double> times;
  timespec start, stop;
  ofstream iterfile;
  if (WORKER == 0) {
    iterfile.open("iterations.dat");
  }

  // initial estimate of width at infinite temperature
  // (random initialization requires practically no thermalization)
  unsigned width = 10;
  long double nupdates_run = 1;
  // heuristic factor that determines the number of statistic per iteration
  // should be related to the integrated autocorrelation time
  double z = 2.25;
  // main iteration loop
  for (size_t k = 0; k < MAX_ITER; k++) {
    // start timer
    MPI_Barrier(MPI_COMM_WORLD);
  	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    // distribute weight of rank0 to all
    MPI_Bcast(h_log_weights.data(), h_log_weights.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // acceptance rate and correlation time corrected "random walk"
    // in factor 30 we adjusted acceptance rate and >L range requirement of our present Ising situation
    NUPDATES_THERM = 30*width;
    if(width<N) {
      // 6 is motivated by the average acceptance rate of a multicanonical simulation ~0.45 -> (1/0.45)**z~6
      nupdates_run = 6*pow(width,z)/NUM_WORKERS;
    }
    else {
      // for a flat spanning histogram, we assume roughly equally distributed
      // walkers and reduce the thermalization time
      // heuristic modification factor;
      // (>1; small enough to introduce statistical fluctuations on the convergence measure)
      nupdates_run *= 1.1;
    }
    // local iterations on each task, writing to local histograms
    NUPDATES = static_cast<my_uint64>(nupdates_run)+1;
    mucaIteration(h_lattice, h_histograms, h_log_weights, energy, WORKER, k, seed, NUPDATES_THERM, NUPDATES);
    // merge histograms to rank0
    MPI_Reduce(h_histograms.data(), mpi_hist.data(), h_histograms.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    // stop timer
   	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    long double elapsed = 1e9* (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);
    times.push_back(elapsed);
    TOTAL_THERM  +=NUPDATES_THERM;
    TOTAL_UPDATES+=NUPDATES;

    int converged = 0;
    // check for convergence only on rank0
    if (WORKER == 0) {
      // flatness in terms of kullback-leibler-divergence;
      // requires sufficient thermalization!!!
      double dk  = d_kullback(mpi_hist);
      if (dk<1e-4) { converged = 1;}
      iterfile << "#NITER = " << k  << " width= " << width << "; nupdates= " << nupdates_run <<" dk= " << dk << endl;
      writeHistograms(h_log_weights, mpi_hist, iterfile);
      // measure width of the current histogram
      size_t start,end;
      getHistogramRange(mpi_hist, start, end);
      unsigned width_new = end-start;
      if (width_new > width) width = width_new;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast convergence state to all ranks
    MPI_Bcast(&converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (converged == 1) break;

    // update logarithmic weights with basic scheme if not converged on rank0
    if (WORKER == 0) {
      updateWeights(h_log_weights, mpi_hist);
    }
  }

  if (WORKER == 0) {
    iterfile.close();

    ofstream sout;
    sout.open("stats.dat");
    writeStatistics(times, sout);
    sout << "total number of thermalization steps/Worker : " << TOTAL_THERM << "\n";
    sout << "total number of iteration updates   /Worker : " << TOTAL_UPDATES << "\n";
    sout << "total number of all updates         /Worker : " << TOTAL_THERM+TOTAL_UPDATES << "\n";
    sout.close();
  }

  if (production) {
    if (WORKER==0) {
      std::cout << "start production run ..." << std::endl;
    }
    // thermalization
    NUPDATES_THERM = pow(N,z);
    mucaIteration(h_lattice, h_histograms, h_log_weights, energy, WORKER, 0, seed+1000, NUPDATES_THERM, 0);
    size_t JACKS = 100;
    NUPDATES = NUPDATES_PRODUCTION/JACKS;
    // start timer
    MPI_Barrier(MPI_COMM_WORLD);
  	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    // loop over Jackknife bins
    for (size_t k = 0; k < JACKS; k++) {
      // local iterations on each task, writing to local histograms
      mucaIteration(h_lattice, h_histograms, h_log_weights, energy, WORKER, k, seed+2000, 0, NUPDATES);
      // merge histograms to rank0
      MPI_Reduce(h_histograms.data(), mpi_hist.data(), h_histograms.size(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

      if (WORKER == 0) {
        std::stringstream filename;
        filename << "production" << std::setw(3) << std::setfill('0') << k << ".dat";
        iterfile.open(filename.str().c_str());
        writeHistograms(h_log_weights, mpi_hist, iterfile);
        iterfile.close();
      }
    }
    // stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
    if (WORKER==0) {
      std::cout << "production run updates  JACK: " << NUPDATES     << "*WORKER \n";
      std::cout << "production run updates total: " << NUPDATES*100 << "*WORKER \n";
      std::cout << "production run time total   : " << (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1e-9 << "s\n"; 
      ofstream sout;
      sout.open("stats.dat", std::fstream::out | std::fstream::app);
      sout << "production run updates  JACK: " << NUPDATES     << "*WORKER \n";
      sout << "production run updates total: " << NUPDATES*100 << "*WORKER \n";
      sout << "production run time total   : " << (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec)*1e-9 << "s\n"; 
      sout.close();
    }
  }
  MPI_Finalize();

  return 0;
}
