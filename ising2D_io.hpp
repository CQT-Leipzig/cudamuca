#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

// POSIX standard (required for parsing command line arguments, may not be available on Windows)
#include <getopt.h>

typedef unsigned long long my_uint64;
double my_uint64_max = pow(2.0,64)-1;

// system parameters: set here or via command line
unsigned MAX_ITER = 1000;
unsigned L;
unsigned seed;
unsigned NUM_WORKERS;
int REQUESTED_GPU;
bool modify;
bool production;

// system parameters: determined by program
my_uint64 NUPDATES_THERM;
my_uint64 NUPDATES;
my_uint64 NUPDATES_PRODUCTION;
long double TOTAL_THERM   = 0.0;
long double TOTAL_UPDATES = 0.0;
unsigned N;

// system parameters: copy for GPU
#ifdef __CUDACC__
__constant__ unsigned d_L;
__constant__ unsigned d_N;
__constant__ unsigned d_NUM_WORKERS;
#endif

// print log-weights and histogram to filestream
void writeHistograms(const std::vector<float>& weights, const std::vector<my_uint64>& histogram, std::ofstream& fout)
{
  for (size_t i = 0; i < histogram.size(); i++) {
    // do not write unoccupied energy bins
    if (i == 1 || i == histogram.size() - 2) {
      continue;
    }
    fout << 4 * static_cast<int>(i) - 2 * static_cast<int>(N) << " " << std::setprecision(10) << weights.at(i) << " " << histogram.at(i) << std::endl;
  }
}

// print statistics to filestream
void writeStatistics(const std::vector<long double>& times, std::ofstream& fout)
{
  int iter = times.size();

  double total = 0.0;
  fout << "Simulation parameters: " << std::endl;
  fout << "L = " << L << std::endl;
  fout << "seed = " << seed << std::endl;
  fout << "MODIFY = " << (modify ? "ON" : "OFF") << std::endl;
  fout << "PRODUCTION = " << (production ? "ON" : "OFF") << std::endl;
  if (production) fout << "UPDATES PRODUCTION = " << NUPDATES_PRODUCTION << std::endl;
  fout << "NUM_WORKERS = " << NUM_WORKERS << std::endl;
  fout << "UPDATES PER WORKER = " << NUPDATES << std::endl;
  fout << std::endl;
  fout << "#iter | time in nsec" << std::endl;
  fout << "--------------------" << std::endl;
  for (size_t i = 0; i < times.size(); i++) {
    fout << std::setw(5) << i + 1 << " | " << std::setw(10) << std::setprecision(5) << times.at(i) << std::endl;
    total += times.at(i);
  }

  fout << std::endl;
  fout << "Simulation finished after " << iter << " iterations." << std::endl;
  fout << "Total simulation time: " << std::setprecision(5) << total << " nsec" << std::endl;
  fout << "Average time per spin flip: " << std::setprecision(5) << total / (NUM_WORKERS*(TOTAL_THERM+TOTAL_UPDATES)) << " nsec" << std::endl;
}

// print command-line help
void print_usage(char* progName)
{
  std::cout << "Usage :" << std::endl;
#ifndef __CUDACC__
  std::cout << progName << " [mt] [-s seed] [-p nupdates] -L size" << std::endl;
#endif
#ifdef __CUDACC__
  std::cout << progName << " [mt] [-s seed] [-p nupdates] [-i dev] -L size -W workers" << std::endl;
#endif
  std::cout << "-m turns on modifyWeights, default: false" << std::endl;
  std::cout << "-p final production run (set number of updates, if selected), default: 0" << std::endl;
  std::cout << "-L sets the system size, required parameter" << std::endl;
  std::cout << "-s sets the initial seed, default: 1000" << std::endl;
#ifdef __CUDACC__
  std::cout << "-i select device from list of available GPUs, optional parameter, default: automatic" << std::endl;
  std::cout << "-W sets the number of workers, required parameter" << std::endl;
#endif
}

// argument parser for command line
void parseArgs(int ac, char** av)
{
  int option = 0;
  L = 0;
  N = 0;
  seed = 1000;
  modify = false;
  production = false;
#ifdef __CUDACC__
  NUM_WORKERS = 0;
  REQUESTED_GPU = -1;
#endif
  NUPDATES_PRODUCTION = 0;

#ifdef __CUDACC__
  while ((option = getopt(ac, av, "mp:s:L:W:i:")) != -1) {
#else
  while ((option = getopt(ac, av, "mp:s:L:")) != -1) {
#endif
    switch(option) {
      case 'm' : modify = true;
                 break;
      case 'p' : production = true;
                 NUPDATES_PRODUCTION=strtoll(optarg,NULL,10);
                 break;
      case 's' : seed = atoi(optarg);
                 break;
      case 'L' : L = atoi(optarg);
                 break;
#ifdef __CUDACC__
      case 'W': NUM_WORKERS = atoi(optarg);
                break;
      case 'i': REQUESTED_GPU = atoi(optarg);
                break;
#endif
      default: print_usage(av[0]);
               exit(-1);
    }
  }
  N = L * L;
  if (L == 0 || N == 0) {
      print_usage(av[0]);
      exit(-1);
  }
}
