
#include "subroutines.cuh"

#ifdef _WIN32
	#include <windows.h>
	#include <profileapi.h>
#elif __linux__
	#include <chrono>
#else
	#error "Not Implemented"
#endif

// Precise timing function
#ifdef _WIN32
	
	typedef struct
	{
		LARGE_INTEGER  freq;        // ticks per second
		LARGE_INTEGER  start, end;

		double second;

		void StartTime(){
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&start);
		}

		double EndTime(){
			QueryPerformanceCounter(&end);
			second = (end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
			printf("\tTotal %-8.5fs passed on ", second);
			return second;
		}

		double EndTime_blank(){
			QueryPerformanceCounter(&end);
			second = (end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
			return second;
		}
	} PreciseTimeDef;

#elif __linux__

	typedef struct
	{
		double second;
		typedef std::chrono::high_resolution_clock clock_;
		typedef std::chrono::duration<double, std::ratio<1> > sec_;
		std::chrono::time_point<clock_> beg_;

		void StartTime(){
			beg_ = clock_::now();
		}
		double EndTime() {
			second = std::chrono::duration_cast<sec_>
				(clock_::now() - beg_).count();
			printf("\tTotal %-8.5fs passed on ", second);
			return second;
		}
		double EndTime_blank() {
			second = std::chrono::duration_cast<sec_>
				(clock_::now() - beg_).count();
			return second;
		}
	} PreciseTimeDef;

#elif __APPLE__
	#error "To be implemented later"
#else
	#error "Unknown compiler"
#endif



double reduction_cpu(double *h_data, unsigned data_size);


float timing_experiment_cpu(double(*func)(double*, unsigned), double *h_data, int data_size, int nreps, double *result);

