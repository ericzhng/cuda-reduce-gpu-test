
#include "cpu_subs.h"


double reduction_cpu(double *h_data, unsigned data_size){

	double sum = 0.0;
	for (unsigned i = 0; i < data_size; i++){
		sum += h_data[i];
	}
	return sum;
}



float timing_experiment_cpu(double(*func)(double*, unsigned), double *h_data, int data_size, int nreps, double *result)
{
	PreciseTimeDef timePast;

	timePast.StartTime();
	for (unsigned i = 0; i < nreps; i++){
		*result = func(h_data, data_size);
	}

	return (float) timePast.EndTime_blank() * 1E3;
}
