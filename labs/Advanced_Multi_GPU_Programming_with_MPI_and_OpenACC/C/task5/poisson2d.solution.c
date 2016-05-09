/*
 *  Copyright 2015 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#ifdef _OPENACC
#include <openacc.h>
#endif /*_OPENACC*/
#include "common.h"

#include <mpi.h>

#define NY 4096
#define NX 4096

real A[NY][NX];
real Aref[NY][NX];
real Anew[NY][NX];
real rhs[NY][NX];

real to_left[NY];
real from_left[NY];
real to_right[NY];
real from_right[NY];

int main(int argc, char** argv)
{
    int iter_max = 1000;
    
    const real tol = 1.0e-5;

    int rank = 0;
    int size = 1;

    //Initialize MPI and determine rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if ( size > MAX_MPI_SIZE )
    {
        if ( 0 == rank )
        {
            fprintf(stderr,"ERROR: Only up to %d MPI ranks are supported.\n",MAX_MPI_SIZE);
        }
        return -1;
    }
    
    dim2 size2d = size_to_2Dsize(size);
    int sizex = size2d.x;
    int sizey = size2d.y;
    assert(sizex*sizey == size);
    
    int rankx = rank%sizex;
    int ranky = rank/sizex;

    memset(A, 0, NY * NX * sizeof(real));
    memset(Aref, 0, NY * NX * sizeof(real));
    
    // set rhs
    for (int iy = 1; iy < NY-1; iy++)
    {
        for( int ix = 1; ix < NX-1; ix++ )
        {
            const real x = -1.0 + (2.0*ix/(NX-1));
            const real y = -1.0 + (2.0*iy/(NY-1));
            rhs[iy][ix] = expr(-10.0*(x*x + y*y));
        }
    }
    
#if _OPENACC
    acc_device_t device_type = acc_get_device_type();
    if ( acc_device_nvidia == device_type )
    {
        int ngpus=acc_get_num_devices(acc_device_nvidia);
        
        int devicenum=rank%ngpus;
        acc_set_device_num(devicenum,acc_device_nvidia);
    }
    // Call acc_init after acc_set_device_num to avoid multiple contexts on device 0 in multi GPU systems
    acc_init(device_type);
#endif /*_OPENACC*/

    // Ensure correctness if NX%sizex != 0
    int chunk_sizex = ceil( (1.0*NX)/sizex );

    int ix_start = rankx * chunk_sizex;
    int ix_end   = ix_start + chunk_sizex;

    // Do not process boundaries
    ix_start = max( ix_start, 1 );
    ix_end = min( ix_end, NX - 1 );

    // Ensure correctness if NY%sizey != 0
    int chunk_sizey = ceil( (1.0*NY)/sizey );

    int iy_start = ranky * chunk_sizey;
    int iy_end   = iy_start + chunk_sizey;

    // Do not process boundaries
    iy_start = max( iy_start, 1 );
    iy_end = min( iy_end, NY - 1 );

    if ( rank == 0) printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);

    if ( rank == 0) printf("Calculate reference solution and time serial execution.\n");
    StartTimer();
    poisson2d_serial( rank, iter_max, tol );
    double runtime_serial = GetTimer();

    //Wait for all processes to ensure correct timing of the parallel version
    MPI_Barrier( MPI_COMM_WORLD );
    if ( rank == 0) printf("Parallel execution.\n");
    StartTimer();
    int iter  = 0;
    real error = 1.0;
    
    #pragma acc data copy(A) copyin(rhs) create(Anew,to_left,from_left,to_right,from_right)
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

        #pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( A[iy][ix+1] + A[iy][ix-1]
                                                       + A[iy-1][ix] + A[iy+1][ix] ));
                error = fmaxr( error, fabsr(Anew[iy][ix]-A[iy][ix]));
            }
        }
        
        real globalerror = 0.0;
        MPI_Allreduce( &error, &globalerror, 1, MPI_REAL_TYPE, MPI_MAX, MPI_COMM_WORLD );
        error = globalerror;
        
        #pragma acc kernels
        for( int ix = ix_start; ix < ix_end; ix++ )
        {
            A[iy_start][ix] = Anew[iy_start][ix];
            A[(iy_end-1)][ix] = Anew[(iy_end-1)][ix];
        }
        
        #pragma acc kernels async(2)
        for( int iy = iy_start; iy < iy_end; iy++ )
        {
                to_left[iy]  = Anew[iy][ix_start];
                to_right[iy] = Anew[iy][ix_end-1];
        }
        
        #pragma acc kernels async(1)
        for (int iy = (iy_start+1); iy < (iy_end-1); iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }

        //Periodic boundary conditions
        int topy    = (ranky == 0) ? (sizey-1) : ranky-1;
        int bottomy = (ranky == (sizey-1)) ? 0 : ranky+1;
        int top    = topy    * sizex + rankx;
        int bottom = bottomy * sizex + rankx;
        #pragma acc host_data use_device( A )
        {
            //1. Sent row iy_start (first modified row) to top receive lower boundary (iy_end) from bottom
            MPI_Sendrecv( &A[iy_start][ix_start], (ix_end-ix_start), MPI_REAL_TYPE, top   , 0, &A[iy_end][ix_start], (ix_end-ix_start), MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //2. Sent row (iy_end-1) (last modified row) to bottom receive upper boundary (iy_start-1) from top
            MPI_Sendrecv( &A[(iy_end-1)][ix_start], (ix_end-ix_start), MPI_REAL_TYPE, bottom, 0, &A[(iy_start-1)][ix_start], (ix_end-ix_start), MPI_REAL_TYPE, top   , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
        
        int leftx  = (rankx == 0) ? (sizex-1) : rankx-1;
        int rightx = (rankx == (sizex-1)) ? 0 : rankx+1;
        int left   = ranky * sizex + leftx;
        int right  = ranky * sizex + rightx;
        #pragma acc wait(2)
        #pragma acc host_data use_device( to_left, from_left, to_right, from_right )
        {
            //1. Sent to_left starting from first modified row (iy_start) to last modified row to left and receive the same rows into from_right from right 
            MPI_Sendrecv( to_left+iy_start, (iy_end-iy_start), MPI_REAL_TYPE, left   , 0, from_right+iy_start, (iy_end-iy_start), MPI_REAL_TYPE, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //2. Sent to_right starting from first modified row (iy_start) to last modified row to left and receive the same rows into from_left from left
            MPI_Sendrecv( to_right+iy_start, (iy_end-iy_start), MPI_REAL_TYPE, right , 0, from_left+iy_start, (iy_end-iy_start), MPI_REAL_TYPE, left  , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
        #pragma acc kernels
        for( int iy = iy_start; iy < iy_end; iy++ )
        {
                A[iy][ix_start-1] = from_left[iy];
                A[iy][ix_end]     = from_right[iy];
        }
        
        #pragma acc wait
        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    MPI_Barrier( MPI_COMM_WORLD );
    double runtime = GetTimer();

    if (check_results( rank, ix_start, ix_end, iy_start, iy_end, tol ) && rank == 0)
    {
        printf( "Num GPUs: %d with a (%d,%d) layout.\n", size, sizey,sizex );
        printf( "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, efficiency: %8.2f%\n", NY,NX, runtime_serial/ 1000.0, size, runtime/ 1000.0, runtime_serial/runtime, runtime_serial/(size*runtime)*100 );
    }

    MPI_Finalize();
    return 0;
}

#include "poisson2d_serial.h"
