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

#ifndef LAPLACE2D_SERIAL_H
#define LAPLACE2D_SERIAL_H

void poisson2d_serial( int rank, int iter_max, real tol )
{
    int iter  = 0;
    real error = 1.0;
    #pragma acc data copy(Aref) copyin(rhs) create(Anew)
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

#pragma acc kernels
        for( int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( Aref[iy][ix+1] + Aref[iy][ix-1]
                                                       + Aref[iy-1][ix] + Aref[iy+1][ix] ));
                error = fmaxr( error, fabsr(Anew[iy][ix]-Aref[iy][ix]));
            }
        }

#pragma acc kernels
        for( int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                Aref[iy][ix] = Anew[iy][ix];
            }
        }

        //Periodic boundary conditions
#pragma acc kernels
        for( int ix = 1; ix < NX-1; ix++ )
        {
                Aref[0][ix]     = Aref[(NY-2)][ix];
                Aref[(NY-1)][ix] = Aref[1][ix];
        }
#pragma acc kernels
        for( int iy = 1; iy < NY-1; iy++ )
        {
                Aref[iy][0]     = Aref[iy][(NX-2)];
                Aref[iy][(NX-1)] = Aref[iy][1];
        }

        if(rank == 0 && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }

}

int check_results( int rank, int ix_start, int ix_end,  int iy_start, int iy_end, real tol )
{
    int result_correct = 1;
    for( int iy = iy_start; iy < iy_end && (result_correct == 1); iy++)
    {
        for( int ix = ix_start; ix < ix_end && (result_correct == 1); ix++ )
        {
            if ( fabs ( Aref[iy][ix] - A[iy][ix] ) >= tol )
            {
                fprintf(stderr,"[MPI%d] ERROR: A[%d][%d] = %f does not match %f (reference)\n", rank, iy,ix, A[iy][ix], Aref[iy][ix]);
                result_correct = 0;
            }
        }
    }
#ifdef MPI_VERSION
    int global_result_correct = 0;
    MPI_Allreduce( &result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
    result_correct = global_result_correct;
#endif //MPI_VERSION
    return result_correct;
}

#endif // LAPLACE2D_SERIAL_H
