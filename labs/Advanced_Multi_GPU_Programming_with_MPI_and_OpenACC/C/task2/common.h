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

#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>

#ifdef USE_DOUBLE
    typedef double real;
    #define fmaxr fmax
    #define fabsr fabs
    #define expr exp
    #define MPI_REAL_TYPE MPI_DOUBLE
#else
    typedef float real;
    #define fmaxr fmaxf
    #define fabsr fabsf
    #define expr expf
    #define MPI_REAL_TYPE MPI_FLOAT
#endif

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef WIN32
double PCFreq = 0.0;
__int64 timerStart = 0;
#else
struct timeval timerStart;
#endif

typedef struct
{
    int y;
    int x;
} dim2;

#define MAX_MPI_SIZE 16

static dim2 size_to_size2d_map[MAX_MPI_SIZE+1] = { {0,0},
    {1,1}, {2,1}, {3,1}, {2,2},
    {5,1}, {3,2}, {7,1}, {4,2},
    {3,3}, {5,2}, {11,1}, {6,2},
    {13,1}, {7,2}, {5,3}, {4,4}
};

void StartTimer()
{
#ifdef WIN32
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
        printf("QueryPerformanceFrequency failed!\n");

    PCFreq = (double)li.QuadPart/1000.0;

    QueryPerformanceCounter(&li);
    timerStart = li.QuadPart;
#else
    gettimeofday(&timerStart, NULL);
#endif
}

// time elapsed in ms
double GetTimer()
{
#ifdef WIN32
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart-timerStart)/PCFreq;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
}

int min( int a, int b)
{
    return a < b ? a : b;
}

int max( int a, int b)
{
    return a > b ? a : b;
}

void laplace2d_serial( int rank, int iter_max, real tol );

int check_results( int rank, int ix_start, int ix_end,  int iy_start, int iy_end, real tol );

dim2 size_to_2Dsize( int size )
{
    assert(size<=MAX_MPI_SIZE);
    return size_to_size2d_map[size];
}

#endif // COMMON_H
