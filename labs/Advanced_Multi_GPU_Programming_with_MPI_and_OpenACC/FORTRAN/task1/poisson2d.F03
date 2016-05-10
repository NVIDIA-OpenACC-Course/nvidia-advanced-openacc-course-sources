!  Copyright 2015 NVIDIA Corporation
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.

PROGRAM poisson2d
    USE openacc
    USE mpi
    IMPLICIT NONE
    INTEGER, PARAMETER :: nx = 4096
    INTEGER, PARAMETER :: ny = 4096
    INTEGER, PARAMETER :: iter_max = 1000
    REAL, PARAMETER :: tol = 1.0E-5
    INTEGER :: ix, iy, ix_start, ix_end, iy_start, iy_end, iter, mpi_rank, mpi_size, device_type, ngpus, devicenum, ierror, chunk_size, right, left
    REAL :: x,y, error, runtime_serial, runtime, start, finish, globalerror
    LOGICAL, EXTERNAL :: check_results
    REAL, DIMENSION(nx,ny) :: a, a_ref, a_new, rhs
    
    mpi_rank = 0
    mpi_size = 1
    
    !Initialize MPI and determine rank and size
    CALL MPI_Init(ierror)
    CALL MPI_Comm_rank(MPI_COMM_WORLD,mpi_rank,ierror)
    CALL MPI_Comm_size(MPI_COMM_WORLD,mpi_size,ierror)
    
    a = 0.0
    a_ref = 0.0
    
    DO iy = 2, ny-1
        DO ix = 2, nx-1
            x = -1.0 + (2.0*ix/(nx-1.0))
            y = -1.0 + (2.0*iy/(ny-1.0))
            rhs(ix,iy) = EXP(-10.0*(x*x+y*y))
        END DO
    END DO
    
#if _OPENACC
    device_type = acc_get_device_type()
    IF ( acc_device_nvidia == device_type ) THEN
        ngpus=acc_get_num_devices( acc_device_nvidia )
        !choose device to use by this rank
        devicenum = MOD( mpi_rank, ngpus )
        call acc_set_device_num( devicenum, acc_device_nvidia )
    END IF
    !Call acc_init after acc_set_device_num to avoid multiple contexts on device 0 in multi GPU systems
    call acc_init( device_type )
#endif
   
    ix_start = 2
    ix_end   = nx-1
    
    !set first and last row to be processed by this rank.
    !Ensure correctness if ny%size != 0
    chunk_size = CEILING( (1.0*ny)/mpi_size )
    iy_start = mpi_rank * chunk_size
    iy_end = iy_start + chunk_size - 1
    
    !Do not process boundaries
    iy_start = MAX( iy_start, 2 )
    iy_end = MIN( iy_end, ny-1 )

    IF ( mpi_rank == 0 ) THEN
        WRITE(*,"('Jacobi relaxation Calculation: ',I4,' x ',I4,' mesh')") nx,ny
        WRITE(*,*) 'Calculate reference solution and time serial execution.'
    END IF
    CALL cpu_time(start)
    CALL poisson2d_serial( nx, ny, iter_max, mpi_rank, tol, a_ref, a_new, rhs )
    CALL cpu_time(finish)
    runtime_serial = finish-start
    
    !Wait for all processes to ensure correct timing of the parallel version
    CALL MPI_Barrier( MPI_COMM_WORLD, ierror )
    
    IF ( mpi_rank == 0 ) THEN
        WRITE(*,*) 'Parallel execution.'
    END IF 
    
    CALL cpu_time(start)
    iter = 1
    error = 1.0
    !$acc data copy(a) copyin(rhs) create(a_new)
    DO WHILE ( error > tol .AND. iter <= iter_max )
        error = 0.0
        !$acc kernels
        DO iy = iy_start, iy_end
            DO ix = ix_start, ix_end
                a_new(ix,iy) = -0.25 * (rhs(ix,iy) - ( a(ix+1,iy) + a(ix-1,iy) + a(ix,iy-1) + a(ix,iy+1) ))
                error = MAX( error, ABS( a_new(ix,iy) - a(ix,iy) ) )
            END DO
        END DO
        !$acc end kernels
        !Calculate global error across all ranks
        globalerror = 0.0
        call MPI_Allreduce( error, globalerror, 1, MPI_REAL, MPI_MAX, MPI_COMM_WORLD, ierror )
        error = globalerror
        
        !TODO: Split into halo and bulk part
        !$acc kernels
        DO iy = iy_start, iy_end
            DO ix = ix_start, ix_end
                a(ix,iy) = a_new(ix,iy)
            END DO
        END DO
        !$acc end kernels
        !TODO: Start bulk part asynchronously
        
        !Handle periodic boundary conditions and halo exchange with MPI
        left = mpi_rank-1
        IF ( mpi_rank == 0 ) THEN
            left = mpi_size-1
        END IF
        right = mpi_rank+1
        IF ( mpi_rank == mpi_size-1 ) THEN
            right = 0
        END IF
        
        !$acc host_data use_device( A )
            !1. Sent column iy_start (first modified column) to left receive right boundary (iy_end+1) from right
            CALL MPI_Sendrecv( a(ix_start,iy_start), (ix_end-ix_start)+1, MPI_REAL, left   , 0, a(ix_start,iy_end+1), (ix_end-ix_start)+1, MPI_REAL, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )

            !2. Sent column iy_end (last modified column) to right receive left boundary (iy_start-1) from left
            CALL MPI_Sendrecv( a(ix_start,iy_end), (ix_end-ix_start)+1, MPI_REAL, right, 0, a(ix_start,(iy_start-1)), (ix_end-ix_start)+1, MPI_REAL, left   , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror )
        !$acc end host_data
        
        !TODO: wait for bulk part
        
        !$acc kernels
        DO iy = iy_start, iy_end
            a(1,iy) = a(nx-1,iy)
            a(nx,iy) = a(2,iy)
        END DO
        !$acc end kernels

        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F8.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc end data
    !Wait for all processes to ensure correct timing of the parallel version
    CALL MPI_Barrier( MPI_COMM_WORLD, ierror )
    CALL cpu_time(finish)
    runtime = finish-start
    
    IF ( check_results( mpi_rank, ix_start, ix_end, iy_start, iy_end, nx, ny, tol, a, a_ref ) ) THEN
        IF ( mpi_rank == 0 ) THEN
            WRITE(*,*) 'Num GPUs: ', mpi_size
            WRITE(*,"(I4,'x',I4,': 1 GPU: ',F8.4,' s ',I1,' GPUs: ',F8.4,' s, speedup: ',F8.2,' efficiency: ',F8.2)"),nx,ny,runtime_serial,mpi_size,runtime,runtime_serial/runtime,runtime_serial/(mpi_size*runtime)*100
        END IF
    END IF
    !Finalize MPI
    CALL MPI_Finalize(ierror)
END PROGRAM poisson2d
