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

SUBROUTINE laplace2d_serial(nx, ny, iter_max, mpi_rank,tol,a_ref, a_new,rhs)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nx, ny, iter_max, mpi_rank
    REAL, INTENT(IN) :: tol
    REAL, DIMENSION(nx,ny), INTENT(INOUT) :: a_ref, a_new
    REAL, DIMENSION(nx,ny), INTENT(IN) :: rhs
    INTEGER :: ix,iy, iter
    REAL :: error
    
    iter = 1
    error = 1.0
    !$acc data copy(a_ref) copyin(rhs) create(a_new)
    DO WHILE ( error > tol .AND. iter <= iter_max )
        error = 0.0
        !$acc kernels
        DO iy = 2, ny-1
            DO ix = 2, nx-1
                a_new(ix,iy) = -0.25 * (rhs(ix,iy) - ( a_ref(ix+1,iy) + a_ref(ix-1,iy) + a_ref(ix,iy-1) + a_ref(ix,iy+1) ))
                error = MAX( error, ABS( a_new(ix,iy) - a_ref(ix,iy) ) )
            END DO
        END DO
        
        DO iy = 2, ny-1
            DO ix = 2, nx-1
                a_ref(ix,iy) = a_new(ix,iy)
            END DO
        END DO
        
        DO ix = 2, nx-1
            a_ref(ix,1) = a_ref(ix,ny-1)
            a_ref(ix,ny) = a_ref(ix,2)
        END DO
        
        DO iy = 2, ny-1
            a_ref(1,iy) = a_ref(nx-1,iy)
            a_ref(nx,iy) = a_ref(2,iy)
        END DO
        !$acc end kernels
        
        IF ( mpi_rank == 0 .AND. ( iter == 1 .OR. MOD( iter, 100 ) == 0 ) ) THEN
            WRITE(*,"('  ',I4,' ',F8.6)") iter, error
        END IF
        
        iter = iter+1
    END DO
    !$acc end data
END SUBROUTINE laplace2d_serial

LOGICAL FUNCTION check_results( mpi_rank, ix_start, ix_end, iy_start, iy_end, nx, ny, tol, a, a_ref )
    USE mpi
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: mpi_rank, ix_start, ix_end, iy_start, iy_end, nx, ny
    REAL, INTENT(IN) :: tol
    REAL, DIMENSION(nx,ny), INTENT(IN) :: a, a_ref
    INTEGER :: ix,iy,ierror
    LOGICAL :: no_errors, global_no_errors, mpi_is_initialized
    no_errors = .TRUE.
    iy = iy_start
    ix = ix_start
    DO WHILE ( iy <= iy_end .AND. no_errors )
        DO WHILE ( ix <= ix_end .AND. no_errors )
            IF ( ABS( a_ref(ix,iy) - a(ix,iy)) >= tol ) THEN
                WRITE(*,"('[MPI',I1,'] ERROR: a(',I4,',',I4,') = ',F8.6,' does not match ',F8.6,' (reference)')") mpi_rank,ix,iy,a(ix,iy),a_ref(ix,iy)
                no_errors = .FALSE.
            END IF
            ix = ix + 1
        END DO
        iy = iy + 1
    END DO
    
    CALL MPI_Initialized(mpi_is_initialized, ierror)
    IF ( mpi_is_initialized ) THEN
        global_no_errors = .FALSE.
        CALL MPI_ALLREDUCE(no_errors, global_no_errors, 1, MPI_LOGICAL, MPI_LAND, MPI_COMM_WORLD, ierror)
        no_errors = global_no_errors
    END IF
    check_results = no_errors
END FUNCTION check_results
