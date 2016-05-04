COMPILER ?= PGI-tesla
CC = mpicc
CFLAGS = -DUSE_DOUBLE
ifeq ($(COMPILER),GCC)
	CFLAGS += -march=native -O3 -lm
else ifeq ($(COMPILER),PGI-tesla)
	CFLAGS += -Minfo=accel -fast -acc -ta=tesla
else ifeq ($(COMPILER),PGI-multicore)
	CFLAGS += -Minfo=accel -fast -acc -ta=multicore
endif
all: run

laplace2d: laplace2d.c common.h laplace2d_serial.h Makefile
	$(CC) $(CFLAGS) laplace2d.c -o laplace2d

clean:
	rm -f laplace2d laplace2d.solution laplace2d.*.nvprof laplace2d.solution.*.nvprof

run: laplace2d
	mpirun -np 4 ./laplace2d

profile: laplace2d
	mpirun -np 4 nvprof --process-name "MPI %q{OMPI_COMM_WORLD_RANK}" --context-name "MPI %q{OMPI_COMM_WORLD_RANK}" -o laplace2d.%q{OMPI_COMM_WORLD_RANK}.nvprof ./laplace2d

laplace2d.solution: laplace2d.solution.c common.h laplace2d_serial.h Makefile
	$(CC) $(CFLAGS) laplace2d.solution.c -o laplace2d.solution

solution: laplace2d.solution
	mpirun -np 4 ./laplace2d.solution

profile.solution: laplace2d.solution
	mpirun -np 4 nvprof --process-name "MPI %q{OMPI_COMM_WORLD_RANK}" --context-name "MPI %q{OMPI_COMM_WORLD_RANK}" -o laplace2d.solution.%q{OMPI_COMM_WORLD_RANK}.nvprof ./laplace2d.solution
