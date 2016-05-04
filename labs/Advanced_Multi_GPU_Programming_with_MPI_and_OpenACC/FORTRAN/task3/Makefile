all: run

laplace2d: laplace2d.F03 laplace2d_serial.F03 Makefile
	mpifort -fast -acc -ta=tesla -Minfo=accel laplace2d.F03 laplace2d_serial.F03 -o laplace2d

clean:
	rm -f laplace2d laplace2d.solution *.o laplace2d.*.nvprof laplace2d.solution.*.nvprof

run: laplace2d
	mpirun -np 4 ./laplace2d

profile: laplace2d
	mpirun -np 4 nvprof --process-name "MPI %q{OMPI_COMM_WORLD_RANK}" --context-name "MPI %q{OMPI_COMM_WORLD_RANK}" -o laplace2d.%q{OMPI_COMM_WORLD_RANK}.nvprof ./laplace2d

laplace2d.solution: laplace2d.solution.F03 laplace2d_serial.F03 Makefile
	mpifort -fast -acc -ta=tesla -Minfo=accel laplace2d.solution.F03 laplace2d_serial.F03 -o laplace2d.solution

solution: laplace2d.solution
	mpirun -np 4 ./laplace2d.solution

profile.solution: laplace2d.solution
	mpirun -np 4 nvprof --process-name "MPI %q{OMPI_COMM_WORLD_RANK}" --context-name "MPI %q{OMPI_COMM_WORLD_RANK}" -o laplace2d.%q{OMPI_COMM_WORLD_RANK}.solution.nvprof ./laplace2d.solution
