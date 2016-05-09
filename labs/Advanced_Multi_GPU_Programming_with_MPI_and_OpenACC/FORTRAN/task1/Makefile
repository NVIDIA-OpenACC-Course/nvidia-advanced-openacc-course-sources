all: run

poisson2d: poisson2d.F03 poisson2d_serial.F03 Makefile
	mpifort -fast -acc -ta=tesla -Minfo=accel poisson2d.F03 poisson2d_serial.F03 -o poisson2d

clean:
	rm -f poisson2d poisson2d.solution *.o poisson2d.*.nvprof poisson2d.solution.*.nvprof

run: poisson2d
	mpirun -np 4 ./poisson2d

profile: poisson2d
	mpirun -np 4 nvprof --process-name "MPI %q{OMPI_COMM_WORLD_RANK}" --context-name "MPI %q{OMPI_COMM_WORLD_RANK}" -o poisson2d.%q{OMPI_COMM_WORLD_RANK}.nvprof ./poisson2d

poisson2d.solution: poisson2d.solution.F03 poisson2d_serial.F03 Makefile
	mpifort -fast -acc -ta=tesla -Minfo=accel poisson2d.solution.F03 poisson2d_serial.F03 -o poisson2d.solution

solution: poisson2d.solution
	mpirun -np 4 ./poisson2d.solution

profile.solution: poisson2d.solution
	mpirun -np 4 nvprof --process-name "MPI %q{OMPI_COMM_WORLD_RANK}" --context-name "MPI %q{OMPI_COMM_WORLD_RANK}" -o poisson2d.%q{OMPI_COMM_WORLD_RANK}.solution.nvprof ./poisson2d.solution
