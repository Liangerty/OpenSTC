OpenSTC
======

**OpenSTC** is a CFD program written in C++ and CUDA. The program tends to provide an all-in-one CFD simulation tool for Supersonic Turbulent Combustion simulations, where the *all-in-one* means the application levels can vary from RANS to DNS. The target users can be industrial engineers (RANS) as well as scientific researchers (LES and DNS).

Comments
--------------------

The final direction of developing this program is stated above, but the current ability is far from completeness (doge). The situation is more like, that I want to have a rocket engine, but what I currently have are just some wood sticks. Let me introduce what the code can do currently.

Current ability
-----------------

The code is based on finite difference method and structured grid. The Navier-Stokes equations with species conservation equations are solved on curvilinear coordinates.

- Turbulent simulations: Laminar / RANS
- RANS model: k-$\omega$ SST
- Mixture model: Air(single species) / Mixture(thermally perfect gas given in CHEMKIN form)
- Combustion model: Finite rate chemistry
- Boundary conditions: Supersonic inlet / Supersonic outlet / No-slip wall / Symmetry

Numerical methods:

- Inviscid flux: AUSM+
- Viscous flux: 2nd-order central difference
- Reconstruction method: MUSCL / NND-2
- Temporal scheme: Explicit Euler / DPLUR
- Stiff chemistry: Point implicit / Diagonal approximation

About the interface:

- Grid: Plot3D with a little variation (The code to generate readable grid for this program can be acquired by emailing me. And the interface would be generalized later to avoid such transforming requirements.)
- Output: Tecplot file in `.plt` format with MPI-IO.
- Parallelization: MPI support(a CUDA-aware MPI is required).
- Continue from a previous simulation: The program can be started from a previous result if the grid is not changed and the flowfield info is given. We can continue solving with a completely identical setup, or with a changed setup. For example, we can continue computing from a simulation with fewer species to a simulation with more species, from laminar to turbulent state, from pure mixing case to combustion case, etc.
- Chemistry: CHEMKIN format
- Setups: Txt files to be edited.

Functions to be implemented in the near future
-----------------

- Unsteady simulation ability with SSPRK.
- High order inviscid discretization methods such as WENO/TENO, hybrid method, etc.
- More boundary conditions such as farfield, inlet profile, periodic, etc.
- Turbulent inflow generation method such as synthetic turbulence generator, turbulence library, etc.
- LES and DNS turbulent statistics.

Compile OpenSTC
-----------------

OpenSTC requires

1. a C++20 compiler
2. a CUDA compiler, CUDA toolkit version should be larger than 11.0
3. a CUDA-aware MPI implementation.

Currently, OpenSTC has been successfully compiled on Linux system with the following configuration: gcc 11.1 + CUDA 11.8 + OpenMPI 4.1.1.

> I have not tested OpenSTC on Windows system because I have not dedicated to find a CUDA-aware MPI on Windows. If we just use the code to run serially, any MPI (such as MSMPI) on Windows can also work fine (VS2022 + CUDA 12.0 + MSMPI). But once parallel is a necessity, CUDA-aware MPI would be required. And I have not found such a CUDA-aware one on Windows.

Besides, a formatting library 'fmtlib' is also required. I have included such a copy in the `depends` directory. There is a possibility that the '.a' file could not be linked because the GLIBC-version that the `.a` file uses does not match. If such thing happens, you would need to download the 'fmtlib' and compile it by yourself, after which you copy the `.a` file into `depends/lib/release` library.

After we prepare these files, we can compile the program with CMake as follows:

1. We open the bash in the current directory, because the `CMakeLists.txt` is there.
2. Create the build directory and find the compiler info:

    ```bash
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    ```

3. Compile and link to get the executable file

   ```bash
   cmake --build build --parallel 4
   ```

   The number after `--parallel` is the number of threads that to be used when compiling the codes.

After these operation, the executable `openstc` should be in the current directory.

Hey
-----------------

Let me argue for another time that the code is still under development and the range of application is confined now.

But I do hope you can enjoy using it and make it your major tool in researching or engineering (to make more money).

Therefore, if you have any question about how to use it or any suggestions on better development, or even more functions to be achieved, please let me know by emailing me at 16051068@buaa.edu.cn. Once I see the email, I would reply you with solutions or 'sorry' (for some things I cannot solve immediately).

I'm not an expert in CFD or STC simulation, just a Ph.D. candidate who wants to learn and make progress.

I'd be really appreciate if you can use my program and supply suggestions for me.
