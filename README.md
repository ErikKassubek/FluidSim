# Fluid simulation
Implementation of an lattice Bolzman fluid simulation.<br>

## How
### Serial
Run the serial simulation with
```
cd src/
python main.py [args]
```

with the following possible args:

- -h: show help
- -m2: run the collision operator simulation
- -m3d: run the shear wave decay simulation with sin in density
- -m3v: run the shear wave decay simulation with sin in velocity
- -m4: run the couette flow simulation
- -m5: run the poiseuille flow simulation
- -m6s: run the serial sliding lid simulation
- -x 'len_x': set the length of the lattice to 'len_x'. Default: 100
- -y 'len_y': set the height of the lattice to 'len_y'. Default: 100
- -s 'steps': set the number of simulation steps to 'steps'. Default: 1000
- -o 'omega': set omega. Default 1
- -d: create a density plot
- -f: create a flow plot
- -g: create a gif
- -c: save the values of a cut throw the lattice. The cut position is defined by the simulations.
- -a: save the values of a point of the lattice. The point position is defined by the simulations.
- -v: show the current time step during the simulation
- -t: measure the time of the simulation

Info:

- At least one from -m2 -m3d -m3v -m4 -m5 or -m6s must be set
- At least one from -d, -f, or -c must be set
- If -g is set at least one of -d or -f must be set
- steps $>$ 0
- len_x $\geq$ 10, len_y $\geq$ 10
- 0 $<$ omega $<$ 2

### Parallel
Run the parallel sliding lid with

```
cd src/
mpiexec -n [no. of processes] python main.py -m6p [args]
```

with the following possible args:

- -x 'len_x': set the length of the lattice to 'len_x'. Default: 100
- -y 'len_y': set the height of the lattice to 'len_y'. Default: 100
- -s 'steps': set the number of simulation steps to 'steps'. Default: 1000
- -o 'omega': set omega. Default 1
- -v: show the current time step during the simulation
- -t: measure the time of the simulation

Info:

- steps $> 0$
- len_x $\geq$ 10, len_y $\geq$ 10
- 0 $<$ omega $<$ 2
- Avoid prime numbers for no. of processes


## Example:
```
python main.py -m6s -s 900 -x 90 -y 90 -f -g -v
```
will run the sliding lid simulation for 900 time steps on a lattes with
90x90 points. It will create the images for the flow as well as a gif of
the flow:
<p align="middle">
    <img src="./img/sliding_lid.gif" width="60%" />
</p>

```
mpiexec -n 8 python main.py -m6p -s 900 -x 90 -y 90 -t
```
will run the same simulation in parallel with 8 processes.