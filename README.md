# Mordor

<img src="https://github.com/thanatom/mordor/blob/main/img/bd.jpg" width=10% height=10%> 

`Mordor` (MORphological DecOmposeR) is a `Python` tool to perform a morphology decomposition based on stellar kinematics of simulated galaxies, read through `Pynbody`.

<p align="center">
  <img src="https://github.com/thanatom/mordor/blob/main/img/components.png" width=25% height=25%>
</p>

> NB: The current version of the package does not include the bar identification algorithm. If you are interested, please contact me at tommaso.zana at gmail.com

This version contains slight improvments to the circularity calculation with respect to Zana et al. 2022.

## Dependencies

`Mordor` requires the following packages:

- `Pynbody` (https://github.com/pynbody/pynbody)
- `Numpy` (https://numpy.org/)
- `Scipy` (https://scipy.org/)
- `Numba` (https://numba.pydata.org/)
- `Matplotlib` (https://matplotlib.org/)

## Quick Start

`Mordor` is really simple to get started with. From command line type

```py
python mordor.py SrcName
```
if **SrcName** is a snapshot containing the galaxy to decompose, which should be named 'PREFIX_ID.EXTENSION' (e.g. Gal_000000.hdf5), or

```py
python mordor.py SrcName -l
```

if **SrcName** is an ASCII list of snapshot names, like

PREFIX_ID0.EXTENSION
PREFIX_ID1.EXTENSION
...
PREFIX_IDN.EXTENSION

Type

```py
python mordor.py -h
```

for a quick description of the usage.

## Arguments

Positional arguments:

**SrcName** - particle file or list to process. Particle files should be named 'PREFIX_ID.EXTENSION', e.g. 'Gal_000000.hdf5'

Options:

|     **Command**     |                    **Options [default]**                    |                                           **Description**                                          |
|:-------------------:|:-----------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|
| **-l, -list**       | Boolean [*False*]                                           | source file **SrcName** is interpreted as a list of files                                          |
| **-m, --mode**      | *direct*, *pm*, [*tree*], *cosmo_sim*, *iso_sim*, *auxiliary* | gravitational potential computation mode                                                           |
| **--ShowPlots**     | Boolean [*False*]                                           | show the resulting potential profile and the faceon galaxy-map divided in morphological components |
| **--DumpPotential** | Boolean [*False*]                                           | dump a file 'potential_ID.npy' with the gravitational potential of the galaxy ID                   |
| **--LoadOff**       | *ascii*, *bin*, [*None*]                                    | read from the file 'offsets' position and velocity of the galaxy centre                   |
| **--DumpOff**       | *ascii*, *bin*, [*None*]                                    | print the file 'offsets' with position and velocity of the galaxy centre                                       |
| **--OutPrefix**     | String [morphology]                                         | if a source list is given, the output is redirected to the file 'OUTPREFIX_SrcName'                |

## Details

`Mordor` can operate on any snapshot supported by `Pynbody`.
After the gravitational potential energy is [evaluated](#potential-evaluation), the galaxy is centred and rotated, assuming it's the unique or most massive object in the snapshot.
The procedure keeps into account that, sometimes, halo-finder algorithms may fail to associate the dark-matter component to satellite galaxies.

> Make sure the galaxy is not split across the boundaries of the simulation box with periodic conditions.

The actual decomposition is performed by the function

```py
 morph(gal, j_circ_from_r=False, LogInterp=False, BoundOnly=False, Ecut=None, jThinMin=0.7, mode='tree', theta=0.5, dimcell='1 kpc', DumpProb=False)
```

where the arguments and parameters are

**gal** - Sim-object. The galaxy to analyse.

**Ecut** - energy boundary between bulge/pseudobulge and halo/thick disc. If set to 'None', Ecut is evaluated as the smallest local minimum of the particle energy distribution.

**j_circ_from_r** - circular angular momentum is computed as a function of radius, rather than as a function of orbital energy.

**LogInterp** - use a logarithmic interpolation/extrapolation, instead of a linear one, to evaluate the circular angular momentum.

**BoundOnly** - exclude particles with $E\geq0$, $\lvert\eta\rvert\geq1.5$, or $\lvert j_{p}/j_{\rm circ}\rvert\geq1.5$.

**mode** - choose amongst *direct*, *pm*, *tree*, *cosmo_sim*, *iso_sim*, or *auxiliary*. If mode is *cosmo_sim*, an offset is applied (see Zana et al. 2022 for details).

**DumpProb** - particles falling in overlapping regions of the circularity distribution are assigned to the spheroidal components (bulge or halo) according to a probabilistic scheme. If DumpProb is enabled, an additional SimArray is created and filled with a float for each stellar particle, where the integer part refers to the alternative morphological component (not assigned) and the decimal part to the probability of assignment. if **prob**$=0$, the particle has been assigned to the only possible component.

**j_disc_min** - minimum angular momentum (in terms of the circular angular momentum) that a particle must have to be part of the 'thin disc' component. Default is 0.7.

**theta** - opening angle for gravitational force calculation when mode is set on 'tree'. It tunes the accuracy of the force evaluation. Default is 0.5.

**dimcell** - size of the grid spacing when mode is set on *pm*. Default is 1 kpc.

This function produces a SimArray, named **morph**, where an integer from 0 to 5 is assigned to each star particle, to identify a morpho-kinematic component:

	0 -- unbound/excluded
	1 -- thin/cold disc
	2 -- thick/warm disc
	3 -- pseudo-bulge
	4 -- bulge
	5 -- stellar halo   

Stellar circularities are computed by re-evaluating the gravitational potential on the midplane. Argument **mode** controls the computation strategy.
The default mode is set to be the same adopted to compute the potential for the whole galactic particles, when possible.
Otherwise, Brute-force computation is performed.

> NB: gravitational potentials evaluated with different methods may differ by a constant. If this is not handled correctly, the whole decomposition is undermined.

Some additional parameters, necessary to find **Ecut**, can be modified/tuned within the function **morph()** in *decomposition.py*.

If **BounOnly** is set to *False* no mass element is excluded from the computation.

> NB: If **mode** = *pm*, non-excluded particles may force the construction of a much larger grid, possibly resulting in a memory overflow.

### Softening

`Mordor` includes a definition for the gravitational softening length, which important for the entire decomposition procedure.
The current definition applies to all kind of particles in the TNG50 simulation.

> If you're using a different relation for the softening length, please update the function **gsoft()** in *mordor.py*

### Disc definition

Galaxies are identified as disc galaxies when $M_{\rm thin}+M_{\rm thick}+M_{\rm pseudo-bulge}>0.5M_{*}$

> Different definitions can be adopted in the function **isDisc()** in *mordor.py*.

### Potential evaluation

The evaluation of the gravitational potential energy is the computational bottleneck for kinematic decompositions.
`Mordor` provides several alternative ways to do it, which can be enabled through the optional argument *--mode*.
The options are:

| Type                    | Keyword        | Description                                                                                                                                                                                                                                                                                                                                              |
|-------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Direct summation        | direct         | the potential is recomputed via direct summation. This is the most CPU-expensive and accurate way available.                                                                                                                                                                                                                                             |
| Particle-mesh           | pm             | the potential is computed via the FFT of the particle density field on a Cartesian uniform grid. The method is based on a `Pynbody` routine. Faster than the direct mode but memory-expensive. It may fail to construct the grid when **BoundOnly**=False                                                                                                 |
| KD-tree                 | tree [default] | the potential is recomputed via a KD-tree. It is based on a suitably modified version of `Pytreegrav` (https://github.com/mikegrudic/pytreegrav)  and gives very accurate results at a reasonable computational cost.                                                                                                                                    |
| Cosmological simulation | cosmo_sim      | the potential is directly loaded from the snapshot, which has to belong to a cosmological simulation. When the midplane potential is recalculated  "in isolation", an offset is automatically applied to account for the large-scale matter distribution.                                                                                                 |
| Isolated simulation     | iso_sim        | the potential is directly loaded from the snapshot, but it is assumed to come from an isolated simulation. The mode can even be exploited by  recomputing the potential of a galaxy that formed in a fully cosmological simulation, as it is in isolation. This approach would take advantage of the strongly parallelised nature of modern cosmo-codes. |
| Auxiliary file          | auxiliary      | the potential is loaded from an auxiliary file named potential_ID.npy in km$^2$ s$^{-2}$, ordered according to the particle ids. Auxiliary files can be created by `Mordor` itself though the option **--DumpPotential**, to speed up future calculations.                                                                                                       |

## Examples

```py
python mordor.py gal_00000.hdf5 --mode cosmo_sim --ShowPlots --DumpOff bin
```

Here the snapshot *gal_00000.hdf5* is decomposed and its gravitational potential is directly loaded from file (**--mode** *cosmo_sim*), assuming it comes from a cosmological simulation.
An offset is applied later, when the potential in the midplane is evaluated.
A quick plot of the gravitational potential and of the different components identified for the galaxy is shown (--ShowPlots).
A native binary file named 'offsets' is created, containing the galaxy ID (00000), the centre of the galaxy in the original reference frame, and the velocity offset.
The output, with the component masses and the mean energies and circularities is printed on screen.

```py
python mordor.py galaxy_list -l --mode direct --OutPrefix out --DumpPot
```

All the galaxies in the list 'galaxy_list.txt' are decomposed and the output is printed in the text file 'out_galaxy_list'.
The gravitational potential is calculated for every galaxy through direct summation and is saved in the files 'potential_ID0.npy', 'potential_ID1.npy', ..., 'potential_IDN.npy'. 

## Credit and support

If you use Mordor in preparing a scientific publication, please cite the following BibTex entry:

	@ARTICLE{mordor,
		author = {{Zana}, Tommaso and {Lupi}, Alessandro and {Bonetti}, Matteo and {Dotti}, Massimo and {Rosas-Guevara}, Yetli and {Izquierdo-Villalba}, 	David and {Bonoli}, Silvia and {Hernquist}, Lars and {Nelson}, Dylan},
		title = "{Morphological decomposition of TNG50 galaxies: methodology and catalogue}",
		journal = {arXiv e-prints},
		keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
		year = 2022,
		month = jun,
		eid = {arXiv:2206.04693},
		pages = {arXiv:2206.04693},
		archivePrefix = {arXiv},
		eprint = {2206.04693},
		primaryClass = {astro-ph.GA},
		adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220604693Z},
		adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}

Please contact me (tommaso.zana at gmail.com) if you have question/troubles with `Mordor`.


