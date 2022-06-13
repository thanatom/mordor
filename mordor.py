import numpy as np
import pynbody
import argparse
import decomposition
import gravity

'''
Mordor (MORphological DecOmposeR) performs a morphology decomposition based on stellar kinematics,
of simulated galaxies, read through pynbody (see Zana et al. 2022 at https://arxiv.org/abs/2206.04693)

Mordor input is a file of a simulated galaxy (all formats readable with pynbody are accepted) or a list of files of simulated galaxies.
The files should be named 'PREFIX_ID.EXTENSION' (e.g. Gal_000000.hdf5).
Each file should contain a single galaxy to avoid possible failures in the centring procedure and reduce the gravitational potential computation. 
The computation of the gravitational potential can be addressed via the following modes:

direct -- direct summation (the most computationally demanding)
pm -- particle mesh. If selected the default cell dimension is set to the simulation resolution (dimcell)
tree -- kdtree (the same method is adopted by 'morph'). The default aperture angle is set to 0.5 (theta)
cosmo_sim -- use the potential energy already present in the snapshot (if present). In cosmological simulations the galaxy is a chunk of
		a larger particle set and when 'cosmo_sim' is selected an offset is taken into account by 'morph'. Check out the potential units as they
		are loaded by pynbody where default is km^2 s^-2.
iso_sim -- use the potential from a snapshot reprocessed by a simulation code (such as Arepo; the snapshot must be given as the only input).
		Differently from 'cosmo_sim', here te simulation code is used to recompute the gravitational potential on the isolated galaxy only and no
		offset is considered by 'morph'. heck out the potential units as they are loaded by pynbody where default is km^2 s^-2.
auxiliary -- read the potentials from an auxiliary file named potential_ID.npy in km^2 s^-2, ordered according to the particle ids.
		Auxiliary files can be printed by Mordor itself (to speed up future calculations)

The outputs are the mass, mean enercy and mean circularity of all the different morpho/kinematical stellar components.
If a list of snapshots is given as input (enable the flag '-l'), an ASCII file is printed.
'''

#--------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(prog='Mordor', 
								description='Mordor (MORphological DecOmposeR) performs a morphology decomposition based on stellar kinematics', 
								usage='%(prog)s SrcName [options]',
								epilog='Please, if you use Mordor to prepare a scientific publication, cite Zana et al. 2022. Enjoy!')

parser.add_argument('SrcName', action='store', nargs=1, help='particle file or list to process.\nParticle files should be named \'PREFIX_ID.EXTENSION\', e.g. \'Gal_000000.hdf5\'')
parser.add_argument('-l', '--list', action='store_true', help='source file \'SrcName\' is interpreted as a list of files')
parser.add_argument('-m', '--mode', action='store', nargs=1, default=['tree'], choices=['direct', 'pm', 'tree', 'cosmo_sim', 'iso_sim', 'auxiliary'], help='gravitational potential computation mode')
parser.add_argument('--ShowPlots', action='store_true', help='show the potential profile and the faceon galaxy-map of components')
parser.add_argument('--DumpPotential', action='store_true', help='dump a file \'potential_ID.npy\' with the gravitational potential of the galaxy ID')
parser.add_argument('--LoadOff', action='store', nargs=1, default=['None'], choices=['ascii', 'bin', 'None'], help='read from the file \'offsets\' the space and velocity offsets to centre the galaxy from file (can be ASCII or binary)')
parser.add_argument('--DumpOff', action='store', nargs=1, default=['None'], choices=['ascii', 'bin', 'None'], help='print a \'offsets\' file with the space and velocity offsets (can be ASCII or binary)')
parser.add_argument('--OutPrefix', action='store', nargs=1, default=['morphology'], metavar='', help='if a source list is given, the output is redirected to the file \'OutPrefix_SrcName\', default is \'morphology\'')

args = parser.parse_args()

SrcName = args.SrcName[0]
isList = args.list 
ShowPlots = args.ShowPlots
DumpPotential = args.DumpPotential
DumpOff = args.DumpOff[0]
LoadOff = args.LoadOff[0]
OutPrefix = args.OutPrefix[0]

if DumpOff == 'None': DumpOff = None
if LoadOff == 'None': LoadOff = None

print('potential mode:', args.mode[0])

#--------------------------------------------------------------------------------------------------------

#Import the remaining packages

if args.mode[0] == 'direct':
	from kdtree import BruteForcePotentialTarget
elif args.mode[0] == 'pm':
	from gravity import pmesh
elif args.mode[0] == 'tree':
	from kdtree import KDPotential

if DumpOff or LoadOff:
	import struct
	off_file_name = 'offsets'

if ShowPlots:
	from matplotlib import pyplot as plt
	from matplotlib import gridspec
	from matplotlib.lines import Line2D

	orange_point = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Particles')
	blue_line = Line2D([0], [0], color='blue', label='Profile - mode: %s' % (args.mode[0]))
	red_line = Line2D([0], [0], color='red', label='Midplane profile')
	thin_p = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Thin disc')
	thick_p = Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Thick disc')
	pbulge_p = Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', label='Pseudo-bulge')
	bulge_p = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Bulge')
	halo_p = Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', label='Halo')

#--------------------------------------------------------------------------------------------------------

def gsoft(z):

	'''
	Compute the softening length (this is for TNG50)
	'''

	if z<=1:
		return 0.288
	elif z>1:
		return (0.288*2)/(1+z)

#-------

def isDisc(Mthin, Mthick, Mpbulge, Mbulge, Mhalo, M_Jcut=0.5):

	'''
	Assess wether a galaxy with Mthin, Mthick, Mpbulge, Mbulge, and Mhalo is a disc

	Arguments:
	Mthin -- Thin/cold disc mass
	Mthick -- Thick/warm disc mass
	Mpbulge -- Pseudo-bulge mass
	Mbulge -- Bulge mass
	Mhalo -- Stellar halo mass
	M_Jcut -- Mass Threshold to define a galaxy as a disc galaxy. Default is 0.5

	'''
	return (Mthin+Mthick+Mpbulge)/(Mthin+Mthick+Mpbulge+Mbulge+Mhalo)>M_Jcut
	#Alternative identification adopted in Zana et al. 2022
	#return (Mthin+Mthick)/(Mbulge+Mpbulge)>M_Jcut

#--------------------------------------------------------------------------------------------------------

def get_hmr(f):
	prof = pynbody.analysis.profile.Profile(gal, ndim=3, type='log')
	return np.min(prof['rbins'][prof['mass_enc']>0.5*prof['mass_enc'][-1]])

#-------

if isList:
	with open(SrcName, 'r') as filelist:
		filenames = filelist.read().splitlines()
	outname = OutPrefix+'_'+SrcName
	outfile = open(outname, 'a')
	outfile.close()
else:
	filenames = [SrcName]

#-------

if DumpOff == 'bin':
	import struct
	off = open(off_file_name, 'wb')
	off.close()
elif DumpOff == 'ascii':
	off = open(off_file_name, 'w')
	off.close()

#-------

if LoadOff == 'bin':
	struct_len = struct.calcsize('@idddddd')
	OffList = []
	with open(off_file_name, 'rb') as off:
		while True:
			data = off.read(struct_len)
			if not data: break 
			idd, xoff, yoff, zoff, vxoff, vyoff, vzoff = struct.Struct('@idddddd').unpack_from(data)
			OffList.append([idd, xoff, yoff, zoff, vxoff, vyoff, vzoff])
	OffList = np.array(OffList)
elif LoadOff == 'ascii':
	OffList = np.atleast_2d(np.genfromtxt(off_file_name))

#--------------------------------------------------------------------------------------------------------

#Read the snapshots
for filename in filenames:

	mode = args.mode[0]

	print('Decomposing galaxy %s ...\n' % (filename))

	gal = pynbody.load(filename)

#--------------------------------------------------------------------------------------------------------

	#Start the analysis
	gal.physical_units()
	#Define the softening as the Plummer-equivalent radius if not present
	if 'eps' not in gal:
		gal['eps'] = pynbody.array.SimArray(2.8*gsoft(gal.properties['z'])*np.ones_like(gal['x'], dtype=gal['x'].dtype),'kpc')
	print(gal['eps'])

#--------------------------------------------------------------------------------------------------------

	#Evaluate the gravitational potential
	#NB: most of these methods do not work if the galaxy is splitted across the box boundaries with periodic conditions

	#dimcell is re-assigned if mode=='pm'
	dimcell = None
	
	if mode == 'direct':
		gal['phi'] = BruteForcePotentialTarget(gal['pos'], gal['pos'], gal['mass'], gal['eps'])
		#Units
		gal['phi'].units = pynbody.units.G * gal['mass'].units / gal['pos'].units
		gal['phi'].convert_units('km^2 s^-2') 

	elif mode == 'pm':
		#Size of the cell
		dimcell = min(gal['eps'])

		gal['phi'] = pmesh(gal, dimcell)
		#Units
		gal['phi'].units = pynbody.units.G * gal['mass'].units / gal['pos'].units
		gal['phi'].convert_units('km^2 s^-2') 

	elif mode == 'tree':
		#Amplitude of the tree opening-angle
		theta=0.5

		#Add a minimum noise only to those particles which happen to be in the same position to avoid computation issues
		if len(np.unique(gal['pos'][:,0])) < len(gal['pos']):
			a = np.sort(gal['x'])
			for i in range(np.size(a)-1):
				if a[i]==a[i+1]:
					gal['x'][np.where(gal['x']==a[i])[0][0]] += a[i]*1e-5

		#Units
		gal['phi'] = KDPotential(gal['pos'], gal['mass'], gal['eps'], theta)
		gal['phi'].convert_units('km^2 s^-2') 

	elif mode == 'cosmo_sim' or mode == 'iso_sim':
		#Conversion rule from the units employed in the simulation code and pynbody
		#Arepo code: 1/a converts the potential in physical units, the other 1/a accounts for the velocity units km/s * sqrt(a) 
		gal['phi'] /= gal.properties['a']**2

	elif mode == 'auxiliary':
		try:
			aux_name = 'potential_'+filename.split('_')[1].split('.')[0]+'.npy'
			with open(aux_name, 'rb') as sp:
				phi = np.load(sp)
				try:
					mode = np.load(sp)
					print("Potential mode found:", mode)
				except:
					print("No potential mode found in", aux_name)
					mode = 'iso_sim'
					print("Default mode:", mode)

				ids = np.argsort(gal['iord'])
				temp_phi = np.zeros_like(phi)
				temp_phi[ids] = phi

				gal['phi'] = pynbody.array.SimArray(temp_phi, 'km^2 s^-2')
				
		except IOError:
			print('Auxiliary potential file', aux_name, 'not founs\n')
			quit()

#--------------------------------------------------------------------------------------------------------

	#Print the auxiliary file
	if DumpPotential:
		dump_name = 'potential_'+filename.split('_')[1].split('.')[0]+'.npy'
		with open(dump_name, 'wb') as sp:
			ids = np.argsort(gal['iord'])
			np.save(sp, gal['phi'][ids].in_units('km^2 s^-2'))
			np.save(sp, mode)

#--------------------------------------------------------------------------------------------------------

	if DumpOff:
		xa = gal['pos'][0]*1.0
		va = gal['vel'][0]*1.0

#--------------------------------------------------------------------------------------------------------

	#Centring the galaxy
	if LoadOff:
		if any(char.isdigit() for char in filename.split('_')[1]):
			dump_id = int(filename.split('_')[1].split('.')[0])
		else:
			dump_id = int(filename.split('_')[2].split('.')[0])

		#Looking for the galaxy in DumpList
		try:
			sel = np.where(OffList[:,0]==dump_id)[0][0]
			FOUND=True
			gal['pos'] -= OffList[sel][1:4]
			gal['vel'] -= OffList[sel][4:7]

			hmr = get_hmr(gal.s)
		except:
			print("Galaxy \'%s\' not found in offset-list \'%s\'\nCentre will be recalculated...\n" % (filename, off_file_name))
			FOUND=False

	if not (LoadOff and FOUND):
		pynbody.analysis.halo.center(gal, wrap=True, mode='hyb')

		#Gettin the half-mass radius of stars
		hmr = get_hmr(gal.s)
		#Check if the galaxy has a strange shape through the distance of the centre of mass of the stars
		sc = pynbody.analysis.halo.center(gal.s, retcen=True, mode='hyb')
		if np.sqrt(np.sum(sc*sc)) > max(0.5*hmr, 2.8*gsoft(gal.properties['z'])):
			#Re-centre on stars
			try:
				pynbody.analysis.halo.center(gal.s, mode='hyb')
			except:
				pynbody.analysis.halo.center(gal.s, mode='hyb', cen_size='3 kpc')
			hmr = get_hmr(gal.s)

	#Region where to compute the angular momentum to align the galaxy
	size_ang = max(3*hmr, 2.8*gsoft(gal.properties['z']))

#--------------------------------------------------------------------------------------------------------

	#Dump the offsets
	if DumpOff:
		xb = gal['pos'][0]*1.0
		vb = gal['vel'][0]*1.0

		if any(char.isdigit() for char in filename.split('_')[1]):
			dump_id = int(filename.split('_')[1].split('.')[0])
		else:
			dump_id = int(filename.split('_')[2].split('.')[0])

		if DumpOff == 'bin':
			off = open(off_file_name, 'ab')
			off.write(struct.pack('@idddddd', dump_id, xa[0]-xb[0], xa[1]-xb[1], xa[2]-xb[2], va[0]-vb[0], va[1]-vb[1], va[2]-vb[2]))
		elif DumpOff == 'ascii':
			off = open(off_file_name, 'a')
			off.write('%d %lf %lf %lf %lf %lf %lf\n' % (dump_id, xa[0]-xb[0], xa[1]-xb[1], xa[2]-xb[2], va[0]-vb[0], va[1]-vb[1], va[2]-vb[2]))

		off.close()

#--------------------------------------------------------------------------------------------------------
	
	#Rotate the galaxy in order to align itz angular momentum with the z-axis
	pynbody.analysis.angmom.faceon(gal.s, disk_size='%g kpc' % (size_ang), cen=[0,0,0], vcen=[0,0,0])
	try:
		#Launch the decomposition
		#Change 'mode' in decomposition.morph() in order to change potential computation in the midplane
		profiles = decomposition.morph(gal, j_circ_from_r=False, LogInterp=False, BoundOnly=True, Ecut=None, jThinMin=0.7, mode=mode, dimcell=dimcell)
	except Exception as exc:
		print(exc)
		gal.s['morph'][:] = -1
		if 'prob' in gal.s.all_keys():
			gal.s['prob'][:] = -1.

#--------------------------------------------------------------------------------------------------------

	isThin = gal.s['morph']==1
	isThick = gal.s['morph']==2
	isPbulge = gal.s['morph']==3
	isBulge = gal.s['morph']==4
	isHalo = gal.s['morph']==5
	isUnbound = gal.s['morph']==0

#-------

	if ShowPlots:
		#Print some useful information

		axs = (plt.figure(constrained_layout=True).subplots(1, 2))

		axs[0].plot(gal['rxy'], gal['phi'], ',', c='orange', alpha=0.5)
		try:
			axs[0].plot(profiles['rbins'], profiles['phi'], c='blue')
			axs[0].plot(profiles['rbins'], profiles['pot'], c='red')
		except Exception as exc:
			print('No profile has been created')
		axs[0].set_xlabel('R [kpc]')
		axs[0].set_ylabel('Potential [km^2 / s^2]')

		axs[1].set(aspect='equal')
		axs[1].plot(gal.s['x'][isThin], gal.s['y'][isThin], ',', c='blue', alpha=0.3)
		axs[1].plot(gal.s['x'][isThick], gal.s['y'][isThick], ',', c='green', alpha=0.3)
		axs[1].plot(gal.s['x'][isPbulge], gal.s['y'][isPbulge], ',', c='yellow', alpha=0.3)
		axs[1].plot(gal.s['x'][isBulge], gal.s['y'][isBulge], ',', c='red', alpha=0.3)
		axs[1].plot(gal.s['x'][isHalo], gal.s['y'][isHalo], ',', c='orange', alpha=0.3)
		axs[1].set_title('Face-on projection')
		axs[1].set_xlabel('X [kpc]')
		axs[1].set_ylabel('Y [kpc]')

		axs[0].legend(handles=[orange_point, blue_line, red_line])	
		axs[1].legend(handles=[thin_p, thick_p, pbulge_p, bulge_p, halo_p])	

		plt.show()

#-------

	#Compute the stellar mass of the various components
	Mthin = np.sum(gal.s['mass'][isThin])
	Mthick = np.sum(gal.s['mass'][isThick])	
	Mpbulge = np.sum(gal.s['mass'][isPbulge])
	Mbulge = np.sum(gal.s['mass'][isBulge])
	Mhalo = np.sum(gal.s['mass'][isHalo])
	Munbound = np.sum(gal.s['mass'][isUnbound])

	Mstar = np.sum(gal.s['mass'])

	#Evaluate if the galaxy is a disc galaxy
	flag = isDisc(Mthin, Mthick, Mpbulge, Mbulge, Mhalo)

	#Compute the mean energy and circularity of each components
	Emax = np.abs(gal.s['te'][gal.s['morph']!=0]).max()

	Ethin = np.mean(gal.s['te'][isThin]/Emax)
	Ethick = np.mean(gal.s['te'][isThick]/Emax)	
	Epbulge = np.mean(gal.s['te'][isPbulge]/Emax)
	Ebulge = np.mean(gal.s['te'][isBulge]/Emax)
	Ehalo = np.mean(gal.s['te'][isHalo]/Emax)

	Cthin = np.mean(gal.s['jz_by_jzcirc'][isThin])
	Cthick = np.mean(gal.s['jz_by_jzcirc'][isThick])
	Cpbulge = np.mean(gal.s['jz_by_jzcirc'][isPbulge])
	Cbulge = np.mean(gal.s['jz_by_jzcirc'][isBulge])
	Chalo = np.mean(gal.s['jz_by_jzcirc'][isHalo])

#-------

	if isList:
		with open(outname, 'a') as outfile:
			outfile.write('%s %g %g %g %g %g %g %g %d %f %f %f %f %f %f %f %f %f %f\n' % (filename, Mstar, Munbound, Mthin, Mthick, Mbulge, Mpbulge, Mhalo, flag, Ethin, Ethick, Ebulge, Epbulge, Ehalo, Cthin, Cthick, Cbulge, Cpbulge, Chalo))
	else:
		print ("\nMstar = %g\nMunbound = %g\nMthin = %g\nMthick = %g\nMpbulge = %g\nMbulge = %g\nMhalo = %g\nIsDisc = %g\nEthin = %g\nEthick = %g\nEpbulge = %g\nEbulge = %g\nEhalo = %g\nCthin = %g\nCthick = %g\nCpbulge = %g\nCbulge = %g\nChalo = %g\n"
		 % (Mstar, Munbound, Mthin, Mthick, Mpbulge, Mbulge, Mhalo, flag, Ethin, Ethick, Epbulge, Epbulge, Ehalo, Cthin, Cthick, Cpbulge, Cpbulge, Chalo))		
