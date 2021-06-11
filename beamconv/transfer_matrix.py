"""
transfer_matrix.py

Version 2

This program contains a set of scripts for analyzing stratified media
that vary only in the z-direction, and is infinite in the x-y plane,
with a uniaxial crystal as one layer. This implements the calculations
detailed in:

T. Essinger-Hileman, "Transfer matrix for treating stratified media
including birefringent crystals," Applied Optics, 512:2, 212-218 (2013)

It can also translate the transfer matrix of a stack to Jones
and Mueller matrices, and can calculate polarized transmission, reflection,
and absorption properties of the stack.

Limitations:
** This code will only deal with the case, relevant to ABS, where the
extraordinary axis is tangent to the surfaces of the crystal, at an
angle chi with the x-axis.

** The program would need to be reconsidered for biaxial crystals.

Tom Essinger-Hileman
Johns Hopkins University
March 2014
"""

import numpy as np
from numpy import arcsin, sin, cos, exp, trace, abs
from math import pi
from numpy.lib.scimath import sqrt
from numpy.linalg import inv
from scipy import integrate, interpolate
import os


###############################################################################
# Define constants to be used in the code.
###############################################################################
 
c      = 2.998e8       # m/s
GHz    = 1e9         # gigaHertz
THz    = 1e12        # teraHertz
deg    = pi/180.     # converts degrees to radians.
cm     = 0.01
mm     = 0.001
micron = 1e-6
inch   = 0.0254      # inch to meter conversion.
mil    = 2.54e-5     # mil (0.001") to meter conversion.

###############################################################################
# Start with definitions of objects that will be used in the code.
###############################################################################

class material(object):
    """
    Represents the properties of a given material,
    possibly a unixial crystal.
    """

    def __init__(self, ordinaryIndex, extraIndex,
                 ordinaryLoss, extraLoss, name, materialType='isotropic'):
        """
        Creates an instance of a material class object.
        Inputs are:

        ordinaryIndex - (float) real part of the refractive index for
                        the ordinary axes.
        extraIndex    - (float) real part of the refractive index for
                        the extraordinary axis. 
        ordinaryLoss  - (float) ratio of the imaginary part of the dielectric
                        constant to the real part for ordinary ray.
        extraLoss     - (float) ratio of the imaginary part of the dielectric
                        constant to the real part for extraordinary ray.
        name          - (string) name of the material in question.
        materialType  - (string) 'isotropic' or 'uniaxial'. If 'isotropic'
                        ordinaryIndex = extraIndex.
        """

        self.ordinaryIndex = ordinaryIndex
        self.extraIndex    = extraIndex
        self.extraLoss     = extraLoss
        self.ordinaryLoss  = ordinaryLoss
        self.name          = name
        self.materialType  = materialType

        # Now create complex dielectric constant and refractive indices.
        self.ordinaryEpsilon \
            = (1 - 1j*ordinaryLoss) * ordinaryIndex**2
        self.extraEpsilon \
            = (1 - 1j*extraLoss) * extraIndex**2

        self.ordinaryComplexIndex = sqrt(self.ordinaryEpsilon)
        self.extraComplexIndex    = sqrt(self.extraEpsilon)


    def __str__(self):
        """
        Print material properties in human-readable form.
        """

        s=[]
        s.append("\n")
        s.append("Material: %s"%(self.name))
        s.append("Material Type: %s"%(self.materialType))

        if self.materialType == 'isotropic':
            s.append("Refractive Index: %f"%(self.ordinaryIndex))
            s.append("Loss tangent: %f"%(self.ordinaryLoss))
        elif self.materialType == 'uniaxial':
            s.append("Refractive Index (ordinary): %f"%(self.ordinaryIndex))
            s.append("Refractive Index (extraordinary): %f"%(self.extraIndex))
            s.append("Loss Tangent (ordinary): %f"%(self.ordinaryLoss))
            s.append("Loss Tangent (extraordinary): %f"%(self.extraLoss))
        else: raise ValueError ("materialType is invalid. Must be either 'isotropic' or 'uniaxial.'")
        
        s.append("\n")

        return "\n".join(s)


class Stack(object):
    """
    An object to hold all the properties of a stack, including thicknesses,
    angles of any uniaxial crystals' extraordinary axes, and material objects
    for all the layer materials, as defined above.
    """

    def __init__(self, thicknesses, materials, angles):
        """
        Creates a Stack object using the following inputs:

        thicknesses  - list of thicknesses (float) in meters for the layers.
        materials    - list of material objects describing the materials of each
                       layer. The ordering is the same as the thicknesses list.
        angles       - important only for uniaxial crystal layers when there are
                       multiple uniaxial crystals at different angles in the stack.
                       Gives the angle in radians (float) between the extraordinary
                       axis and the x axis. Normally zero for all layers.
        """

        self.thicknesses = thicknesses
        self.materials   = materials
        self.angles      = angles

        self.numLayers = thicknesses.__len__()


    def __str__(self):
        """
        Prints the details of a stack in human-readable form
        """

        s = []
        s.append("\n")
        s.append("______________________________________________________")
        s.append("\n")

        for layerNum in range(self.numLayers):
            s.append("Layer %d: Thickness %f"%(layerNum+1, \
                                               self.thicknesses[layerNum]))
            materialString = self.materials[layerNum].__str__()
            s.append(materialString)
            if self.materials[layerNum].materialType == 'uniaxial':
                s.append("\t Angle of extraordinary axis to x-axis: %f"\
                         %(self.angles[layerNum]))
            s.append("______________________________________________________\n")

        return "\n".join(s)

        
class transferMatrix(object):         
    """
    Holds the transfer matrix for a single layer, along with
    the properties of the layer in a material object, defined above. This is
    calculated for a single frequency, index of refraction, and rotation of
    the layer about the z axis. This last rotation is only important for
    an anisotropic material.
    """

    def __init__(self, material, thickness, frequency, nsin, rotation):
        """
        Creates a transferMatrix object for a stack. Inputs are:

        material       - a material object that describes the properties of the layer.
        thickness      - thickness in meters of the layer.
        frequency      - frequency of incoming plane wave in Hz.
        nsin           - The invariant n*sin(theta) from Snell's Law.
        rotation       - angle that the stack is rotated about the z axis,
                         CCW from the x axis, also in radians.
        """
        
        self.material       = material
        self.frequency      = frequency
        self.nsin           = nsin
        self.rotation       = rotation

        self.optic_axis     = np.array((cos(rotation),sin(rotation),0),dtype=np.float)

        # Pull quantities that will show up in equations below.
        chi     = rotation
        t       = thickness
        k0      = 2*pi*frequency / c  # Wavenumber in free space.
        
        nO = material.ordinaryIndex
        nEmat = material.extraIndex
        nE = nEmat * np.sqrt(1 + (nEmat**(-2) - nO**(-2)) * nsin**2 *  cos(chi)**2)

        thetaO = arcsin( nsin / nO )
        thetaE = arcsin( nsin / nE )

        self.thetaO = thetaO
        self.thetaE = thetaE

        self.nE = nE        

        nComplexO = material.ordinaryComplexIndex
        nComplexE = sqrt(nE**2 *(1 - 1j*material.extraLoss))

        self.nComplexO = nComplexO
        self.nComplexE = nComplexE

        self.thetaOrdinary = thetaO
        self.thetaExtra    = thetaE
        self.extraordinaryIndex = nE
        self.ordinaryIndex = nO
        self.k0 = k0
        self.chi = chi
        self.ordinaryAngle = arcsin( nsin/nO )
        self.extraAngle    = arcsin( nsin/nE )

        ########################################################################
        # Calculate the rotated dielectric tensor for the material.
        ########################################################################

        rot    = np.array(((cos(rotation), -1*sin(rotation),0),\
                        (sin(rotation), cos(rotation), 0),\
                        (0, 0, 1)), dtype=np.complex)
        rotinv = inv(rot)
        eps    = np.array(((nE**2, 0, 0),(0, nO**2, 0),(0, 0, nO**2)), dtype=np.complex)
        roteps = np.dot( rot, np.dot( eps, rotinv))
        rotepsinv = inv(roteps)
        self.dielectric_tensor = roteps
        self.rot = rot
        self.rotinv = rotinv
        self.eps = eps
        
        ########################################################################
        # Calculate field components transmitted from Interface I.
        # All other fields can be related directly to these.
        ########################################################################

        # Electric displacement D = epsilon*E for ordinary ray.
        DenomDOrd = sqrt( cos(thetaO)**2 + sin(thetaO)**2 * sin(chi)**2)
        DOrd      = np.array(( -1*sin(chi)*cos(thetaO), \
                               cos(chi)*cos(thetaO), \
                               sin(chi)*sin(thetaO)), dtype=np.complex)
        DOrd      = DOrd / DenomDOrd
        self.DOrd = DOrd
        self.EOrd = np.dot(roteps, DOrd)


        # Electric displacement D = epsilon*E for extraordinary ray.
        DenomDExtra = sqrt( cos(chi)**2 *cos(thetaO)**2 + sin(chi)**2 * cos(thetaO-thetaE)**2)
        DExtra      = np.array(( 1*cos(chi)*cos(thetaO)*cos(thetaE), \
                                 1*sin(chi)*( sin(thetaO)*sin(thetaE) + cos(thetaO)*cos(thetaE)),\
                                 - cos(chi)*cos(thetaE)*sin(thetaO)), dtype=np.complex)
        DExtra      = DExtra / DenomDExtra
        self.DExtra = DExtra
        self.EExtra = np.dot(roteps, DExtra)

        # Magnetic field for ordinary ray.
        DenomHOrd = sqrt(cos(thetaO)**2 * cos(chi)**2 + sin(chi)**2)
        HOrd      = np.array(( -cos(thetaO)**2 * cos(chi),\
                               -sin(chi),\
                               cos(thetaO)*sin(thetaO)*cos(chi)), dtype=np.complex)
        HOrd      = HOrd / DenomHOrd
        self.HOrd = HOrd

        # Magnetic field for extraordinary ray.
        DenomHExtra = sqrt( cos(thetaO-thetaE)**2 * sin(chi)**2 + cos(thetaO)**2 * cos(chi)**2 ) 
        HExtra      = np.array(( -cos(thetaO-thetaE) * cos(thetaE) * sin(chi),\
                                 1*cos(thetaO)*cos(chi),\
                                 1*cos(thetaO-thetaE) * sin(thetaE) * sin(chi)),\
                               dtype=np.complex)
        HExtra      = HExtra / DenomHExtra
        self.HExtra = HExtra


        ##########################################################################
        # Form matrices that relate total fields at the two interfaces to the
        # field components above. The matrices will be used to eliminate the
        # field components above to get direct relations between the total fields
        # at the interfaces.
        ##########################################################################

        # Form the matrix relating vI = (Dx,Hy,Dy,-Hx) at interface I to the four
        # D components at the interface in the material. These are formed
        # into a vector v = (D^(o)_(tI), D^(e)_(tI), D^(o)_(r'II), D^(e)_(r'II)).
        # Then vI = M1.v

        Phi = np.array(((DOrd[0], DExtra[0], 1*DOrd[0], 1*DExtra[0]),\
                        (HOrd[1]/nO, HExtra[1]/nE, -1*HOrd[1]/nO, -1*HExtra[1]/nE),\
                        (DOrd[1], DExtra[1], DOrd[1], DExtra[1]),\
                        (-1*HOrd[0]/nO, -1*HExtra[0]/nE, HOrd[0]/nO, HExtra[0]/nE)),\
                      dtype=np.complex)

        # Form the matrix relating vII = (Dx,Hy,Dy,-Hx) at interface II to the four
        # D components at the interface in the material. These are formed
        # into a vector v = (D^(o)_(tI), D^(e)_(tI), D^(o)_(r'II), D^(e)_(r'II)).
        # Then vII = M1.v

        # The fields at the two interfaces are related adding a phase to the wave as it
        # travels across the medium. This phase depends on the index of refraction
        # and angle of travel, as well as the thickness of the material.

        deltaO = 1j * k0 * nComplexO * t * cos(thetaO)
        self.deltaO = deltaO
        deltaE = 1j * k0 * nComplexE * t * cos(thetaE)
        self.deltaE = deltaE

        # Define the propagation matrix P.
        P  = np.array( ((exp(-1*deltaO), 0, 0, 0),\
                        (0, exp(-1*deltaE), 0, 0),\
                        (0, 0, exp(deltaO), 0),\
                        (0, 0, 0, exp(deltaE))), dtype=np.complex)

        # Define the conversion matrix from D field components to those of E.
        Psi = np.array( ((rotepsinv[0,0], 0, rotepsinv[0,1], 0),\
                        (0, 1, 0, 0),\
                        (rotepsinv[1,0], 0, rotepsinv[1,1], 0),\
                        (0, 0, 0, 1)), dtype=np.complex)

        # Now compute the transfer matrix as Psi.Phi.inv(Psi.Phi.P)
        self.Phi = Phi
        self.P   = P
        self.Psi = Psi

        self.transferMatrix = np.dot( Psi, np.dot( Phi, inv(np.dot( Psi, np.dot(Phi, P)))))

        
class stackTransferMatrix(object):
    """
    Calculates the transfer matrix for a stack, and creates 
    an object which holds the individual layers plus the matrix for the stack as a whole.
    """

    def __init__(self, stack, frequency, incidenceAngle, rotation, inputIndex, exitIndex):
        """
        Creates a stackTransferMatrix class instance, with inputs:

        stack          - Stack class instance holding information on the stack materials and thicknesses.
        frequency      - (float) frequency of incoming plane wave in Hz.
        incidenceAngle - (float) angle of incidence of the incoming plane wave in radians.
        rotation       - (float) rotation angle about the z-axis of the stack in radians.
        inputIndex     - (float) refractive index of medium containing incoming and reflected waves.
        exitIndex      - (float) refractive index of medium containing transmitted wave.
        """

        self.stack          = stack
        self.frequency      = frequency
        self.incidenceAngle = incidenceAngle
        self.rotation       = rotation
        self.inputIndex     = inputIndex
        self.exitIndex      = exitIndex
        self.nsin           = sin(incidenceAngle)*inputIndex

        numLayers   = stack.numLayers
        materials   = stack.materials
        thicknesses = stack.thicknesses
        angles      = stack.angles

        self.transfers = []
        self.totalTransfer = np.eye(4, dtype=np.complex)

        for layerNum in range(numLayers):
            material      = materials[layerNum]
            thickness     = thicknesses[layerNum]
            angle         = angles[layerNum]
            layerRotation = angle + rotation

            # Get the input and output angles for this layer. This depends on the
            # refractive indices of the materials before and after.
            if layerNum == 0: theta1 = incidenceAngle
            else: theta1 = arcsin( sin(incidenceAngle)*inputIndex / materials[layerNum-1].ordinaryIndex)

            if layerNum == (numLayers-1):
                theta3 = arcsin( sin(incidenceAngle)*inputIndex / exitIndex)
                self.exitAngle = theta3
            else: theta3 = arcsin( sin(incidenceAngle)*inputIndex / materials[layerNum+1].ordinaryIndex)

            layerTransfer = transferMatrix(material, thickness, frequency, self.nsin, layerRotation)
            self.transfers.append( layerTransfer )
            self.totalTransfer = np.dot( self.totalTransfer, layerTransfer.transferMatrix )


#################################################################################################################
# The functions below calculate Jones and Mueller matrices given a transfer matrix.
#################################################################################################################

def TranToJones(transfer):
    """
    Calculates the Jones matrices for reflected and transmitted waves from a transfer
    matrix. The input is a stackTransferMatrix class instance. Outputs a list where the first element is the
    2 x 2 Jones matrix for the transmitted wave and the second element is the 2 x 2 Jones matrix for
    the reflected wave.

    output - (list) list[0] is the transmitted jones matrix. list[1] is the reflected jones matrix.
    """

    # Grab the information we need from the stackTransferMatrix object.
    m      = transfer.totalTransfer # total transfer matrix.
    n1     = transfer.inputIndex
    n3     = transfer.exitIndex
    theta1 = transfer.incidenceAngle
    theta3 = transfer.exitAngle
   
    # Define a series of constants which will then go to making the relevant equations.
    A = (m[0,0] * cos(theta3) + m[0,1]*n3) / cos(theta1)
    B = (m[0,2] + m[0,3]*n3 * cos(theta3)) / cos(theta1)
    C = (m[1,0] * cos(theta3) + m[1,1]*n3) / n1
    D = (m[1,2] + m[1,3]*n3 * cos(theta3)) / n1
    N = (m[2,0] * cos(theta3) + m[2,1]*n3) 
    K = (m[2,2] + m[2,3]*n3 * cos(theta3)) 
    P = (m[3,0] * cos(theta3) + m[3,1]*n3) / (n1*cos(theta1))
    S = (m[3,2] + m[3,3]*n3 * cos(theta3)) / (n1*cos(theta1))

    # Now construct the Jones matrices for the reflected and transmitted rays.
    denom = (A+C)*(K+S)-(B+D)*(N+P) # Both matrices share a factor dividing all elements.

    # Transmitted Jones matrix.
    Jtran = np.array(((K+S, -B-D),(-N-P, A+C)), dtype=np.complex) * 2 / denom

    # Reflected Jones matrix.
    Jref = np.array(( ((C-A)*(K+S)-(D-B)*(N+P), 2*(A*D - C*B)),\
                      (2*(N*S - P*K), (A+C)*(K-S) - (D+B)*(N-P))),\
                    dtype=np.complex) / denom

    return Jtran, Jref


def JonesToMueller(jones):
    """
    Given a Jones matrix, computes the corresponding Mueller matrix. The input
    matrix should be a 2x2 complex np.array object.
    """

    # Form the Pauli matrices into a list where Sigma[i] is the corresponding 2x2 matrix.
    # Note that they're in an unorthodox order, so that they match up properly with the
    # Stokes parameters. Note that to match up with Polarized Light by Goldstein,
    # I had to multiply the last Pauli matrix by -1, a change from the Jones et al paper.
    Sigma = []
    Sigma.append( np.array(( (1,0),(0,1)), dtype=np.complex)) # identity matrix
    Sigma.append( np.array(( (1,0),(0,-1)), dtype=np.complex))
    Sigma.append( np.array(( (0,1),(1,0)), dtype=np.complex))
    Sigma.append( np.array(( (0,-1j),(1j,0)), dtype=np.complex)) # Need to multiply by -1 to change back to normal.

    # Now the Mueller matrix elements are given by Mij = 1/2 * Tr(sigma[i]*J*sigma[j]*J^dagger)
    m = np.zeros((4,4), dtype=np.float)

    for i in range(4):
        for j in range(4):
            temp = trace( np.dot(Sigma[i], np.dot(jones, np.dot(Sigma[j], jones.conj().transpose())))) / 2

            # This is just a sanity check to make sure that the formula works and doesn't leave
            # an imaginary part floating around where the Mueller-matrix elements should be real.
            if np.imag(temp) > 0.000000001: print ('Discarding an imaginary part unnecessarily!!!!')
            m[i,j] = np.real(temp)
    return m



############################################################################################################
# Scripts to go straight to Mueller and Jones matrices from a Stack object.
#
############################################################################################################

def Mueller( stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False):
    """
    Returns the Mueller matrix as a 4x4 numpy array for the given Stack, frequency, angle of incidence,
    rotation of the stack, and optionally input and exit indices of refraction. If you want the reflected
    Mueller matrix, set reflected to True.
    """

    transfer = stackTransferMatrix( stack, frequency, incidenceAngle, rotation, inputIndex, exitIndex )
    jones   = TranToJones(transfer)

    if reflected==False:
        mueller = JonesToMueller( jones[0] )
    elif reflected==True:
        mueller = JonesToMueller( jones[1] )
    else: raise ValueErrors ("Invalid value for reflected. Must be True or False.")

    return mueller


def Jones( stack, frequency, incidenceAngle, rotation, inputIndex=1.0, exitIndex=1.0, reflected=False):
    """
    Returns the Jones matrix as a 2x2 complex numpy array for the given Stack, frequency, angle of incidence,
    rotation of the stack, and optionally input and exit indices of refraction. If you want the reflected
    Jones matrix, set reflected to True.
    """

    transfer = stackTransferMatrix( stack, frequency, incidenceAngle, rotation, inputIndex, exitIndex )
    jones   = TranToJones(transfer)

    if reflected==False:
        output = jones[0] 
    elif reflected==True:
        output = jones[1]
    else: raise ValueError ("Invalid value for reflected. Must be True or False.")

    return output


#############################################################################################################
# These scripts calculate band-averaged quantities. In particular, individual Mueller-matrix elements can
# be band averaged. The band averages depend upon the frequency spectrum of the source being looked at.
############################################################################################################

def BandAveragedMueller( stack, spectrumFile, minFreq, maxFreq, reflected=False, passBandFile=False,\
                         incidenceAngle=0.0, numFreqs=1000, rotation=0.0, inputIndex=1.0, outputIndex=1.0 ): 
    """
    Calculates the band-averaged Mueller matrix, given a Stack object describing an optical
    system and a frequency spectrum of the source you are observing. 

    Required arguments are:

    stack         - (Stack class instance) stack object describing the optical system in question.
    spectrumFile  - (string) Name of the tab-delimitted TXT file which holds the frequency spectrum of the
                        source being observed. The first column in this file must be frequency in Hz and the
                        second column is the power spectral density at that frequency in W/(m**2 * Hz).
                        The first two lines of the file will be ignored and can be used as explanatory text.
    minFreq       - (float) The lowest frequency, in Hz, to integrate over.
    maxFreq       - (float) The highest frequency, in Hz, to integrate over.
    
    
    Optional arguments are:

    reflection     - (boolean) Whether or not to look at reflected Mueller matrix. Default is False,
                        meaning that the Mueller matrix for transmission will be calculated.
    passbandFile   - (string) Name of the tab-delimitted TXT file that holds the detector passband.
                        The first column is frequency in GHz. The second column is normalized detector response,
                        a value between 0 and 1. Default is a flat band (equal to unity) over the whole
                        frequency range. The first two lines of the file will be ignored and can be
                        used as explanatory text.
    incidenceAngle - (float) Angle of incidence in degrees of the incoming plane wave. Default is zero.
    numFreqs       - (integer) The number of frequencies between minFreq and maxFreq to consider. default is 1000.
    rotation       - (float) Rotation of the stack in degrees. Default is zero.
    inputIndex     - (float) Index of refraction of the input medium to the stack. Default is vacuum.
    OutputIndex    - (float) Index of refraction of the output medium to the stack. Default is vacuum.    

    Returns a 4x4 numpy array of floats, which is the Mueller matrix of the stack integrated
    against the detector passband and the spectrum of the source being observed.
    """

    frequencies = minFreq + np.arange(numFreqs, dtype=np.float) * (maxFreq - minFreq)/numFreqs

    # Create arrays to hold the Mueller-matrix elements at each frequency.
    MuellerII = np.zeros(numFreqs, dtype=np.float)
    MuellerIU = np.zeros(numFreqs, dtype=np.float)
    MuellerIQ = np.zeros(numFreqs, dtype=np.float)
    MuellerIV = np.zeros(numFreqs, dtype=np.float)
    MuellerQI = np.zeros(numFreqs, dtype=np.float)
    MuellerQU = np.zeros(numFreqs, dtype=np.float)
    MuellerQQ = np.zeros(numFreqs, dtype=np.float)
    MuellerQV = np.zeros(numFreqs, dtype=np.float)
    MuellerUI = np.zeros(numFreqs, dtype=np.float)
    MuellerUU = np.zeros(numFreqs, dtype=np.float)
    MuellerUQ = np.zeros(numFreqs, dtype=np.float)
    MuellerUV = np.zeros(numFreqs, dtype=np.float)
    MuellerVI = np.zeros(numFreqs, dtype=np.float)
    MuellerVQ = np.zeros(numFreqs, dtype=np.float)
    MuellerVU = np.zeros(numFreqs, dtype=np.float)
    MuellerVV = np.zeros(numFreqs, dtype=np.float)

    
    # Calculate the Mueller matrix of the stack at each of the frequencies. 
    for i in range(numFreqs):
        transfer = stackTransferMatrix( stack, frequencies[i], incidenceAngle, rotation, inputIndex, outputIndex)
        jones   = TranToJones(transfer)

        if reflected==True:
            tempMueller = JonesToMueller( jones[1] ) # Use the Jones matrix for reflection.
        else:
            tempMueller = JonesToMueller( jones[0] ) # Use the Jones matrix for transmission.

        MuellerII[i] = tempMueller[0,0]
        MuellerIQ[i] = tempMueller[0,1]
        MuellerIU[i] = tempMueller[0,2]
        MuellerIV[i] = tempMueller[0,3]
        MuellerQI[i] = tempMueller[1,0]
        MuellerQQ[i] = tempMueller[1,1]
        MuellerQU[i] = tempMueller[1,2]
        MuellerQV[i] = tempMueller[1,3]
        MuellerUI[i] = tempMueller[2,0]
        MuellerUQ[i] = tempMueller[2,1]
        MuellerUU[i] = tempMueller[2,2]
        MuellerUV[i] = tempMueller[2,3]
        MuellerVI[i] = tempMueller[3,0]
        MuellerVQ[i] = tempMueller[3,1]
        MuellerVU[i] = tempMueller[3,2]
        MuellerVV[i] = tempMueller[3,3]
        

    # Now read in the spectrum from the file.
    specFile = open( spectrumFile, 'r')
    specText = specFile.readlines()
    spectrumData = np.zeros((2,len(specText)-2), dtype=np.float)


    # Clip the first two lines of the spectrum. These are descriptive text.
    for i in range(2, len(specText)):
        freq, spec = specText[i].split()
        spectrumData[0,i-2] = freq
        spectrumData[1,i-2] = spec        

    # We need to interpolate the spectrum data to get approximate values at
    # each of the frequencies chosen above.
    spectrumFunction = interpolate.interp1d( spectrumData[0,:], spectrumData[1,:]) 
    observedSpectrum = spectrumFunction( frequencies ) 


    # Now get the detector passband from the file, if a file has been specified. Otherwise
    # the passband is flat and unity across the entire range of frequencies. Will once again
    # need to interpolate the data to the frequencies chosen above.

    if passBandFile==False:
        passBand = np.zeros(numFreqs, dtype=np.float)
        
        for i in range(numFreqs):
            passBand[i] = 1.0

    else:
        bandFile = open( passBandFile, 'r')
        bandText = bandFile.readlines()
        passBandData = np.zeros( (2,len(bandText)-2), dtype=np.float)

        for j in range(2, len(bandText)):
            freq, band = bandText[j].split()
            passBandData[0,j-2] = freq
            passBandData[1,j-2] = band

        passBandFunction = interpolate.interp1d( passBandData[0,:], passBandData[1,:] )
        passBand = passBandFunction( frequencies )


    # Integrate to compute the band-averaged Mueller-matrix elements. The integral is:
    # Int( Spectrum(f) passBand(f) Mueller(f) df) / Int( Spectrum(f) passBand(f) df )
    # The integration is carried out for each Mueller-matrix element separately.

    IntegratedMuellerII = integrate.trapz(MuellerII*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerIQ = integrate.trapz(MuellerIQ*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)    
    IntegratedMuellerIU = integrate.trapz(MuellerIU*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerIV = integrate.trapz(MuellerIV*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerQI = integrate.trapz(MuellerQI*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerQQ = integrate.trapz(MuellerQQ*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerQU = integrate.trapz(MuellerQU*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerQV = integrate.trapz(MuellerQV*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerUI = integrate.trapz(MuellerUI*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerUQ = integrate.trapz(MuellerUQ*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerUU = integrate.trapz(MuellerUU*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerUV = integrate.trapz(MuellerUV*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerVI = integrate.trapz(MuellerVI*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerVQ = integrate.trapz(MuellerVQ*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerVU = integrate.trapz(MuellerVU*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)
    IntegratedMuellerVV = integrate.trapz(MuellerVV*observedSpectrum*passBand, frequencies)/ \
                          integrate.trapz(observedSpectrum*passBand, frequencies)


    # Now return the total Mueller matrix as a numpy array.
    IntegratedMueller = np.array( \
        ((IntegratedMuellerII,IntegratedMuellerIQ,IntegratedMuellerIU,IntegratedMuellerIV),\
         (IntegratedMuellerQI,IntegratedMuellerQQ,IntegratedMuellerQU,IntegratedMuellerQV),\
         (IntegratedMuellerUI,IntegratedMuellerUQ,IntegratedMuellerUU,IntegratedMuellerUV),\
         (IntegratedMuellerVI,IntegratedMuellerVQ,IntegratedMuellerVU,IntegratedMuellerVV)),\
        dtype=np.float)

    return IntegratedMueller
    

def JonesRotation(jones, theta):

    rot = np.array(((cos(theta*deg),-sin(theta*deg)),(sin(theta*deg),cos(theta*deg))),dtype=np.complex)

    return np.dot(rot, np.dot( jones, inv(rot)))

def MuellerRotation(mueller, theta):

    rot = np.array(((1.,0.,0.,0.), (0.,cos(2*theta),sin(2*theta),0.), (0.,-sin(2*theta),cos(2*theta),0.), (0.,0.,0.,1.)))
    m_rot = np.array(((1.,0.,0.,0.), (0.,cos(-2*theta),sin(-2*theta),0.), (0.,-sin(2*theta),cos(-2*theta),0.), (0.,0.,0.,1.)))

    return np.dot(m_rot, np.dot(mueller, rot))

