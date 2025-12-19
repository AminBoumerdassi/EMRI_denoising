import numpy as np

def sample_EMRI_parameters(no_EMRIs=1):
    ''' 
    Generate data according to randomised parameters.
    
    The parameters are:
    
    eta: mass ratio mu/M
    M: larger BH mass
    mu: smaller CO mass
    e0: initial eccentricity
    p0: Initial semilatus rectum
    theta: Polar viewing angle
    phi: Azimuthal viewing angle
    phi_phi0: Initial phase of azimuthal viewing angle in phi plane
    phi_r0: Initial phase of azimuthal viewing angle is r plane
    dist: luminosity distance
    qS: Sky location polar angle in ecliptic coordinates
    phiS: Sky location azimuthal angle in ecliptic coordinates
    qK: Initial BH spin polar angle in ecliptic coordinates
    phiK: Initial BH spin azimuthal angle in ecliptic coordinates
    
    LDCM model used to convert redshifts to luminosity distances.
    '''
    #Intrinsic parameters
    set_of_eta= np.random.uniform(1e-6,1e-4, size= no_EMRIs)#Used indirectly in calculating mu
    set_of_M= np.random.uniform(1e4, 1e7, size = no_EMRIs)
    set_of_mu= set_of_eta*set_of_M
    set_of_a= np.zeros(no_EMRIs)#No spin for schwarzchild model!
    set_of_e0= np.random.uniform(0, .7, size = no_EMRIs)#Initial eccentricity
    set_of_p0= np.random.uniform(10, 16+set_of_e0)#Needs to be based on set_of_e0
    set_of_x0= np.ones(no_EMRIs)#No inclination for scharzchild model
    #Extrinsic parameters
    set_of_dist = 0.01 * np.ones(no_EMRIs)#Units of Gpc
    set_of_qS= np.random.uniform(-np.pi/2, +np.pi/2, size=no_EMRIs)
    set_of_phiS= np.random.uniform(0, 2*np.pi, size=no_EMRIs)
    set_of_qK= np.random.uniform(-np.pi/2, +np.pi/2, size=no_EMRIs)
    set_of_phiK= np.random.uniform(0, 2*np.pi, size=no_EMRIs)
    set_of_phi_phi0= np.random.uniform(0, 2*np.pi, size = no_EMRIs)
    set_of_phi_theta0= np.random.uniform(0, 2*np.pi, size = no_EMRIs)
    set_of_phi_r0= np.random.uniform(0, 2*np.pi, size = no_EMRIs)
    return np.vstack((set_of_M, set_of_mu, set_of_a,
                     set_of_p0,set_of_e0, set_of_x0,
                     set_of_dist, set_of_qS,set_of_phiS,
                     set_of_qK, set_of_phiK, set_of_phi_phi0,
                     set_of_phi_theta0, set_of_phi_r0)).T
