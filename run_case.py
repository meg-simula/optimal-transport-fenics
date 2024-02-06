import os
import os.path
import pylab
import numpy

from dolfin import *
from optimalflow import *
from cases import *

from optimalflow_adjoint import *

def run_synthetic(n=4, method=None, alpha=1.e-4, D_e=0.0, g=0.0, noise=0.0, output=None,
                  eps=1.e-4, D=0.0):
    """ 
    n:       meshsize 
    method:  "OMT", "OCD" or "rOCD"
    alpha:   Regularization parameter (both for OMT, OCD and OCDr)
    D_e:     Diffusion coefficient for the data generated (and for OCDR)
    noise:   Noise level for the generated data
    output:  String with output directory

    eps:     Used for filtering OMT approximation if OMT is given
    D:       Used as diffusion coefficient for (r)OCD if (r)OCD is given
    """
        
    # Generate synthetic data: c_1, c_2, tau = t_2 - t_1, and exact phi
    mesh = UnitSquareMesh(n, n)
    (c1, c2, phi_e, tau) = generate_data_bump(mesh, g, D=D_e, noise=noise, output=output)

    # Compute Linfty, L2 and H10 norms of the exact velocity field phi_e
    phi_e_mag = compute_magnitude_field(phi_e)
    phi_e_max = phi_e_mag.vector().max()
    values = (norm(phi_e, "L2"),  phi_e_max, norm(phi_e, "H10"))

    # Output it
    filename = lambda f: os.path.join(output, f)
    name = filename("norms.csv")
    labels = ("|phi_e|_0", "phi_e_max", "|grad phi_e|_0")
    csvwrite(name, values, header=labels)

    # Compute the OMT approximation
    if (method == "OMT"):
        phi = compute_omt(c1, c2, tau, alpha, space="CG", threshold=eps)

    # or the OCD approximation
    elif (method == "OCD"):
        (c, y, phi) = compute_ocd(c1, c2, tau, D, alpha)

    # or the OCD approximation
    elif (method == "rOCD"):
        set_working_tape(Tape())
        (c, phi) = compute_reduced_ocd(c1, c2, tau, D, alpha, output)

    # Store computed solution to .h5 and .pvd format
    hdf5write(filename("phi.h5"), phi)
    file = File(filename("phi.pvd"))
    file << phi

    # Also store the concentration c if we estimate it
    if not (method == "OMT"):
        hdf5write(filename("c.h5"), c)
        file = File(filename("c.pvd"))
        file << c
    
    # Compute magnitude of the field, and extract peak magnitude
    m = compute_magnitude_field(phi)
    phi_max = m.vector().max()
        
    labels = ("|phi|_0", "phi_max", "|grad phi|_0")
    values = (norm(phi, "L2"),  phi_max, norm(phi, "H10"))
    csvwrite(name, values, header=labels, mode="a")

if __name__ == "__main__":

    # Turn on FEniCS optimizations
    parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

    n = 32
    g = 2.e-2                # Velocity field phi = (g, g)
    D = 1.e-2                # Diffusion coefficient (for data and OCD)
    noise = 0                # Data generation noise level
    method= "rOCD"           # "OMT", "OCD" or "rOCD"
    alpha = 1.e-3            # Regularization parameter
    eps = 1.e-6              # OMT threshold
    results = os.path.join("results-r0", "method_%s_n_%d_alpha_%2.1g" % (method, n, alpha))
    
    run_synthetic(n=n, method=method, alpha=alpha, D_e=D, g=g, noise=noise,
                  output=results, eps=eps, D=D)
