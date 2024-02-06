import os
import os.path
import csv
import numpy

from dolfin import *
from optimalflow import non_negativize

def hdf5write(name, field):
    mesh = field.function_space().mesh()
    file = HDF5File(mesh.mpi_comm(), name, "w")
    file.write(field, "/function", 0)
    file.close()

def csvwrite(name, values, header=None, debug=True, mode="w"): 
    if debug:
        print(header)
        print(" & ".join("$%.4f$" % v for v in values))

    with open(name, mode) as f:
        writer = csv.writer(f)
        writer.writerow(header)     
        writer.writerow(values)     

def generate_data_bump(mesh, g, D=0.0, r=0.0, noise=0.0, output=None):
    """Generate data for a relatively smooth test case with a bump moving
    in the middle of the domain.

    Domain: [0, 1]**2
    c_0: Bump function 
    phi = (g, g)

    Let c_1 be defined by one timestep of the convection-diffusion-reaction equation

      c_t + div (phi c) - D div grad c + r c = 0    
    
    where c_0 and phi are as given, and D and r if given. Evolve by a
    timestep tau = 1.0.

    If outputdir is given as a string, write generated data to given
    directory.
    """
    #tau = Constant(1.0) # Time step
    tau = 1.0 # Time step

    Q = FunctionSpace(mesh, "CG", 1)

    # Define initial c0
    bump_x = "beta/(2*alpha*tgamma(1./beta))*std::exp(-pow((std::abs(x[0] - mu)/alpha), beta))"  
    bump_y = "beta/(2*alpha*tgamma(1./beta))*std::exp(-pow((std::abs(x[1] - mu)/alpha), beta))"  
    bump = Expression("1 + 1./7*%s*%s" % (bump_x, bump_y), degree=3, beta=8.0, alpha=0.2, mu=0.5)
    c0 = interpolate(bump, Q)
    
    # Define initial phi 
    V = VectorFunctionSpace(mesh, "CG", 1)
    phi = Expression((str(g), str(g)), degree=1)
    phi = interpolate(phi, V)

    # Define variational problem for evolution of c
    c = TrialFunction(Q)
    d = TestFunction(Q)
    a = (inner(c, d) + tau*inner(div(c*phi), d))*dx()  
    if D is not None and abs(D) > 0.0:
        a += tau*inner(D*grad(c), grad(d))*dx()
    if r is not None and abs(r) > 0.0:
        a += tau*r*c*d*dx()

    L = inner(c0, d)*dx()
    A = assemble(a)
    b = assemble(L)

    if D is not None and D > 0:
        bc = DirichletBC(Q, c0, "on_boundary")
        bc.apply(A, b)
    
    # Solve it
    c1 = Function(Q)
    solve(A, c1.vector(), b, "mumps")

    # Add noise to the concentrations:
    if noise: 
        N = c0.vector().size()
        r0 = numpy.random.normal(0, noise, N)
        r1 = numpy.random.normal(0, noise, N)
        c0.vector()[:] += r0
        c1.vector()[:] += r1

    # Zero any negative values 
    c0 = non_negativize(c0)
    c1 = non_negativize(c1)

    # Optional: Store solutions to .h5 and .pvd format 
    if output is not None:
        filename = lambda f: os.path.join(output, "data", f)
        hdf5write(filename("c0_e.h5"), c0)
        hdf5write(filename("c1_e.h5"), c1)
        hdf5write(filename("phi_e.h5"), phi)

        file = File(filename("c_e.pvd"))
        file << c0
        file << c1
        file = File(filename("phi_e.pvd"))
        file << phi

        csvwrite(filename("info_e.csv"), (float(tau), D, g), header=("tau", "D", "g"))

    return (c0, c1, phi, tau)

def generate_data_bdry(mesh, g, D=0.0, r=0.0, noise=0.0, tau=1.0, output=None):
    """Generate data for a test case with a concentration appearing on the boundary.

    Domain: Given
    c_0: 1.0 on the left and lower boundary if d = 2, on_boundary iff d=3
    phi = (g, 0) if d == 2, (g, 0, 0) iff d == 3

    Let c_1 be defined by one timestep of the convection-diffusion-reaction equation

      c_t + div (phi c) - D div grad c + r c = 0    
    
    where c_0 and phi are as given, and D and r if given. Evolve by a
    timestep tau = 1.0.

    If outputdir is given as a string, write generated data to given
    directory.
    """
    tau = Constant(tau) # Time step

    Q = FunctionSpace(mesh, "CG", 1)
    dim = mesh.topology().dim()
    if dim == 2:
        bc = DirichletBC(Q, 1.0, "near(x[0], 0.0) || near(x[1], 0.0)", "pointwise")
    else:
        bc = DirichletBC(Q, 1.0, "on_boundary")
    c_ = Function(Q)
    bc.apply(c_.vector())
    
    # Define initial phi 
    V = VectorFunctionSpace(mesh, "CG", 1)
    if dim == 2:
        phi = Expression((str(g), "0.0"), degree=1)
    else:
        phi = Expression((str(g), "0.0", "0.0"), degree=1)
    phi = interpolate(phi, V)

    # Define variational problem for evolution of c
    c = TrialFunction(Q)
    d = TestFunction(Q)
    a = (inner(c, d) + tau*inner(div(c*phi), d))*dx()  
    if D is not None and abs(D) > 0.0:
        a += tau*inner(D*grad(c), grad(d))*dx()
    if r is not None and abs(r) > 0.0:
        a += tau*r*c*d*dx()

    L = inner(c_, d)*dx()
    A = assemble(a)
    b = assemble(L)

    if D is not None and D > 0:
        bc = DirichletBC(Q, c_, "on_boundary")
        bc.apply(A, b)
    
    # Solve it for c0
    c0 = Function(Q)
    solve(A, c0.vector(), b, "mumps")

    # Evolve further to c1
    c_.assign(c0)
    b = assemble(L)
    if D is not None and D > 0:
        bc = DirichletBC(Q, c0, "on_boundary")
        bc.apply(A, b)

    c1 = Function(Q)
    solve(A, c1.vector(), b, "mumps")
    
    # Add noise to the concentrations:
    if noise: 
        N = c0.vector().size()
        r0 = numpy.random.normal(0, noise, N)
        r1 = numpy.random.normal(0, noise, N)
        c0.vector()[:] += r0
        c1.vector()[:] += r1

    # Zero any negative values 
    c0 = non_negativize(c0)
    c1 = non_negativize(c1)

    # Optional: Store solutions to .h5 and .pvd format 
    if output is not None:
        filename = lambda f: os.path.join(output, f)
        hdf5write(filename("c0_e.h5"), c0)
        hdf5write(filename("c1_e.h5"), c1)
        delta_c = project(c1 - c0, c0.function_space())
        hdf5write(filename("delta_c_e.h5"), delta_c)
        hdf5write(filename("phi_e.h5"), phi)

        file = File(filename("c_e.pvd"))
        file << c0
        file << c1
        file << delta_c
        file = File(filename("phi_e.pvd"))
        file << phi

        csvwrite(filename("info_e.csv"), (float(tau), g, D, r), header=("tau", "g", "D", "r"))

    return (c0, c1, phi, tau)

if __name__ == "__main__":

    mesh = UnitSquareMesh(32, 32)
    (c0, c1, phi, tau) = generate_data_bump(mesh, 0.02, output="tmp-bump")
    (c0, c1, phi, tau) = generate_data_bdry(mesh, 0.02, output="tmp-bdry")
