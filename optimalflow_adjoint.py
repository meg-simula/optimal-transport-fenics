import resource
import csv
import os
import os.path
from cases import csvwrite

from dolfin import *
from dolfin_adjoint import *

def compute_reduced_ocd(c0, c1, tau, D, alpha, results_dir, space="CG", reg="H1"):
    # c0:
    # c1:
    # tau:   time step
    # D:     diffusion coefficient
    # alpha: regularization parameter
    # space: finite element space for the velocity field ("CG" | "RT" | "BDM")
    # reg:   if "H1", use H1-regularization, else use H(div)
    
    info("Computing OCD via reduced approach")

    # Define mesh and function space for the concentration
    mesh = c0.function_space().mesh()
    C = FunctionSpace(mesh, "CG", 1)
    
    # Space for the convective velocity field phi
    if space == "CG":
        Q = VectorFunctionSpace(mesh, "CG", 1)
    else:
        Q = FunctionSpace(mesh, space, 1)
    phi = Function(Q, name="Control")

    # Regularization term
    def R(phi, alpha, mesh):
        if reg == "H1":
            form = 0.5*alpha*(inner(phi, phi) + inner(grad(phi), grad(phi)))*dx(domain=mesh)
        else:
            form = 0.5*alpha*(inner(phi, phi) + inner(div(phi), div(phi)))*dx(domain=mesh)
        return form

    # Define previous solution
    c_ = Function(C)
    c_.assign(c0) # Hack to make dolfin-adjoint happy, maybe just start tape here?

    c2 = Function(C)
    c2.assign(c1)
    
    # Define variational problem
    c = TrialFunction(C)
    d = TestFunction(C)
    F = (1.0/tau*(c - c_)*d + div(c*phi)*d + inner(D*grad(c), grad(d)))*dx() 
    a, L = system(F)
    bc = DirichletBC(C, c2, "on_boundary")

    # ... and solve it once
    c = Function(C, name="State")
    solve(a == L, c, bc, solver_parameters={"linear_solver": "mumps"})

    # Output max values of target and current solution for progress
    # purposes
    info("\max c_1 = %f" % c2.vector().max())
    info("\max c = %f" % c.vector().max())

    # Define the objective functional
    j = 0.5*(c - c2)**2*dx(domain=mesh) + R(phi, alpha, mesh)
    J = assemble(j)
    info("J (initial) = %f" % J)
    
    # Define control field
    m = Control(phi)

    # Define call-back for output at each iteration of the optimization algorithm
    name = lambda s: os.path.join(results_dir, "opts", s)
    dirname = os.path.join(results_dir, "opts")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    header = ("j", "\max \phi")

    # Make an optimization counter file
    with open(os.path.join(results_dir, "counter.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow((0,))     
        
    def eval_cb(j, phi):
        values = (j, phi.vector().max())
        mem = resource.getrusage(resource.RUSAGE_SELF)[2]
        info("Current memory usage: %g (MB)" % (mem/1024))
        info("\tj = %f, \max phi = %f (mm/h)" % values)
        csvwrite(name("optimization_values.csv"), values, header, mode="a")

        # Read the optimization counter file, update counter, and
        # write it back, geez. 
        counter = 0
        with open(os.path.join(results_dir, "counter.csv"), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                counter = int(row[0])
        counter += 1

        with open(os.path.join(results_dir, "counter.csv"), "w") as f:
            info("Updating counter file, counter is now %d " % counter)
            writer = csv.writer(f)
            writer.writerow((counter,))     

        # Write current control variable to file in HDF5 and PVD formats
        file = HDF5File(mesh.mpi_comm(), name("opt_phi_%d.h5" % counter), "w")
        file.write(phi, "/function", 0)
        file.close()

    # Define reduced functional in terms of J and m
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    # Minimize functional
    tol = 1.0e-12
    phi_opt = minimize(Jhat,
                       tol=tol, 
                       options={"gtol": tol, "maxiter": 500, "disp": True})
    pause_annotation()

    # Update phi, and do a final solve to compute c
    phi.assign(phi_opt)
    solve(a == L, c, bc, solver_parameters={"linear_solver": "mumps"})

    J = assemble(j)
    j0 = 0.5*(c - c2)**2*dx(domain=mesh)
    jr = R(phi, alpha, mesh)
    J0 = assemble(j0)
    Jr = assemble(jr)
    info("J  = %f" % J)
    info("J0 = %f" % J0)
    info("Jr = %f" % Jr)
    
    return (c, phi) 

