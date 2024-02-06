import numpy
import pylab
from dolfin import *

def non_negativize(I):
    """Given a scalar Function c, return the field with negative values
    replaced by zero.

    """
    c = I.copy(deepcopy=True) 
    vec = c.vector().get_local()
    # Extract dof indices where the vector values are negative
    negs = numpy.where(vec < 0)[0]
    # Set these values to zero
    c.vector()[negs] = 0.0
    return c

def compute_omt(I0, I1, tau, alpha, space="CG", adjust=False, threshold=1.e-3):
    """Given two image intensity fields I0 and I1, a time step tau > 0 and
    a regularization parameter alpha > 0, return a vector field phi
    such that
    
      I_t + div (phi I) \approx 0  in \Omega

    where I(t = 0) = I0 and I(t = tau) = I1. 

    Mueller, M., Karasev, P., Kolesov, I. & Tannenbaum, A. Optical
    flow estimation for flame detection in videos. IEEE Transactions on
    image processing, 22, 2786–2797

    If adjust is given as True, filter the input values by replacing
    negative values by zero, 

    Moreover, set ident_zeros for those rows for which no values
    exceed the given threshold.

    """

    # Map negative values to zero in I0 and I1:
    if adjust:
        info("Adjusting negative input values to zero.")
        I0 = non_negativize(I0)
        I1 = non_negativize(I1)

    # Check balance of mass assumption
    m1 = assemble(I0*dx())
    m2 = assemble(I1*dx())
    info("OMT: Mass preserved (\int I0 == \in I1)? I0 = %.5g, I1 = %0.5g" % (m1, m2))

    Q = I0.function_space()
    mesh = Q.mesh()
    d = mesh.topology().dim()
    
    if space == "RT":
        V = FunctionSpace(mesh, "RT", 1)
    else:
        V = VectorFunctionSpace(mesh, "CG", 1)
        
    # Define dI and barI
    dI = (I1 - I0)/tau
    barI = 0.5*(I0 + I1)
    I = barI

    # Define OMT variational forms
    phi = TrialFunction(V)
    psi = TestFunction(V)
    a = (inner(div(I*phi), div(I*psi)) + alpha*inner(I*phi, psi))*dx()
    L = - inner(dI, div(I*psi))*dx()

    # Assemble linear system
    A = assemble(a)
    b = assemble(L)

    # Set zero (tol) rows to identity as relevant
    A.ident_zeros(tol=threshold)

    # Solve the linear system
    phi = Function(V)
    solve(A, phi.vector(), b, "mumps")

    return phi

def element_cell(mesh):
    "Return the finite element cell."
    return triangle if mesh.topology().dim() == 2 else tetrahedron

def compute_ocd(c1, c2, tau, D, alpha):

    mesh = c1.function_space().mesh()

    cell = element_cell(mesh)
    C = FiniteElement("CG", cell, 1) # H1
    Y = FiniteElement("CG", cell, 1) # H10
    Q = VectorElement("CG", cell, 1) # H1 (H(div) formulation also natural)
    Mx = MixedElement([C, Y, Q])
    M = FunctionSpace(mesh, Mx)

    m = Function(M)
    (c, y, phi) = split(m)
    (d, z, psi) = TestFunctions(M)

    F1 = (c*d + (1./tau*d + div(d*phi))*y + inner(D*grad(d), grad(y)) - c2*d)*dx()
    F2 = (div(c*psi)*y + alpha*(inner(phi, psi) + inner(grad(phi), grad(psi))))*dx()
    F3 = ((1./tau*c + div(c*phi))*z + inner(D*grad(c), grad(z)) - 1./tau*c1*z)*dx()
    F = F1 + F2 + F3

    assign(m.sub(0), c2)

    #set_log_level(LogLevel.DEBUG)
    
    bcs = [DirichletBC(M.sub(0), c2, "on_boundary"),
           DirichletBC(M.sub(1), 0.0, "on_boundary")]

    #info(NonlinearVariationalSolver.default_parameters(), True)
    ps = {"nonlinear_solver": "snes", "snes_solver": {"linear_solver": "mumps", "maximum_iterations": 50, "relative_tolerance": 1.e-10}}
    solve(F == 0, m, bcs, solver_parameters=ps)
    (c, y, phi) = m.split(deepcopy=True)

    return (c, y, phi) 

def compute_magnitude_field(phi):

    # Split phi into its components (really split it)
    phis = phi.split(deepcopy=True)

    # Get the map from vertex index to dof
    v2d = [vertex_to_dof_map(phii.function_space()) for phii in phis]
    dim = len(v2d)
    
    # Create array of component values at vertex indices so p0[0] is
    # value at vertex 0 e.g. 
    if dim == 2:
        p0 = phis[0].vector().get_local(v2d[0])
        p1 = phis[1].vector().get_local(v2d[1])
        # Take element-wise magnitude
        m = numpy.sqrt((numpy.square(p0) + numpy.square(p1)))
    else:
        p0 = phis[0].vector().get_local(v2d[0])
        p1 = phis[1].vector().get_local(v2d[1])
        p2 = phis[2].vector().get_local(v2d[2])
        # Take element-wise magnitude
        m = numpy.sqrt((numpy.square(p0) + numpy.square(p1) + numpy.square(p2)))
        
    # Map array of values at vertices back to values at dofs and
    # insert into magnitude field.
    M = phis[0].function_space()
    magnitude = Function(M)
    d2v = dof_to_vertex_map(M)
    magnitude.vector()[:] = m[d2v]
    return magnitude

if __name__ == "__main__":
    mesh = UnitSquareMesh(4, 4)
    V = VectorFunctionSpace(mesh, "CG", 1)
    phi = Function(V)
    phi.vector()[1] = 2.0
    phi.vector()[0] = 2.0
    a = compute_magnitude_field(phi)
    print(a.vector().max())
    print(a.vector().size())
