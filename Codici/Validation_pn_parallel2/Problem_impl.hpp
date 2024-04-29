using namespace dealii;
using namespace std;

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CONSTRUCTOR
template <int dim>
Problem<dim>::Problem(parallel::distributed::Triangulation<dim> &tria)
  : triangulation(tria)                      // linear elements, we approximate our variables linearly on the elements
  , fe(1) 
  , dof_handler(tria)  
  , mapping()                   // initialize mapping 2D
  , step_number(0)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{}

// since only these data members are initialized in the constructor, all the other members will be initialized by their own default constructor

//---------------------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void Problem<dim>::setup_dofs()
{
    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs(fe);     //Distribute the dof needed for the given fe on the triangulation that i passed in the constructor   // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    DoFRenumbering::Cuthill_McKee(dof_handler);                      //Renumber the degrees of freedom according to the Cuthill-McKee method.

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);     //Extract the set of global DoF indices that are active on the current DoFHandler. This is the union of DoFHandler::locally_owned_dofs() and the DoF indices on all ghost cells.

    pcout << "   Number of active cells: "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <int dim>
void Problem<dim>::make_constraints_poisson()
{
	constraints_poisson.clear();  // Reset the flag determining whether new entries are accepted or not
	zero_constraints_poisson.clear(); 

	constraints_poisson.reinit(locally_relevant_dofs);      
    zero_constraints_poisson.reinit(locally_relevant_dofs);

	DoFTools::make_hanging_node_constraints(dof_handler, constraints_poisson);       // Compute the constraints resulting from the presence of hanging nodes. We put the result in constraints_poisson
	DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints_poisson);

	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(V_E*log(N1/N_0)), constraints_poisson); // Set the values of constraints_poisson on the left edge boundary of the domain
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(V_E*log(N2/N_0)), constraints_poisson); // Set the values of constraints_poisson on the right edge boundary of the domain
	
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints_poisson); // Set the values of zero_constraints_poisson on the left edge boundary of the domain to zero
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints_poisson); // Set the values of zero_constraints_poisson on the right edge boundary of the domain to zero (homo Dirichlet)

	constraints_poisson.close();
	zero_constraints_poisson.close();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void Problem<dim>::make_constraints_drift_diffusion()
{   
	constraints_electron.clear();
	constraints_ion.clear();   

	constraints_electron.reinit(locally_relevant_dofs);      
    constraints_ion.reinit(locally_relevant_dofs); 

	DoFTools::make_hanging_node_constraints(dof_handler, constraints_electron); 
	DoFTools::make_hanging_node_constraints(dof_handler, constraints_ion); 
	
	VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(N1), constraints_electron); 

	VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(N2), constraints_electron);

	VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(P1), constraints_ion);

	VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(P2), constraints_ion);
	
	constraints_electron.close();
	constraints_ion.close(); 
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void Problem<dim>::initialize_system_poisson()
{
	system_matrix_poisson.clear();
    laplace_matrix_poisson.clear();
    mass_matrix_poisson.clear();

	// SPARSITY PATTERN
	DynamicSparsityPattern dsp(locally_relevant_dofs);      // it mostly represents a SparsityPattern object that is kept compressed at all times, memory reason. We initialize a square pattern of size n_dof
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_poisson, true);    // Compute which entries of a matrix built on the given dof_handler may possibly be nonzero, and create a sparsity pattern object that represents these nonzero locations.
    
	SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

	 // INITIALIZATION OF SYSTEM MATRICES AND RHS											 
	system_matrix_poisson.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator); // Reinitialize the sparse matrix with the given sparsity pattern. The latter tells the matrix how many nonzero elements there need to be reserved.
    laplace_matrix_poisson.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);// As above
	mass_matrix_poisson.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);   // As above

	poisson_rhs.reinit(locally_owned_dofs, mpi_communicator); // initialize the rhs vector

	poisson_newton_update.reinit(locally_owned_dofs, mpi_communicator);
	
	assemble_laplace_matrix();
	assemble_mass_matrix();

	cout << "laplace_matrix linf norm 1 is " << laplace_matrix_poisson.linfty_norm() << endl;
  	cout << "mass_matrix linf norm 1 is " << mass_matrix_poisson.linfty_norm() << endl;
  	cout << "laplace_matrix frob norm 1 is " << laplace_matrix_poisson.frobenius_norm() << endl;
  	cout << "mass_matrix frob norm 1 is " << mass_matrix_poisson.frobenius_norm() << endl;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void Problem<dim>::initialize_system_drift_diffusion()
{
    electron_system_matrix.clear();
    electron_drift_diffusion_matrix.clear();
	ion_system_matrix.clear();
	drift_diffusion_matrix.clear();
	
    
	DynamicSparsityPattern dsp_electron(locally_relevant_dofs); // it mostly represents a SparsityPattern object that is kept compressed at all times, memory reason. We initialize a square pattern of size n_dof
	DoFTools::make_sparsity_pattern(dof_handler, dsp_electron, constraints_electron); // Compute which entries of a matrix built on dof_handler may possibly be nonzero, and create a sparsity pattern object that represents these nonzero locations, we put it in dsp
	SparsityTools::distribute_sparsity_pattern(dsp_electron, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
 
	electron_system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_electron, mpi_communicator);          //Same 
	electron_drift_diffusion_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_electron, mpi_communicator); //Same                                

	DynamicSparsityPattern dsp_ion(locally_relevant_dofs);            //This class implements an array of compressed sparsity patterns (one for velocity and one for pressure in our case) that can be used to initialize objects of type BlockSparsityPattern.
    DoFTools::make_sparsity_pattern(dof_handler, dsp_ion, constraints_ion);     //Compute which entries of a matrix built on the given dof_handler may possibly be nonzero, and create a sparsity pattern (assigning it to dsp) object that represents these nonzero locations.
    SparsityTools::distribute_sparsity_pattern(dsp_ion, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
	
    ion_system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_ion, mpi_communicator);               //Reinitialize sparse matrix ion_system_matrix with the given sparsity pattern. The latter tells the matrix how many nonzero elements there need to be reserved.
	drift_diffusion_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_ion, mpi_communicator);          //Same 
	

	electron_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);      //Resize the dimension of the vector electron_density to the number of the dof (unsigned int)
	old_electron_density.reinit(locally_owned_dofs, mpi_communicator);  //Resize the dimension of the vector old_electron_density to the number of the dof (unsigned int)
	electron_rhs.reinit(locally_owned_dofs, mpi_communicator); 

	ion_density.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);           //Resize the dimension of the vector ion_density to the number of the dof (unsigned int)
    old_ion_density.reinit(locally_owned_dofs, mpi_communicator);       //Resize the dimension of the vector old_ion_density to the number of the dof (unsigned int)
	ion_rhs.reinit(locally_owned_dofs, mpi_communicator); 

	potential.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);  // ??
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void Problem<dim>::assemble_laplace_matrix()
{
	const QTrapezoid<dim> quadrature_formula;

	laplace_matrix_poisson = 0;

	FEValues<dim> fe_values(fe,
							quadrature_formula,
							update_values | update_gradients |
							update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points    = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators())
	if (cell->is_locally_owned())
		{
		cell_matrix = 0.;

		fe_values.reinit(cell);

		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{
			for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
				}
			}

		cell->get_dof_indices(local_dof_indices);
		constraints_poisson.distribute_local_to_global( cell_matrix, local_dof_indices, laplace_matrix_poisson );
		}
	laplace_matrix_poisson.compress(VectorOperation::add);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void Problem<dim>::assemble_mass_matrix()
{
	const QTrapezoid<dim> quadrature_formula;

	mass_matrix_poisson = 0;

	FEValues<dim> fe_values(fe,
							quadrature_formula,
							update_values | update_gradients |
							update_quadrature_points | update_JxW_values);

	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	const unsigned int n_q_points    = quadrature_formula.size();

	FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	for (const auto &cell : dof_handler.active_cell_iterators())
	if (cell->is_locally_owned())
		{
		cell_matrix = 0.;

		fe_values.reinit(cell);

		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{
			for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
				for (unsigned int j = 0; j < dofs_per_cell; ++j)
					cell_matrix(i, j) += fe_values.shape_value(i, q_point) * fe_values.shape_value(j, q_point) * fe_values.JxW(q_point);
				}
			}

		cell->get_dof_indices(local_dof_indices);
		constraints_poisson.distribute_local_to_global( cell_matrix,local_dof_indices, 	mass_matrix_poisson );
		}
	mass_matrix_poisson.compress(VectorOperation::add);
}


// DA QUESTO PUNTO IN AVANTI SI ENTRA NEL CICLO WHILE, QUINDI ABBAIMO A CHE FARE CON QUANTITA' CHE SI AGGIORNANO


//------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int dim>
void Problem<dim>::assemble_nonlinear_poisson()
{
  DynamicSparsityPattern dsp_poisson(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp_poisson, constraints_poisson, true);
  SparsityTools::distribute_sparsity_pattern(dsp_poisson, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

  // ASSEMBLE MATRICES
  system_matrix_poisson = 0; //sets all elements of the matrix to zero, but keep the sparsity pattern previously used.
  PETScWrappers::MPI::SparseMatrix ion_mass_matrix; // We initialized the new ion_mass_matrix with our sparsity pattern
  ion_mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp_poisson, mpi_communicator);
  ion_mass_matrix = 0; // and then set all the elments to zero

  
  // compute the ion mass matrix
  for (unsigned int i = 0; i < old_ion_density.size(); ++i){
	  ion_mass_matrix.set(i,i, mass_matrix_poisson(i,i) * (old_ion_density(i) + old_electron_density(i)));
  }

  ion_mass_matrix.compress(VectorOperation::insert); 

  cout << "Matrix linf norm 1 is " << ion_mass_matrix.linfty_norm() << endl;
  cout << "Matrix frob norm 1 is " << ion_mass_matrix.frobenius_norm() << endl;

  system_matrix_poisson.add(q0 / V_E, ion_mass_matrix);  // A += factor * B with the passed values
  cout << "system_matrix_poisson norm is " << system_matrix_poisson.linfty_norm() << std::endl;  // compute and print the infinit norm of the matrix

  system_matrix_poisson.add(eps_r * eps_0, laplace_matrix_poisson); // same as above
  cout << "system_matrix_poisson norm is " << system_matrix_poisson.linfty_norm() << std::endl;

  // ASSEMBLE RHS
  poisson_rhs = 0; // set all the values to zero
  PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator); //temporary vector of dimension n_dof
  PETScWrappers::MPI::Vector doping_and_ions(locally_owned_dofs, mpi_communicator); // create a new vector of dimension n_dof
  VectorTools::interpolate(mapping,dof_handler, DopingValues<dim>(), doping_and_ions); // We interpolate the previusly created vector with the initial values of Doping provided by DopingValues
  
  doping_and_ions -= old_electron_density;
  doping_and_ions += old_ion_density;

  mass_matrix_poisson.vmult(tmp,doping_and_ions);   //tmp = mass_matrix_poisson * doping_and_ions
  poisson_rhs.add(q0, tmp);//0, tmp);//
  laplace_matrix_poisson.vmult(tmp,potential);     //tmp = laplace_matrix_poisson * potential  //No problem, potential treated as const
  poisson_rhs.add(- eps_r * eps_0, tmp);

  poisson_rhs.compress(VectorOperation::insert); 
  laplace_matrix_poisson.compress(VectorOperation::insert); 
  
  // CONDENSATE
  //Condense a given matrix and a given vector by eliminating rows and columns of the linear system that correspond to constrained dof.
  //The sparsity pattern associated with the matrix needs to be condensed and compressed. This function is the appropriate choice for applying inhomogeneous constraints.
  
  // zero_constraints_poisson.condense(system_matrix_poisson, poisson_rhs);
}


// IDEA: MENESSINI DICEVA CHE BISOGNAVA USARE ZERO_CONSTRAIN_POISSON IN NEWTON, NOI NON LO FACCIMAO


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// in this method we solve a linear system and we impose boundary conditions on the result vector. In particular here
// weare computing the update solution of the newton method related to the Poisson problem
template <int dim>
void Problem<dim>::solve_poisson()
{	
  
  SolverControl sc_p;     
  PETScWrappers::SparseDirectMUMPS solverMUMPS(sc_p);     // choice of the solver, MUMPS in this case

  PETScWrappers::MPI::Vector tmp_poisson;
  tmp_poisson.reinit(locally_owned_dofs, mpi_communicator);

  solverMUMPS.solve(system_matrix_poisson, tmp_poisson, poisson_rhs);
  zero_constraints_poisson.distribute(tmp_poisson);
  poisson_newton_update = tmp_poisson;
  
  
  
  // cambi fatti : aggiunto zero constrains e passaggio intermendio con tmp vector
  /*VECCHIO
  SolverControl sc_p;     
  PETScWrappers::SparseDirectMUMPS solverMUMPS(sc_p);     
  solverMUMPS.solve(system_matrix_poisson, poisson_newton_update, poisson_rhs);
  */
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// given the tolerance and the max number of iterations this method solves the Poisson newton system
template <int dim>
void Problem<dim>::newton_iteration_poisson(const double tolerance, const unsigned int max_n_line_searches)
  {
	unsigned int line_search_n = 1;       // number of iterations
	double current_res =  tolerance + 1;  // initial residual to enter the cycle

	while (current_res > tolerance && line_search_n <= max_n_line_searches)
	  {
			assemble_nonlinear_poisson();  // assemble the system related to the current newton iteration

			cout << "assemble_nonlinear_poisson superato " << endl; 

			solve_poisson();               // solve the current newton iteration

			cout << "solve_poisson superato " << endl; 

			// Update Clamping
			const double alpha = 1.;
			cout << "Norm before clamping is " << poisson_newton_update.linfty_norm() << endl;   // compute and print the L inf norm of the solution of the newton iteration

			for (unsigned int i = 0; i < poisson_newton_update.size(); i++) {

				//const PETScWrappers::MPI::Vector temp(poisson_newton_update);    //Ho bisogno di un oggetto const per accedere al metodo () const. !!! Non efficente
				double result; 
				if (poisson_newton_update(i) < -V_E) {
					result = -V_E;
				} else if (poisson_newton_update(i) > V_E) {
					result = V_E;
				} else {
					result = poisson_newton_update(i);
				}

					
				
				poisson_newton_update(i) = result;
				
				old_electron_density(i) *= std::exp(alpha*result/V_E);
				old_ion_density(i) *= std::exp(-alpha*result/V_E);
			}

			old_electron_density.compress(VectorOperation::insert); 
			old_ion_density.compress(VectorOperation::insert); 
			poisson_newton_update.compress(VectorOperation::insert); 

			constraints_poisson.distribute(old_electron_density);      //apply constrains on old_ion_density
			constraints_poisson.distribute(old_ion_density);           //apply constrains on old_electron_density

			PETScWrappers::MPI::Vector tmp3;
			tmp3.reinit(locally_owned_dofs, mpi_communicator);
			tmp3 = potential;
			tmp3.add(alpha, poisson_newton_update);

			tmp3.compress(VectorOperation::add); 
			// potential.add(alpha, poisson_newton_update);  //potential = potential + alpha * poisson_newton_update

			constraints_poisson.distribute(tmp3);    //apply cointraints_poisson to potential vector (vector can't be ghosted)
			potential = tmp3;

			current_res = poisson_newton_update.linfty_norm(); //update the residual as the L inf norm of the newton iteration

			std::cout << "  alpha: " << std::setw(10) << alpha  << std::setw(0) << "  residual: " << current_res  << std::endl; //print out alpha and the residual (after clamping)
			std::cout << "  number of line searches: " << line_search_n << "  residual: " << current_res << std::endl;          //print out the number of iterations

			++line_search_n; //update number of iterations
			//output_results(step_number); // Only needed to see the update at each step during testing
	  }
  }
//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//This method assemble the drift diffusion matrix (da rivedere con tesi riletta)
template <int dim>
void Problem<dim>::assemble_drift_diffusion_matrix() //del singolo processore
{   
  // we initialize as null all the vectors and the matrices of the system
  electron_rhs = 0;
  ion_rhs = 0; 
  drift_diffusion_matrix = 0;
  electron_drift_diffusion_matrix = 0;

  const unsigned int vertices_per_cell = 4;
  std::vector<types::global_dof_index> local_dof_indices(vertices_per_cell);// we create the local_dof_indeces vector. global_dof_index is basically unsigned int

  const unsigned int t_size = 3;
  Vector<double> cell_rhs(t_size);
  FullMatrix<double> A(t_size,t_size), B(t_size,t_size), neg_A(t_size,t_size), neg_B(t_size,t_size);
  std::vector<types::global_dof_index> A_local_dof_indices(t_size);
  std::vector<types::global_dof_index> B_local_dof_indices(t_size);

  for (const auto &cell : dof_handler.active_cell_iterators()) //for each active cell in dof_handler 
    {
	    A = 0;              //initialize local full matrix A to null matrix
	    B = 0;              //initialize local full matrix B to null matrix
	    neg_A = 0;          //initialize local full matrix neg_A to null matrix
		neg_B = 0;          //initialize local full matrix neg_B to null matrix
		cell_rhs = 0;       //initialize local rhs to null

		cell->get_dof_indices(local_dof_indices);  //get the global indeces of the dof of the current active cell (?)

		// Lexicographic ordering
		const Point<dim> v1 = cell->vertex(2); // top left
		const Point<dim> v2 = cell->vertex(3); // top right
		const Point<dim> v3 = cell->vertex(0); // bottom left
		const Point<dim> v4 = cell->vertex(1); // bottom right

		const double u1 = -potential[local_dof_indices[2]]/V_E;   //access to the global position of potential and store this values (?)
		const double u2 = -potential[local_dof_indices[3]]/V_E;
		const double u3 = -potential[local_dof_indices[0]]/V_E;
		const double u4 = -potential[local_dof_indices[1]]/V_E;

		const double l_alpha = side_length(v1,v4);
		const double l_beta = side_length(v2,v3);

		const double alpha21 = (u1 - u2);
		const double alpha42 = (u2 - u4);
		const double alpha34 = (u4 - u3);
		const double alpha13 = (u3 - u1);

		const double neg_alpha21 =  - (u1 - u2);
		const double neg_alpha42 =  - (u2 - u4);
		const double neg_alpha34 = - (u4 - u3);
		const double neg_alpha13 = - (u3 - u1);

		if (l_alpha >= l_beta) { // l_alpha is the longest diagonal: split by beta
					const double alpha23 =  (u3 - u2);
					const double neg_alpha23 = - (u3 - u2);
					//cout << "Alpha 23 is: " << alpha23 << endl;

					// Triangle A:
					A= compute_triangle_matrix(v2,v1,v3, alpha21, alpha13, -alpha23, Dp);
					neg_A= compute_triangle_matrix(v2,v1,v3, neg_alpha21, neg_alpha13, -neg_alpha23, Dn);

					// Triangle B:
					B = compute_triangle_matrix(v3,v4,v2, alpha34, alpha42, alpha23, Dp);
					neg_B = compute_triangle_matrix(v3,v4,v2, neg_alpha34, neg_alpha42, neg_alpha23, Dn);

					// Matrix assemble
					A_local_dof_indices[0] = local_dof_indices[3];  // assign the global indeces (?)
					A_local_dof_indices[1] = local_dof_indices[2];
					A_local_dof_indices[2] = local_dof_indices[0];

					B_local_dof_indices[0] = local_dof_indices[0];
					B_local_dof_indices[1] = local_dof_indices[1];
					B_local_dof_indices[2] = local_dof_indices[3];

				} else { // l_beta is the longest diagonal: split by alpha
					const double alpha14 = (u4 - u1);
					const double neg_alpha14 = - (u4 - u1);
					//cout << "Alpha 14 is: " << alpha14 << endl;

					// Triangle A:
					A = compute_triangle_matrix(v4,v2,v1, alpha42, alpha21, alpha14, Dp);
					neg_A = compute_triangle_matrix(v4,v2,v1, neg_alpha42, neg_alpha21, neg_alpha14, Dn);

					// Triangle B:
					B = compute_triangle_matrix(v1,v3,v4, alpha13, alpha34, -alpha14, Dp);
					neg_B = compute_triangle_matrix(v1,v3,v4, neg_alpha13, neg_alpha34, -neg_alpha14, Dn);

					A_local_dof_indices[0] = local_dof_indices[1];
					A_local_dof_indices[1] = local_dof_indices[3];
					A_local_dof_indices[2] = local_dof_indices[2];

					B_local_dof_indices[0] = local_dof_indices[2];
					B_local_dof_indices[1] = local_dof_indices[0];
					B_local_dof_indices[2] = local_dof_indices[1];

				}

				constraints_ion.distribute_local_to_global(A, cell_rhs,  A_local_dof_indices, drift_diffusion_matrix, ion_rhs);  //This function simultaneously writes elements into the "global" ( inteso come totale del processore) matrix vector, according to the constraints specified by the calling AffineConstraints
				constraints_ion.distribute_local_to_global(B, cell_rhs,  B_local_dof_indices, drift_diffusion_matrix, ion_rhs);  //Same

				constraints_electron.distribute_local_to_global(neg_A, cell_rhs,  A_local_dof_indices, electron_drift_diffusion_matrix, electron_rhs); //Same
				constraints_electron.distribute_local_to_global(neg_B, cell_rhs,  B_local_dof_indices, electron_drift_diffusion_matrix, electron_rhs); //same
		    }
	//std::cout << "ion_rhs norm after distribute_local_to_global: " << ion_rhs.linfty_norm() << std::endl;
	drift_diffusion_matrix.compress(VectorOperation::add);       //Added compress like in other assemble functions
	ion_rhs.compress(VectorOperation::add);

	electron_drift_diffusion_matrix.compress(VectorOperation::add);
	electron_rhs.compress(VectorOperation::add);



	std::cout << "ion_rhs norm after compress " << ion_rhs.linfty_norm() << std::endl;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//This method solve the drift diffusion system with a UMFPACK solver. UMFPACK is a set of routines for solving non-symmetric sparse linear systems.
//This matrix class implements the usual interface of preconditioners,
template <int dim>
void Problem<dim>::solve_drift_diffusion()
{
  SolverControl sc_dd;
  PETScWrappers::SparseDirectMUMPS solverMUMPS_ion(sc_dd);
  PETScWrappers::SparseDirectMUMPS solverMUMPS_electron(sc_dd);

  
  PETScWrappers::MPI::Vector tmp_electron;
  tmp_electron.reinit(locally_owned_dofs, mpi_communicator);

  solverMUMPS_electron.solve(electron_system_matrix, tmp_electron, electron_rhs);
  constraints_electron.distribute(tmp_electron);
  electron_density = tmp_electron;

  PETScWrappers::MPI::Vector tmp_ion;
  tmp_ion.reinit(locally_owned_dofs, mpi_communicator);
  
  solverMUMPS_ion.solve(ion_system_matrix, tmp_ion, ion_rhs);
  constraints_ion.distribute(tmp_ion);
  ion_density = tmp_ion;
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//this method creates one step in the output file; in this output file stores all the solutions with the correct names
template <int dim>
void Problem<dim>::output_results(const unsigned int step)
{
    // DataOut<dim> data_out;
    // data_out.attach_dof_handler(dof_handler);

    // data_out.add_data_vector(ion_density, "Ion_Density");
    // data_out.add_data_vector(electron_density, "Electron_Density");
    // data_out.add_data_vector(potential, "Potential");
    // //data_out.build_patches();

	// Vector<float> subdomain(triangulation.n_active_cells());
    // for (unsigned int i = 0; i < subdomain.size(); ++i)
    // {
    //     subdomain(i) = triangulation.locally_owned_subdomain();           //For distributed parallel triangulations this function returns the subdomain id of those cells that are owned by the current processor
    // }
    // data_out.add_data_vector(subdomain, "subdomain");

	// data_out.build_patches(fe.degree + 1);

	// std::string basename =
    // "ValidationPN" + Utilities::int_to_string(step, 6) + "-";

	// std::string filename =
    // basename +
    // Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    // ".vtu";
    // // DataOutBase::VtkFlags vtk_flags;
    // // vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    // // data_out.set_flags(vtk_flags);
    // std::ofstream output(filename);
    // data_out.write_vtk(output);

	// static std::vector<std::pair<double, std::string>> steps_and_names;
    // if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    // {
    //     for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
    //     {
    //         steps_and_names.push_back(
    //         {step,
    //         basename + Utilities::int_to_string(i, 4) + ".vtu"});
    //     }
    //     std::ofstream pvd_output("ValidationPN.pvd");
    //     DataOutBase::write_pvd_record(pvd_output, steps_and_names);
    // }

	DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(ion_density, "Ion_Density");
    data_out.add_data_vector(electron_density, "Electron_Density");
    data_out.add_data_vector(potential, "Potential");
    data_out.build_patches();

    std::string filename;
    filename = "solution-" + Utilities::int_to_string(step, 3) + ".vtk";
    DataOutBase::VtkFlags vtk_flags;
    vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
    data_out.set_flags(vtk_flags);
    std::ofstream output(filename);
    data_out.write_vtk(output);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//this method is only public method (with the constructor) that is defined in the class. It basically solve the Poisson and the drift diffusion problem
// moreover, it stores the solutions for the post processing
template <int dim>
void Problem<dim>::run()
{   
	pcout << "Running with PETSc on "
        << Utilities::MPI::n_mpi_processes(mpi_communicator)         //Return the number of MPI processes there exist in the given communicator object
        << " MPI rank(s)..." << std::endl;
	
	setup_dofs();
	std::cout << "setup_dofs superato " << std::endl;

    make_constraints_poisson();
	std::cout << "make_constraints_poisson superato " << std::endl;
    make_constraints_drift_diffusion();
	std::cout << "make_constraints_drift_diffusion superato " << std::endl;

    initialize_system_poisson();
	std::cout << "initialize_system_poisson superato " << std::endl;
    initialize_system_drift_diffusion();
	std::cout << "initialize_system_poisson superato " << std::endl;
    
	//INITIALIZE THE VECTORS
	PETScWrappers::MPI::Vector tmp1;
	tmp1.reinit(locally_owned_dofs, mpi_communicator);
	tmp1 = potential;
	VectorTools::interpolate(mapping, dof_handler, PotentialValues<dim>(), tmp1);   //interpolate vuole un vettore non ghosted
	VectorTools::interpolate(mapping, dof_handler, IonInitialValues<dim>(), old_ion_density);
	VectorTools::interpolate(mapping, dof_handler, ElectronInitialValues<dim>(), old_electron_density);
	potential = tmp1;

	cout << "old_ion_density linf norm 1 is " << old_ion_density.linfty_norm() << endl;
  	cout << "old_electron_density linf norm 1 is " << old_electron_density.linfty_norm() << endl;
  	cout << "old_ion_density l2 norm 1 is " << old_ion_density.l2_norm() << endl;
  	cout << "old_electron_density l2 norm 1 is " << old_electron_density.l2_norm() << endl;
	cout << "potential linf norm 1 is " << potential.linfty_norm() << endl;
  	cout << "potential l2 norm 1 is " << potential.l2_norm() << endl;

	// first step in the output
    output_results(0);

    const double tol = 1.e-9*V_E;
    const unsigned int max_it = 50; //max iterations
    
	// set the tollerances
    double ion_tol = 1.e-10;
    double electron_tol = 1.e-10;
    
	// initial error in order to enter the loop
    double ion_err = ion_tol + 1.;
    double electron_err = electron_tol + 1.;

    std::cout << "entrati nel while " << std::endl;
	// SOLVE THE SYSTEM
    while ( (ion_err > ion_tol || electron_err > electron_tol) && step_number < 10)  //time <= max_time - 0.1*timestep
      {
        ++step_number;
		std::cout << "COUPLING STEP: " << step_number << std::endl;

        // Solve Non-Linear Poisson
		newton_iteration_poisson(tol, max_it);
		std::cout << "Usciti da Newton Poisson: " << std::endl;
		//VectorTools::interpolate(mapping, dof_handler, ExactPotentialValues<dim>(), potential);

		// Drift Diffusion Step
        assemble_drift_diffusion_matrix();
		std::cout << "Usciti da assemble_drift_diffusion_matrix: " << std::endl;
		
		ion_system_matrix.copy_from(drift_diffusion_matrix);
		electron_system_matrix.copy_from(electron_drift_diffusion_matrix);
        
        // apply_drift_diffusion_boundary_conditions();
        solve_drift_diffusion();
		std::cout << "Usciti da solve_drift_diffusion: " << std::endl;

        // Update error for convergence
        electron_tol = 1.e-10*old_electron_density.linfty_norm();
        ion_tol = 1.e-10*old_ion_density.linfty_norm();

		std::cout << "norm of ion_density: " << ion_density.linfty_norm() << std::endl;
		std::cout << "norm of old_ion_density: " << old_ion_density.linfty_norm() << std::endl;
		
		PETScWrappers::MPI::Vector tmp;
		tmp.reinit(locally_owned_dofs, mpi_communicator);
        tmp = ion_density;
        tmp -= old_ion_density;
        ion_err = tmp.linfty_norm();
		std::cout << "ion_err: " << ion_err << std::endl;

        tmp = electron_density;
        tmp -= old_electron_density;
        electron_err = tmp.linfty_norm();
        output_results(step_number);
		std::cout << "electron_err: " << electron_err << std::endl;

        old_ion_density = ion_density;
        old_electron_density = electron_density;

    	std::cout << " 	Elapsed CPU time: " << timer.cpu_time() << " seconds.\n" << std::endl << std::endl;
      }

}

//############################ HELPER FUNCTION DEFINITION #########################################################################################################pragma endregion

// This function is used compute_triangle_matrix, another helper function
void bernoulli (double x, double &bp, double &bn)
{
  const double xlim = 1.0e-2;
  double ax  = fabs(x);       // std::fabs() returns the absolute value of a floating point

  bp  = 0.0; // I set both values to zero
  bn  = 0.0;

  //  X=0 CASE
  if (x == 0.0)
    {
      bp = 1.0;
      bn = 1.0;
      return;
    }

  // ASYMPTOTICS CASE ( absolute vale of x greater than 80.0)
  if (ax > 80.0)
    {
      if (x > 0.0) // positive x
        {
          bp = 0.0;
          bn = x;
        }
      else        // negative x
        {
          bp = -x;
          bn = 0.0;
        }
      return;
    }

  // INTERMEDIATE VALUES CASE ( if the absolute value of x il less than 80 and greater than 0.01)
  if (ax <= 80 &&  ax > xlim)
    {
      bp = x / (exp (x) - 1.0);
      bn = x + bp;
      return;
    }

  // SMALL VALUES CASE (absolute value of x less than 0.01 but not null)
  if (ax <= xlim &&  ax != 0.0)
    {
      double jj = 1.0;
      double fp = 1.0;
      double fn = 1.0;
      double df = 1.0;
      double segno = 1.0;
      while (fabs (df) > 1.0e-16)
        {
          jj += 1.0;
          segno = -segno;
          df = df * x / jj;
          fp = fp + df;
          fn = fn + segno * df;
        }
      bp = 1 / fp;
      bn = 1 / fn;
      return;
    }

}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// This helper function is used in assemble_drift_diffusion_matrix, it computes the distance between two 2D Points
double side_length (const Point<2> a, const Point<2> b) 
{
	double length = 0.;

	if (a[0] == b[0])
		length = std::abs(a[1] - b[1]);
	else if (a[1] == b[1])
		length = std::abs(a[0] - b[0]);
	else
		length = std::sqrt(a[0]*a[0] + b[0]*b[0] - 2.*a[0]*b[0] + a[1]*a[1] + b[1]*b[1] - 2.*a[1]*b[1]);

	return length;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// This function is used in compute_triangle_matrix another helper function, it computes the Area of the triagle builed by the passed points
// NB: 0.5 is multiplied in the other helper function 
double triangle_denom(const Point<2> a, const Point<2> b, const Point<2> c)
{
	const double x1 = a[0];
	const double y1 = a[1];

	const double x2 = b[0];
	const double y2 = b[1];

	const double x3 = c[0];
	const double y3 = c[1];

	const double denom = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2);

	return denom;
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// This function is used in compute_trinagle_matrix, another helper function. 
// it return a vector (rank 1 tensor) that habits the 2D plane. It return a normal vector to the face characterized by the two points
Tensor<1,2> face_normal(const Point<2> a, const Point<2> b) { 

	Tensor<1,2> tangent, normal;

	tangent[0] = b[0] - a[0];
	tangent[1] = b[1] - a[1];

	normal[0] = -tangent[1];
	normal[1] = tangent[0];

	return normal;
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// I pass 3 points, 3 + 1 constants double  
// this method create the local full matrix when we are assebling the drift diffusion matrix
FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31, const double D)
{
	const unsigned int size = 3;
	FullMatrix<double> tria_matrix(size,size);

	tria_matrix = 0; //initialize as null matrix
    
	//compute the area of the triangle builded by the arguments points
	const double denom = triangle_denom(a,b,c);
	const double area = 0.5*std::abs(denom);
    

	const Tensor<1,2> grad_psi_1 = face_normal(b,c)/denom;
	const Tensor<1,2> grad_psi_2 = face_normal(c,a)/denom;
	const Tensor<1,2> grad_psi_3 = face_normal(a,b)/denom;

	const double l_12 = grad_psi_1 * grad_psi_2;
	const double l_23 = grad_psi_2 * grad_psi_3;
	const double l_31 = grad_psi_1 * grad_psi_3;

	double bp12, bn12, bp23, bn23, bp31, bn31;

	bernoulli(alpha12,bp12,bn12);
	bernoulli(alpha23,bp23,bn23);
	bernoulli(alpha31,bp31,bn31);

	tria_matrix(0,1) = D * area * bp12 * l_12;
	tria_matrix(0,2) = D * area * bn31 * l_31;

	tria_matrix(1,0) = D * area * bn12 * l_12;
	tria_matrix(1,2) = D * area * bp23 * l_23;

	tria_matrix(2,0) = D * area * bp31 * l_31;
	tria_matrix(2,1) = D * area * bn23 * l_23;
	
	tria_matrix(0,0) = - (tria_matrix(1,0)+tria_matrix(2,0));
	tria_matrix(1,1) = - (tria_matrix(0,1)+tria_matrix(2,1));
	tria_matrix(2,2) = - (tria_matrix(0,2)+tria_matrix(1,2));

	return tria_matrix;
}

//NB queste ultime helper function utilizzano tutte punti senza reference, forse bisognrebbe metterle in modo tale da
// eliminare copie inutili