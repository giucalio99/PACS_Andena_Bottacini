

using namespace dealii;
using namespace std;

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CONSTRUCTOR
template <int dim>
Problem<dim>::Problem()
  : fe(1)                       // linear elements, we approximate our variables linearly on the elements
  , dof_handler(triangulation)  // !!!!AAAAcome fa a funzionare se triangulation non è inizializzata ? questo è l'unico constructor
  , step_number(0)
  , mapping()                   // initialize mapping 2D
{}

// since only these data members are initialized in the constructor, all the other members will be initialized by their own default constructor

//---------------------------------------------------------------------------------------------------------------------------------------------------------------

// This method create, or import, the mesh over which we perform the simulation. Moreover it sets the unique tag of the boundary
template <int dim>
void Problem<dim>::create_mesh()
{
    const Point<dim> bottom_left(0.,-L/20.);  //bottom left Point of the rectangular mesh
	const Point<dim> top_right(L,L/20.);      //top right Point of the rectangular domain
    
	// NB: these points are needed if you want to define a mesh starting from deal ii, otherwise use an input file as follows:

	// For a structured mesh (**commento di menessini**)
	//GridGenerator::subdivided_hyper_rectangle(triangulation, {100,}, bottom_left, top_right);

    // we read from input file the mesh already generated
	const std::string filename = "./Meshes/small_square.msh"; //name of the .msh file
	ifstream input_file(filename); //ATTENZIONE, PERCHè NON CE OPEN?
	cout << "Reading from " << filename << endl; //screen comment
	GridIn<2>       grid_in; //This class implements an input mechanism for grid data. It allows to read a grid structure into a triangulation object
	grid_in.attach_triangulation(triangulation); //we pass to grid_in our (empty) triangulation
	grid_in.read_msh(input_file); // read the msh file



    //now we identify the boundaries of the mesh
	for (auto &face : triangulation.active_face_iterators())    // we look only at active face, namely only at those faces that carry dofs, these cells are not father of other sub cells
	{
	  if (face->at_boundary())  // if we are looking at a face on the boundary (in our 2D case and edge on the boundary). at_boundary() return true-false values
	  {
		  face->set_boundary_id(0);             // we associate this edge with the unique ID zero (this will be the ID of the boundary faces on up/bottom boundaries)
		  const Point<dim> c = face->center();  // we compute the geometrical center of the edge that is stored in the face iterator

		  	  if ( c[1] < top_right[1] && c[1] > bottom_left[1]) { // if the y coordinate of the Point c is less than TR point and grater than BL point (basically if we are on the left/right boundary)

		  		  if (c[0] < (top_right[0] + bottom_left[0])/2.) { // if we are on the left boundary edge
		  			  face->set_boundary_id(1); //ID of the boundary faces on the left 
		  		  } else
		  			face->set_boundary_id(2);  //ID of the boundary faces on the right
		  	  }
		  }
	}

	 triangulation.refine_global(3); //globally refine the mesh 3 times, namely each cells is subdivided into 3 more cells--> increase mesh resolution
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// This method set up the Poissson problem. In particular it enumerates the dof on the given mesh, it sets the constraints and the boundary conditions of the problem
// it defines the sparsity patterns and initialize the matrices and the vecotrs of the problem.
template <int dim>
void Problem<dim>::setup_poisson()
{   
	// INITIALIZATION OF THE DOF AND VECTORS
	dof_handler.distribute_dofs(fe);  //Go through the triangulation and "distribute" the dof needed for the given finite element. We also enumerate them whereever they are (internal or boundary)

	potential.reinit(dof_handler.n_dofs()); // We resize the dimension of the vector "potential" to the number of the dof (unsigned int)
	poisson_newton_update.reinit(dof_handler.n_dofs()); //same
    
	// CONSTRAIN PROBLEM AND BOUNDARY VALUES
	constraints_poisson.clear();  // Reset the flag determining whether new entries are accepted or not
	DoFTools::make_hanging_node_constraints(dof_handler, constraints_poisson); // Compute the constraints resulting from the presence of hanging nodes. We put the result in constraints_poisson
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(V_E*log(N1/N_0)), constraints_poisson); // Set the values of constraints_poisson on the left edge boundary of the domain
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(V_E*log(N2/N_0)), constraints_poisson); // Set the values of constraints_poisson on the right edge boundary of the domain
	constraints_poisson.close();  // Close the filling of entries and sort the lines and columns

	// Used for the update term in Newton's method
	zero_constraints_poisson.clear();  // As before but on zero_constraints_poisson
	DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints_poisson); // Set the values of zero_constraints_poisson on the left edge boundary of the domain to zero
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints_poisson); // Set the values of zero_constraints_poisson on the right edge boundary of the domain to zero (homo Dirichlet)
	zero_constraints_poisson.close();
    
	// SPARSITY PATTERN
	DynamicSparsityPattern dsp(dof_handler.n_dofs()); // it mostly represents a SparsityPattern object that is kept compressed at all times, memory reason. We initialize a square pattern of size n_dof
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_poisson, true); // Compute which entries of a matrix built on the given dof_handler may possibly be nonzero, and create a sparsity pattern object that represents these nonzero locations.
    
	// at this point we have in dsp the pattern related to the problem 
    sparsity_pattern_poisson.copy_from(dsp); // we copy in sparsity_pattern_poisson the pattern created before
    
    // INITIALIZATION OF SYSTEM MATRICES AND RHS
	system_matrix_poisson.reinit(sparsity_pattern_poisson); // Reinitialize the sparse matrix with the given sparsity pattern. The latter tells the matrix how many nonzero elements there need to be reserved.
    laplace_matrix_poisson.reinit(sparsity_pattern_poisson);// As above
	mass_matrix_poisson.reinit(sparsity_pattern_poisson);   // As above

	MatrixCreator::create_laplace_matrix(mapping, dof_handler, QTrapezoid<dim>(), laplace_matrix_poisson); // Assemble the Laplace matrix with trapezoidal rule for numerical quadrature
	MatrixCreator::create_mass_matrix(mapping, dof_handler, QTrapezoid<dim>(), mass_matrix_poisson);       // Assemble the mass matrix with trapezoidal rule for numerical quadrature

	poisson_rhs.reinit(dof_handler.n_dofs()); // initialize the rhs vector
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

// This method is called in the newton_iteration_poisson method (NON CAPISCO MOLTO A CHE PUNTO DELL 'ALGO SIAMO)
// It defines one iteration of the newton algorithm (?)
template <int dim>
void Problem<dim>::assemble_nonlinear_poisson()
{

  // ASSEMBLE MATRICES
  system_matrix_poisson = 0; //sets all elements of the matrix to zero, but keep the sparsity pattern previously used.
  SparseMatrix<double> ion_mass_matrix(sparsity_pattern_poisson); // We initialized the new ion_mass_matrix with our sparsity pattern
  ion_mass_matrix = 0; // and then set all the elments to zero
 
  // compute the ion mass matrix
  for (unsigned int i = 0; i < old_ion_density.size(); ++i){
	  ion_mass_matrix(i,i) = mass_matrix_poisson(i,i) * (old_ion_density(i) + old_electron_density(i));
  }

  system_matrix_poisson.add(q0 / V_E, ion_mass_matrix);  // A += factor * B with the passed values
  cout << "Ion matrix norm is " << system_matrix_poisson.linfty_norm() << endl;  // compute and print the infinit norm of the matrix

  system_matrix_poisson.add(eps_r * eps_0, laplace_matrix_poisson); // same as above
  cout << "Matrix norm is " << system_matrix_poisson.linfty_norm() << endl;

  // ASSEMBLE RHS
  poisson_rhs = 0; // set all the values to zero
  Vector<double> tmp(dof_handler.n_dofs()); //temporary vector of dimension n_dof
  Vector<double> doping_and_ions(dof_handler.n_dofs()); // create a new vector of dimension n_dof
  VectorTools::interpolate(mapping,dof_handler, DopingValues<dim>(), doping_and_ions); // We interpolate the previusly created vector with the initial values of Doping provided by DopingValues

  doping_and_ions -= old_electron_density;
  doping_and_ions += old_ion_density;

  mass_matrix_poisson.vmult(tmp,doping_and_ions);
  poisson_rhs.add(q0, tmp);//0, tmp);//
  laplace_matrix_poisson.vmult(tmp,potential);
  poisson_rhs.add(- eps_r * eps_0, tmp);//- eps_r * eps_0, tmp);//
  
  // CONDENSATE
  //Condense a given matrix and a given vector by eliminating rows and columns of the linear system that correspond to constrained dof.
  //The sparsity pattern associated with the matrix needs to be condensed and compressed. This function is the appropriate choice for applying inhomogeneous constraints.
  
  zero_constraints_poisson.condense(system_matrix_poisson, poisson_rhs);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::solve_poisson()
{
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix_poisson);
  A_direct.vmult(poisson_newton_update, poisson_rhs);

  zero_constraints_poisson.distribute(poisson_newton_update);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione 
template <int dim>
void Problem<dim>::newton_iteration_poisson(const double tolerance, const unsigned int max_n_line_searches)
  {
	unsigned int line_search_n = 1;
	double current_res =  tolerance + 1;

	while (current_res > tolerance && line_search_n <= max_n_line_searches)
	  {
			assemble_nonlinear_poisson();
			solve_poisson();

			// Update Clamping
			const double alpha = 1.;
			cout << "Norm before clamping is " << poisson_newton_update.linfty_norm() << endl;
			for (unsigned int i = 0; i < poisson_newton_update.size(); i++) {
				poisson_newton_update(i) = std::max(std::min(poisson_newton_update(i),V_E),-V_E);

				old_electron_density(i) *= std::exp(alpha*poisson_newton_update(i)/V_E);
				old_ion_density(i) *= std::exp(-alpha*poisson_newton_update(i)/V_E);
			}
			constraints.distribute(old_ion_density);
			constraints.distribute(old_electron_density);

			potential.add(alpha, poisson_newton_update);
			constraints_poisson.distribute(potential);

			current_res = poisson_newton_update.linfty_norm();

			std::cout << "  alpha: " << std::setw(10) << alpha  << std::setw(0) << "  residual: " << current_res  << std::endl;
			std::cout << "  number of line searches: " << line_search_n << "  residual: " << current_res << std::endl;

			++line_search_n;
			//output_results(step_number); // Only needed to see the update at each step during testing
	  }
  }
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::setup_drift_diffusion()
{
	ion_density.reinit(dof_handler.n_dofs());
	electron_density.reinit(dof_handler.n_dofs());

	old_ion_density.reinit(dof_handler.n_dofs());
	old_electron_density.reinit(dof_handler.n_dofs());

	constraints.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints);
	constraints.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
	sparsity_pattern.copy_from(dsp);

	ion_rhs.reinit(dof_handler.n_dofs());
	electron_rhs.reinit(dof_handler.n_dofs());

	ion_system_matrix.reinit(sparsity_pattern);
	electron_system_matrix.reinit(sparsity_pattern);

	drift_diffusion_matrix.reinit(sparsity_pattern);
	electron_drift_diffusion_matrix.reinit(sparsity_pattern);
}

//------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::assemble_drift_diffusion_matrix()
{
	electron_rhs = 0;
	ion_rhs = 0;
	drift_diffusion_matrix = 0;
	electron_drift_diffusion_matrix = 0;

  const unsigned int vertices_per_cell = 4;
  std::vector<types::global_dof_index> local_dof_indices(vertices_per_cell);

  const unsigned int t_size = 3;
  Vector<double> cell_rhs(t_size);
  FullMatrix<double> A(t_size,t_size), B(t_size,t_size), neg_A(t_size,t_size), neg_B(t_size,t_size);
  std::vector<types::global_dof_index> A_local_dof_indices(t_size);
  std::vector<types::global_dof_index> B_local_dof_indices(t_size);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
	    A = 0;
	    B = 0;
	    neg_A = 0;
		neg_B = 0;
		cell_rhs = 0;
		cell->get_dof_indices(local_dof_indices);

		// Lexicographic ordering
		const Point<dim> v1 = cell->vertex(2); // top left
		const Point<dim> v2 = cell->vertex(3); // top right
		const Point<dim> v3 = cell->vertex(0); // bottom left
		const Point<dim> v4 = cell->vertex(1); // bottom right

		const double u1 = -potential[local_dof_indices[2]]/V_E;
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
					A_local_dof_indices[0] = local_dof_indices[3];
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

				constraints.distribute_local_to_global(A, cell_rhs,  A_local_dof_indices, drift_diffusion_matrix,ion_rhs);
				constraints.distribute_local_to_global(B, cell_rhs,  B_local_dof_indices, drift_diffusion_matrix, ion_rhs);

				constraints.distribute_local_to_global(neg_A, cell_rhs,  A_local_dof_indices, electron_drift_diffusion_matrix, electron_rhs);
				constraints.distribute_local_to_global(neg_B, cell_rhs,  B_local_dof_indices, electron_drift_diffusion_matrix, electron_rhs);
		    }


}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::apply_drift_diffusion_boundary_conditions()
{
	    std::map<types::global_dof_index, double> emitter_boundary_values, collector_boundary_values;

		VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(N1), emitter_boundary_values);
	    MatrixTools::apply_boundary_values(emitter_boundary_values, electron_system_matrix, electron_density, electron_rhs);

		VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(N2), collector_boundary_values);
		MatrixTools::apply_boundary_values(collector_boundary_values, electron_system_matrix, electron_density, electron_rhs);

		VectorTools::interpolate_boundary_values(mapping, dof_handler,1, Functions::ConstantFunction<dim>(P1), emitter_boundary_values);
		MatrixTools::apply_boundary_values(emitter_boundary_values, ion_system_matrix, ion_density, ion_rhs);

		VectorTools::interpolate_boundary_values(mapping, dof_handler,2, Functions::ConstantFunction<dim>(P2), collector_boundary_values);
		MatrixTools::apply_boundary_values(collector_boundary_values, ion_system_matrix, ion_density, ion_rhs);
 }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::solve_drift_diffusion()
{
  SparseDirectUMFPACK P_direct;
  P_direct.initialize(ion_system_matrix);
  P_direct.vmult(ion_density, ion_rhs);
  constraints.distribute(ion_density);

  SparseDirectUMFPACK N_direct;
  N_direct.initialize(electron_system_matrix);
  N_direct.vmult(electron_density, electron_rhs);
  constraints.distribute(electron_density);
}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::output_results(const unsigned int step)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(old_ion_density, "Old_Ion_Density");
    data_out.add_data_vector(old_electron_density, "Old_Electron_Density");
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

//descrizione
template <int dim>
void Problem<dim>::run()
{
	create_mesh();

	setup_poisson();
	setup_drift_diffusion();

	VectorTools::interpolate(mapping, dof_handler, PotentialValues<dim>(), potential);
	VectorTools::interpolate(mapping, dof_handler, IonInitialValues<dim>(), old_ion_density);
	VectorTools::interpolate(mapping, dof_handler, ElectronInitialValues<dim>(), old_electron_density);

    output_results(0);

    Vector<double> tmp(ion_density.size());

    const double tol = 1.e-9*V_E;
    const unsigned int max_it = 50;

    double ion_tol = 1.e-10;
    double electron_tol = 1.e-10;

    double ion_err = ion_tol + 1.;
    double electron_err = electron_tol + 1.;

    while ( (ion_err > ion_tol || electron_err > electron_tol) && step_number < 10)  //time <= max_time - 0.1*timestep
      {
        ++step_number;

        // Solve Non-Linear Poisson
		newton_iteration_poisson(tol, max_it);

		//VectorTools::interpolate(mapping, dof_handler, ExactPotentialValues<dim>(), potential);

		// Drift Diffusion Step
        assemble_drift_diffusion_matrix();

		ion_system_matrix.copy_from(drift_diffusion_matrix);
		electron_system_matrix.copy_from(electron_drift_diffusion_matrix);
        
		// COMMENTO MENESSINI
		// BE integration
        /*mass_matrix.vmult(ion_rhs, old_ion_density);
        mass_matrix.vmult(electron_rhs, old_electron_density);
        assemble_drift_diffusion_matrix();

        RightHandSide<dim> rhs_function;
        rhs_function.set_time(time);
        VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), rhs_function, tmp);
        forcing_terms = tmp;
        forcing_terms *= timestep;

        ion_rhs += forcing_terms;
        ion_system_matrix.copy_from(mass_matrix);
        ion_system_matrix.add(timestep, drift_diffusion_matrix);

        electron_rhs += forcing_terms;
        electron_system_matrix.copy_from(mass_matrix);
        electron_system_matrix.add(timestep, negative_drift_diffusion_matrix);*/

        apply_drift_diffusion_boundary_conditions();
        solve_drift_diffusion();

        // Update error for convergence
        electron_tol = 1.e-10*old_electron_density.linfty_norm();
        ion_tol = 1.e-10*old_ion_density.linfty_norm();

        tmp = ion_density;
        tmp -= old_ion_density;
        ion_err = tmp.linfty_norm();

        tmp = electron_density;
        tmp -= old_electron_density;
        electron_err = tmp.linfty_norm();

        output_results(step_number);

        old_ion_density = ion_density;
        old_electron_density = electron_density;

    	std::cout << " 	Elapsed CPU time: " << timer.cpu_time() << " seconds.\n" << std::endl << std::endl;
      }

}

//############################ HELPER FUNCTION DEFINITION #########################################################################################################pragma endregion

// description
void bernoulli (double x, double &bp, double &bn)  // used in compute_triangle_matrix
{
  const double xlim = 1.0e-2;
  double ax  = fabs(x);       // std::fabs() returns the absolute value of a floating point

  bp  = 0.0;
  bn  = 0.0;

  //  X=0
  if (x == 0.0)
    {
      bp = 1.0;
      bn = 1.0;
      return;
    }

  // ASYMPTOTICS
  if (ax > 80.0)
    {
      if (x > 0.0)
        {
          bp = 0.0;
          bn = x;
        }
      else
        {
          bp = -x;
          bn = 0.0;
        }
      return;
    }

  // INTERMEDIATE VALUES
  if (ax <= 80 &&  ax > xlim)
    {
      bp = x / (exp (x) - 1.0);
      bn = x + bp;
      return;
    }

  // SMALL VALUES
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

//description
double side_length (const Point<2> a, const Point<2> b) // used in Problem<dim>::assemble_drift_diffusion_matrix()
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

//description 
double triangle_denom(const Point<2> a, const Point<2> b, const Point<2> c)//used in compute_triangle_matrix
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

// description
Tensor<1,2> face_normal(const Point<2> a, const Point<2> b) {  //used in compute_triangle_matrix

	Tensor<1,2> tangent, normal;

	tangent[0] = b[0] - a[0];
	tangent[1] = b[1] - a[1];

	normal[0] = -tangent[1];
	normal[1] = tangent[0];

	return normal;
}

// -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//description
FullMatrix<double> compute_triangle_matrix(const Point<2> a, const Point<2> b, const Point<2> c, const double alpha12, const double alpha23, const double alpha31, const double D)
{
	const unsigned int size = 3;
	FullMatrix<double> tria_matrix(size,size);

	tria_matrix = 0;

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