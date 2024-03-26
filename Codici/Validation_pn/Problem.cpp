#include "Problem.hpp"

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CONSTRUCTOR
template <int dim>
Problem<dim>::Problem()
  : fe(1) // linear elements
  , dof_handler(triangulation)
  , step_number(0)
  , mapping()
{}

//---------------------------------------------------------------------------------------------------------------------------------------------------------------

// descrizione 
template <int dim>
void Problem<dim>::create_mesh()
{
    const Point<dim> bottom_left(0.,-L/20.);
	const Point<dim> top_right(L,L/20.);

	// For a structured mesh
	//GridGenerator::subdivided_hyper_rectangle(triangulation, {100,}, bottom_left, top_right);

	const std::string filename = "./Meshes/small_square.msh";
		ifstream input_file(filename);
		cout << "Reading from " << filename << endl;
	    GridIn<2>       grid_in;
	    grid_in.attach_triangulation(triangulation);
	    grid_in.read_msh(input_file);

	  for (auto &face : triangulation.active_face_iterators())
	  {
		  if (face->at_boundary())
		  {
			  face->set_boundary_id(0);
			  const Point<dim> c = face->center();

			  	  if ( c[1] < top_right[1] && c[1] > bottom_left[1]) {

			  		  if (c[0] < (top_right[0] + bottom_left[0])/2.) {
			  			  face->set_boundary_id(1);
			  		  } else
			  			face->set_boundary_id(2);
			  	  }

			  }
		}

	 triangulation.refine_global(3);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//descrizione
template <int dim>
void Problem<dim>::setup_poisson()
{
	dof_handler.distribute_dofs(fe);

	potential.reinit(dof_handler.n_dofs());
	poisson_newton_update.reinit(dof_handler.n_dofs());

	constraints_poisson.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ConstantFunction<dim>(V_E*log(N1/N_0)), constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ConstantFunction<dim>(V_E*log(N2/N_0)), constraints_poisson);
	constraints_poisson.close();

	// Used for the update term in Newton's method
	zero_constraints_poisson.clear();
	DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 1, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
	VectorTools::interpolate_boundary_values(dof_handler, 2, Functions::ZeroFunction<dim>(), zero_constraints_poisson);
	zero_constraints_poisson.close();

	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints_poisson, true);

	sparsity_pattern_poisson.copy_from(dsp);

	system_matrix_poisson.reinit(sparsity_pattern_poisson);

	laplace_matrix_poisson.reinit(sparsity_pattern_poisson);
	MatrixCreator::create_laplace_matrix(mapping, dof_handler, QTrapezoid<dim>(), laplace_matrix_poisson);
	mass_matrix_poisson.reinit(sparsity_pattern_poisson);
	MatrixCreator::create_mass_matrix(mapping, dof_handler, QTrapezoid<dim>(), mass_matrix_poisson);

	poisson_rhs.reinit(dof_handler.n_dofs());
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

//descizione
template <int dim>
void Problem<dim>::assemble_nonlinear_poisson()
{
  // Assemble mattrix
  system_matrix_poisson = 0;

  SparseMatrix<double> ion_mass_matrix(sparsity_pattern_poisson);
  ion_mass_matrix = 0;

  for (unsigned int i = 0; i < old_ion_density.size(); ++i)
	  ion_mass_matrix(i,i) = mass_matrix_poisson(i,i) * (old_ion_density(i) + old_electron_density(i));

  system_matrix_poisson.add(q0 / V_E, ion_mass_matrix);
  cout << "Ion matrix norm is " << system_matrix_poisson.linfty_norm() << endl;
  system_matrix_poisson.add(eps_r * eps_0, laplace_matrix_poisson);

  cout << "Matrix norm is " << system_matrix_poisson.linfty_norm() << endl;

  // Assmble RHS
  poisson_rhs = 0;
    Vector<double> tmp(dof_handler.n_dofs());
    Vector<double> doping_and_ions(dof_handler.n_dofs());
    VectorTools::interpolate(mapping,dof_handler, DopingValues<dim>(), doping_and_ions);

    doping_and_ions -= old_electron_density;
    doping_and_ions += old_ion_density;

    mass_matrix_poisson.vmult(tmp,doping_and_ions);
    poisson_rhs.add(q0, tmp);//0, tmp);//
    laplace_matrix_poisson.vmult(tmp,potential);
    poisson_rhs.add(- eps_r * eps_0, tmp);//- eps_r * eps_0, tmp);//

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

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------