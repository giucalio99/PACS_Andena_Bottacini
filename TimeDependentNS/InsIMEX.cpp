#include "InsIMEX.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <sstream>


// @sect4{InsIMEX::InsIMEX}
template <int dim>
InsIMEX<dim>::InsIMEX(parallel::distributed::Triangulation<dim> &tria)
    : viscosity(0.001),
    gamma(0.1),
    degree(1),
    triangulation(tria),
    fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1),
    dof_handler(triangulation),            //Stores a list of degrees of freedom
    volume_quad_formula(degree + 2),
    face_quad_formula(degree + 2),
    mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
    time(1e0, 1e-3, 1e-2, 1e-2),
    timer(
        mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
{
}

// @sect4{InsIMEX::setup_dofs}
template <int dim>
void InsIMEX<dim>::setup_dofs()
{
    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs(fe);     //Distribute the dof needed for the given fe on my triangulation    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    DoFRenumbering::Cuthill_McKee(dof_handler);                //Renumber the degrees of freedom according to the Cuthill-McKee method. FUnction from numspace DofREnumbering
    std::vector<unsigned int> block_component(dim + 1, 0);     //Musk for the reording
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);    
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);    //Block: in our case are two, one with dimension = dim and one with dimension = 1
    // Partitioning.
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];
    owned_partitioning.resize(2);
    owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, dof_u);      //Extract the set of locally owned DoF indices for each component within the mask that are owned by the current processor.
    owned_partitioning[1] =
    dof_handler.locally_owned_dofs().get_view(dof_u, dof_u + dof_p);
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);     //Extract the set of global DoF indices that are active on the current DoFHandler.
    relevant_partitioning.resize(2);
    relevant_partitioning[0] = locally_relevant_dofs.get_view(0, dof_u);
    relevant_partitioning[1] =
    locally_relevant_dofs.get_view(dof_u, dof_u + dof_p);
    pcout << "   Number of active fluid cells: "
        << triangulation.n_global_active_cells() << std::endl
        << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
        << dof_u << '+' << dof_p << ')' << std::endl;
}

// @sect4{InsIMEX::make_constraints}
template <int dim>
void InsIMEX<dim>::make_constraints()
{
    // Because the equation is written in incremental form, two constraints
    // are needed: nonzero constraint and zero constraint.
    nonzero_constraints.clear();
    zero_constraints.clear();
    nonzero_constraints.reinit(locally_relevant_dofs);      //clear() the AffineConstraints object and supply an IndexSet with lines that may be constrained
    zero_constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);      //Necessary when work with not homogeneous refinment 
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);

    // Apply Dirichlet boundary conditions on all boundaries except for the
    // outlet.
    std::vector<unsigned int> dirichlet_bc_ids;                        //To change it                           
    if (dim == 2)
    dirichlet_bc_ids = std::vector<unsigned int>{0, 2, 3, 4};                
    else
    dirichlet_bc_ids = std::vector<unsigned int>{0, 2, 3, 4, 5, 6};

    FEValuesExtractors::Vector velocities(0);
    for (auto id : dirichlet_bc_ids)
    {
        VectorTools::interpolate_boundary_values(dof_handler,               //Assign B.C. different from zero in nonzero_constraints
                                                id,
                                                BoundaryValues<dim>(),
                                                nonzero_constraints,
                                                fe.component_mask(velocities));
        VectorTools::interpolate_boundary_values(
        dof_handler,
        id,
        Functions::ZeroFunction<dim>(dim + 1),
        zero_constraints,
        fe.component_mask(velocities));
    }
    nonzero_constraints.close();        //After closing, no more entries are accepted
    zero_constraints.close();
}

// @sect4{InsIMEX::initialize_system}
template <int dim>
void InsIMEX<dim>::initialize_system()
{
    preconditioner.reset();
    system_matrix.clear();
    mass_matrix.clear();
    mass_schur.clear();

    BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);            //This class implements an array of compressed sparsity patterns that can be used to initialize objects of type BlockSparsityPattern.
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);     //Assign to dsp the sparsity pattern using the locality of fe basis functions
    sparsity_pattern.copy_from(dsp);                                            //Copy data from an object of type BlockDynamicSparsityPattern
    SparsityTools::distribute_sparsity_pattern(             //Communicate rows in a dynamic sparsity pattern over MPI.
    dsp,
    dof_handler.locally_owned_dofs(),                     //An IndexSet describing the rows owned by the calling MPI process. 
    mpi_communicator,
    locally_relevant_dofs);

    system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);         //Efficiently reinit the block matrix for a parallel computation
    mass_matrix.reinit(owned_partitioning, dsp, mpi_communicator);

    // Only the $(1, 1)$ block in the mass schur matrix is used. (B M_u B.T)
    // Compute the sparsity pattern for mass schur in advance.
    // The only nonzero block has the same sparsity pattern as $BB^T$.
    BlockDynamicSparsityPattern schur_dsp(dofs_per_block, dofs_per_block);
    schur_dsp.block(1, 1).compute_mmult_pattern(sparsity_pattern.block(1, 0),
                                                sparsity_pattern.block(0, 1));
    mass_schur.reinit(owned_partitioning, schur_dsp, mpi_communicator);

    // present_solution is ghosted because it is used in the
    // output and mesh refinement functions.
    present_solution.reinit(
    owned_partitioning, relevant_partitioning, mpi_communicator);
    // solution_increment is non-ghosted because the linear solver needs
    // a completely distributed vector.
    solution_increment.reinit(owned_partitioning, mpi_communicator);
    // system_rhs is non-ghosted because it is only used in the linear
    // solver and residual evaluation.
    system_rhs.reinit(owned_partitioning, mpi_communicator);
}

// @sect4{InsIMEX::assemble}
//
// Assemble the system matrix, mass matrix, and the RHS.
// It can be used to assemble the entire system or only the RHS.
// An additional option is added to determine whether nonzero
// constraints or zero constraints should be used.
// Note that we only need to assemble the LHS for twice: once with the nonzero
// constraint
// and once for zero constraint. But we must assemble the RHS at every time
// step.
template <int dim>
void InsIMEX<dim>::assemble(bool use_nonzero_constraints,
                            bool assemble_system)
{
    TimerOutput::Scope timer_section(timer, "Assemble system");        //Enter the given section in the timer

    if (assemble_system)           //If assemble_system is false i assembly only rhs
    {
        system_matrix = 0;
        mass_matrix = 0;
    }
    system_rhs = 0;

    FEValues<dim> fe_values(fe,                              //It implicitely uses a Q1 mapping
                            volume_quad_formula,
                            update_values | update_quadrature_points |
                            update_JxW_values | update_gradients);
    FEFaceValues<dim> fe_face_values(fe,
                                    face_quad_formula,
                                    update_values | update_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = volume_quad_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);                 //Wht stored in vector of Tensors?
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
    std::vector<double> current_velocity_divergences(n_q_points);
    std::vector<double> current_pressure_values(n_q_points);

    std::vector<double> div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double> phi_p(dofs_per_cell);

    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();      //Iterator from the first active cell to the lasto one
        ++cell)
    {
        if (cell->is_locally_owned())
        {
            fe_values.reinit(cell);

            if (assemble_system)
            {
                local_matrix = 0;
                local_mass_matrix = 0;
            }
            local_rhs = 0;

            fe_values[velocities].get_function_values(present_solution,
                                                    current_velocity_values);     //in current_velocity_values i stored values of present_solution in quadrature points

            fe_values[velocities].get_function_gradients(
            present_solution, current_velocity_gradients);

            fe_values[velocities].get_function_divergences(
            present_solution, current_velocity_divergences);

            fe_values[pressure].get_function_values(present_solution,
                                                    current_pressure_values);

            // Assemble the system matrix and mass matrix simultaneouly.
            // The mass matrix only uses the $(0, 0)$ and $(1, 1)$ blocks.
            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    div_phi_u[k] = fe_values[velocities].divergence(k, q);           //I assign the values in the dofs of the cell
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                    phi_u[k] = fe_values[velocities].value(k, q);
                    phi_p[k] = fe_values[pressure].value(k, q);
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    if (assemble_system)
                    {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) +=
                            (viscosity *
                                scalar_product(grad_phi_u[j], grad_phi_u[i]) -
                            div_phi_u[i] * phi_p[j] -
                            phi_p[i] * div_phi_u[j] +
                            gamma * div_phi_u[j] * div_phi_u[i] +
                            phi_u[i] * phi_u[j] / time.get_delta_t()) *
                            fe_values.JxW(q);
                            local_mass_matrix(i, j) +=
                            (phi_u[i] * phi_u[j] + phi_p[i] * phi_p[j]) *
                            fe_values.JxW(q);
                        }
                    }
                    local_rhs(i) -=
                    (viscosity * scalar_product(current_velocity_gradients[q],
                                                grad_phi_u[i]) -
                    current_velocity_divergences[q] * phi_p[i] -
                    current_pressure_values[q] * div_phi_u[i] +
                    gamma * current_velocity_divergences[q] * div_phi_u[i] +
                    current_velocity_gradients[q] *
                        current_velocity_values[q] * phi_u[i]) *
                    fe_values.JxW(q);
                }
            }

            cell->get_dof_indices(local_dof_indices);

            const AffineConstraints<double> &constraints_used =
            use_nonzero_constraints ? nonzero_constraints : zero_constraints;
            if (assemble_system)
            {
                constraints_used.distribute_local_to_global(local_matrix,             //In practice this function implements a scatter operation
                                                            local_rhs,
                                                            local_dof_indices,        //Contains the corresponding global indexes
                                                            system_matrix,
                                                            system_rhs);
                constraints_used.distribute_local_to_global(
                local_mass_matrix, local_dof_indices, mass_matrix);
            }
            else
            {
                constraints_used.distribute_local_to_global(
                local_rhs, local_dof_indices, system_rhs);
            }
        }
    }

    if (assemble_system)
    {
        system_matrix.compress(VectorOperation::add);
        mass_matrix.compress(VectorOperation::add);
    }
    system_rhs.compress(VectorOperation::add);
}

// @sect4{InsIMEX::solve}
// Solve the linear system using FGMRES solver with block preconditioner.
// After solving the linear system, the same AffineConstraints object as used
// in assembly must be used again, to set the constrained value.
// The second argument is used to determine whether the block
// preconditioner should be reset or not.
template <int dim>
std::pair<unsigned int, double>
InsIMEX<dim>::solve(bool use_nonzero_constraints, bool assemble_system)
{
    if (assemble_system)
    {
        preconditioner.reset(new BlockSchurPreconditioner(timer,
                                                        gamma,
                                                        viscosity,
                                                        time.get_delta_t(),
                                                        owned_partitioning,
                                                        system_matrix,
                                                        mass_matrix,
                                                        mass_schur));
    }

    SolverControl solver_control(                                         //Used by iterative methods to determine whether the iteration should be continued
    system_matrix.m(), 1e-8 * system_rhs.l2_norm(), true);
    // Because PETScWrappers::SolverGMRES only accepts preconditioner
    // derived from PETScWrappers::PreconditionBase,
    // we use dealii SolverFGMRES.
    GrowingVectorMemory<PETScWrappers::MPI::BlockVector> vector_memory;
    SolverFGMRES<PETScWrappers::MPI::BlockVector> gmres(solver_control,
                                                        vector_memory);

    // The solution vector must be non-ghosted
    gmres.solve(system_matrix, solution_increment, system_rhs, *preconditioner);

    const AffineConstraints<double> &constraints_used =
    use_nonzero_constraints ? nonzero_constraints : zero_constraints;
    constraints_used.distribute(solution_increment);

    return {solver_control.last_step(), solver_control.last_value()};
}

// @sect4{InsIMEX::run}
template <int dim>
void InsIMEX<dim>::run()
{
    pcout << "Running with PETSc on "
        << Utilities::MPI::n_mpi_processes(mpi_communicator)         //Return the number of MPI processes there exist in the given communicator object
        << " MPI rank(s)..." << std::endl;

    triangulation.refine_global(0);
    setup_dofs();
    make_constraints();
    initialize_system();

    // Time loop.
    bool refined = false;
    while (time.end() - time.current() > 1e-12)          //We are not still at the end
    {
        if (time.get_timestep() == 0)
        {
            output_results(0);
        }
        time.increment();
        std::cout.precision(6);                        //This means that when floating-point numbers are output to std::cout, they will be displayed with up to 6 digits after the decimal point.
        std::cout.width(12);                           //Sets the minimum field width to 12 characters
        pcout << std::string(96, '*') << std::endl
            << "Time step = " << time.get_timestep()
            << ", at t = " << std::scientific << time.current() << std::endl;
        // Resetting
        solution_increment = 0;
        // Only use nonzero constraints at the very first time step
        bool apply_nonzero_constraints = (time.get_timestep() == 1);
        // We have to assemble the LHS for the initial two time steps:
        // once using nonzero_constraints, once using zero_constraints,
        // as well as the steps imediately after mesh refinement.
        bool assemble_system = (time.get_timestep() < 3 || refined);
        refined = false;
        assemble(apply_nonzero_constraints, assemble_system);
        auto state = solve(apply_nonzero_constraints, assemble_system);
        // Note we have to use a non-ghosted vector to do the addition.
        PETScWrappers::MPI::BlockVector tmp;
        tmp.reinit(owned_partitioning, mpi_communicator);
        tmp = present_solution;
        tmp += solution_increment;
        present_solution = tmp;
        pcout << std::scientific << std::left << " GMRES_ITR = " << std::setw(3)
            << state.first << " GMRES_RES = " << state.second << std::endl;
        // Output
        if (time.time_to_output())
        {
            output_results(time.get_timestep());
        }
        if (time.time_to_refine())
        {
            refine_mesh(0, 4);
            refined = true;
        }
    }
}

// @sect4{InsIMEX::output_result}
//
template <int dim>
void InsIMEX<dim>::output_results(const unsigned int output_index) const
{
    TimerOutput::Scope timer_section(timer, "Output results");
    pcout << "Writing results..." << std::endl;
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;                              //This class is the main class to provide output of data described by finite element fields defined on a collection of cells.
    data_out.attach_dof_handler(dof_handler);
    // vector to be output must be ghosted
    data_out.add_data_vector(present_solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);

    // Partition
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
        subdomain(i) = triangulation.locally_owned_subdomain();           //For distributed parallel triangulations this function returns the subdomain id of those cells that are owned by the current processor
    }
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(degree + 1);

    std::string basename =
    "navierstokes" + Utilities::int_to_string(output_index, 6) + "-";

    std::string filename =
    basename +
    Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
        for (unsigned int i = 0;
            i < Utilities::MPI::n_mpi_processes(mpi_communicator);
            ++i)
        {
            times_and_names.push_back(
            {time.current(),
            basename + Utilities::int_to_string(i, 4) + ".vtu"});
        }
        std::ofstream pvd_output("navierstokes.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
    }
}

// @sect4{InsIMEX::refine_mesh}
//
template <int dim>
void InsIMEX<dim>::refine_mesh(const unsigned int min_grid_level,
                                const unsigned int max_grid_level)
{
    TimerOutput::Scope timer_section(timer, "Refine mesh");
    pcout << "Refining mesh..." << std::endl;

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                    face_quad_formula,
                                    {},
                                    present_solution,
                                    estimated_error_per_cell,
                                    fe.component_mask(velocity));
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
    triangulation, estimated_error_per_cell, 0.6, 0.4);
    if (triangulation.n_levels() > max_grid_level)
    {
        for (auto cell = triangulation.begin_active(max_grid_level);
            cell != triangulation.end();
            ++cell)
        {
            cell->clear_refine_flag();
        }
    }
    for (auto cell = triangulation.begin_active(min_grid_level);
        cell != triangulation.end_active(min_grid_level);
        ++cell)
    {
        cell->clear_coarsen_flag();
    }

    // Prepare to transfer
    parallel::distributed::SolutionTransfer<dim,
                                            PETScWrappers::MPI::BlockVector>
    trans(dof_handler);

    triangulation.prepare_coarsening_and_refinement();

    trans.prepare_for_coarsening_and_refinement(present_solution);

    // Refine the mesh
    triangulation.execute_coarsening_and_refinement();

    // Reinitialize the system
    setup_dofs();
    make_constraints();
    initialize_system();

    // Transfer solution
    // Need a non-ghosted vector for interpolation
    PETScWrappers::MPI::BlockVector tmp(solution_increment);
    tmp = 0;
    trans.interpolate(tmp);
    present_solution = tmp;
}
